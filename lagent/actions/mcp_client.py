import asyncio
import random
import re
import threading
import time
import warnings
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack, nullcontext
from typing import Literal, Optional, TypeAlias

from lagent.actions.base_action import AsyncActionMixin, BaseAction
from lagent.actions.parser import JsonParser, ParseError
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.utils import get_logger
from lagent.utils.rate_limiter import FairAsyncTokenBucket, get_shared_async_token_bucket

ServerType: TypeAlias = Literal['stdio', 'sse', 'http']

logger = get_logger()
_loop = None

warnings.filterwarnings('ignore', category=ResourceWarning, module=r'aiohttp\.client')
warnings.filterwarnings('ignore', category=ResourceWarning, module=r'anyio\._backends\._asyncio')

_failure_log_lock = threading.Lock()
_failure_log_state: dict[tuple[str, str, str], tuple[float, int]] = {}
_FAILURE_LOG_INTERVAL_S = 60.0
_FAILURE_LOG_DETAIL_LIMIT = 1200
_CANCEL_SCOPE_RE = re.compile(r'(cancel scope) [0-9a-fA-F]+')
_HEX_ADDRESS_RE = re.compile(r'0x[0-9a-fA-F]+')


def _normalize_failure_detail(detail: str) -> str:
    detail = _CANCEL_SCOPE_RE.sub(r'\1 <id>', detail)
    return _HEX_ADDRESS_RE.sub('0x<id>', detail)


def _format_failure_detail(exc: Exception) -> tuple[str, str]:
    parts = [f"{type(exc).__name__}: {exc}"]
    type_parts = [type(exc).__name__]

    # Unwrap ExceptionGroup so the actual sub-exception is visible (anyio TaskGroup
    # wraps connect/read errors into ExceptionGroup which str()s to a useless summary).
    pending = list(getattr(exc, 'exceptions', ()) or ())
    while pending:
        sub = pending.pop(0)
        parts.append(f"  -> {type(sub).__name__}: {sub}")
        type_parts.append(type(sub).__name__)
        pending.extend(getattr(sub, 'exceptions', ()) or ())
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        parts.append(f"  caused by {type(cause).__name__}: {cause}")
        type_parts.append(f"caused_by:{type(cause).__name__}")

    detail = _normalize_failure_detail('\n'.join(parts))
    if len(detail) > _FAILURE_LOG_DETAIL_LIMIT:
        detail = f"{detail[:_FAILURE_LOG_DETAIL_LIMIT]}\n  ... truncated"
    return detail, '|'.join(type_parts)


def _log_action_failure(action_name: str, exc: Exception) -> None:
    detail, type_signature = _format_failure_detail(exc)
    key = (action_name, type_signature, detail[:256])
    now = time.monotonic()
    with _failure_log_lock:
        last, suppressed = _failure_log_state.get(key, (0.0, 0))
        if now - last < _FAILURE_LOG_INTERVAL_S:
            _failure_log_state[key] = (last, suppressed + 1)
            return
        _failure_log_state[key] = (now, 0)
    if suppressed:
        logger.warning(
            'MCP Action %s failed (%d similar failures suppressed in the last %.0fs):\n%s',
            action_name,
            suppressed,
            _FAILURE_LOG_INTERVAL_S,
            detail,
        )
    else:
        logger.warning('MCP Action %s failed:\n%s', action_name, detail)


def _get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except Exception:
        logger.warning('Can not found event loop in current thread. Create a new event loop.')
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

    if event_loop.is_running():
        global _loop
        if _loop:
            return _loop

        from threading import Thread

        def _start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        event_loop = asyncio.new_event_loop()
        Thread(target=_start_loop, args=(event_loop,), daemon=True).start()
        _loop = event_loop
    return event_loop


class AsyncMCPClient(AsyncActionMixin, BaseAction):
    """
    Standard Lagent Action that wraps a SINGLE tool from an MCP Server.

    Robustness Fix:
    Creates a new connection for every request and closes it immediately after.
    This prevents connection leaks and 'ConnectTimeout' in high-concurrency RL environments.
    """

    def __init__(
        self,
        server_type: ServerType,
        rate_limit: float = None,
        rate_limit_key: Optional[str] = None,
        rate_limit_burst: Optional[float] = None,
        max_concurrency: int = None,
        metadata_timeout: Optional[float] = 60.0,
        call_timeout: Optional[float] = 300.0,
        description: Optional[dict] = None,
        # name 保留给旧配置；实际工具名优先来自 description 或 MCP metadata
        name: Optional[str] = None,
        extra_args: Optional[dict] = None,
        **server_params,
    ):
        self._is_toolkit = False
        self.server_type = server_type
        self.server_params = server_params
        self.extra_args = extra_args or {}
        self.metadata_timeout = (
            float(metadata_timeout) if metadata_timeout is not None and metadata_timeout > 0 else None
        )
        self.call_timeout = float(call_timeout) if call_timeout is not None and call_timeout > 0 else None

        # 并发控制组件
        if rate_limit is None:
            self.rate_limiter = None
        elif rate_limit_key:
            self.rate_limiter = get_shared_async_token_bucket(rate_limit_key, rate_limit, rate_limit_burst)
        else:
            self.rate_limiter = FairAsyncTokenBucket(rate_limit, capacity=rate_limit_burst)
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()

        if description is None:
            # 临时连接获取工具元数据 (Metadata)。必须在 __init__ 完成，因为 Lagent 需要 self.description。
            loop = _get_event_loop()
            if loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self._fetch_tool_metadata(), loop)
                try:
                    tools = fut.result(timeout=self.metadata_timeout)
                except FutureTimeoutError as exc:
                    fut.cancel()
                    raise TimeoutError(f"Timed out fetching MCP tool metadata after {self.metadata_timeout}s") from exc
            else:
                metadata_task = self._fetch_tool_metadata()
                if self.metadata_timeout is not None:
                    metadata_task = asyncio.wait_for(metadata_task, timeout=self.metadata_timeout)
                tools = loop.run_until_complete(metadata_task)

            # Single Action 约束：一个 Action 实例对应一个 MCP 工具
            if not tools:
                raise RuntimeError('MCP Server returned no tools.')
            if len(tools) != 1:
                logger.warning(
                    f"MCP Server returned {len(tools)} tools, but AsyncMCPAction is designed for a Single Action. "
                    f"Using the first one: {tools[0].name}"
                )

            self.tool_info = tools[0]
            description = {
                'name': self.tool_info.name,
                'description': self.tool_info.description,
                'parameters': [
                    {'name': k, 'type': v['type'].upper(), 'description': v.get('description', '')}
                    for k, v in self.tool_info.inputSchema['properties'].items()
                    if k not in self.extra_args
                ],
                'required': self.tool_info.inputSchema.get('required', []),
            }
        else:
            description = dict(description)
            if 'name' not in description:
                if name is None:
                    raise ValueError('Static MCP action description must include a tool name.')
                description['name'] = name
            self.tool_info = None

        self.tool_name = description['name']
        if self.extra_args:
            description = {
                **description,
                'parameters': [
                    param for param in description.get('parameters', []) if param.get('name') not in self.extra_args
                ],
                'required': [item for item in description.get('required', []) if item not in self.extra_args],
            }

        # 2. 初始化父类 BaseAction
        super().__init__(
            description=description,
            parser=JsonParser,
        )
        self._is_toolkit = False

    async def _connect(self, stack: AsyncExitStack):
        """
        内部辅助：建立连接并注册关闭回调。
        所有网络资源都注册到 `stack` 中，确保自动释放。
        """
        from mcp import ClientSession, StdioServerParameters

        # --- Transport Layer ---
        if self.server_type == 'stdio':
            from mcp.client.stdio import stdio_client

            logger.info(
                f"Connecting to stdio MCP server with command: {self.server_params['command']} "
                f"{self.server_params.get('args', [])}"
            )
            client_kwargs = {'command': self.server_params['command']}
            for key in ['args', 'env', 'cwd']:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]

            server_params_obj = StdioServerParameters(**client_kwargs)
            read, write = await stack.enter_async_context(stdio_client(server_params_obj))

        elif self.server_type == 'sse':
            from mcp.client.sse import sse_client

            logger.info(f"Connecting to SSE MCP server at: {self.server_params['url']}")

            url = self.server_params['url']
            target_url = random.choice(url) if isinstance(url, list) else url

            client_kwargs = {'url': target_url}
            for key in ['headers', 'timeout', 'sse_read_timeout']:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]

            read, write = await stack.enter_async_context(sse_client(**client_kwargs))

        elif self.server_type == 'http':
            from mcp.client.streamable_http import streamablehttp_client

            # logger.debug(f"Connecting to StreamableHTTP MCP server at: {self.server_params['url']}")

            url = self.server_params['url']
            target_url = random.choice(url) if isinstance(url, list) else url

            client_kwargs = {'url': target_url}
            for key in ['headers', 'timeout', 'sse_read_timeout', 'terminate_on_close']:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]

            read, write, _ = await stack.enter_async_context(streamablehttp_client(**client_kwargs))

        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")

        # --- Protocol Layer ---
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _fetch_tool_metadata(self):
        """在 init 阶段使用一次性连接获取工具定义"""
        async with AsyncExitStack() as stack:
            session = await self._connect(stack)
            result = await session.list_tools()
            return result.tools

    async def _call_tool_once(self, kwargs: dict):
        async with AsyncExitStack() as stack:
            session = await self._connect(stack)
            outputs_obj = await session.call_tool(self.tool_name, {**kwargs, **self.extra_args})

            if outputs_obj.content and hasattr(outputs_obj.content[0], 'text'):
                return outputs_obj.content[0].text
            return str(outputs_obj)

    async def run(self, **kwargs) -> ActionReturn:
        """
        Standard Lagent Action Entrypoint.
        """
        fallback_args = kwargs.copy()

        try:
            # 1. 并发/速率控制
            async with self._sem:
                if self.rate_limiter is not None:
                    await self.rate_limiter.acquire()

                call_task = self._call_tool_once(kwargs)
                if self.call_timeout is not None:
                    call_task = asyncio.wait_for(call_task, timeout=self.call_timeout)
                outputs = await call_task

        except ParseError as exc:
            return ActionReturn(fallback_args, type=self.name, errmsg=exc.err_msg, state=ActionStatusCode.ARGS_ERROR)
        except Exception as exc:
            _log_action_failure(self.name, exc)
            return ActionReturn(fallback_args, type=self.name, errmsg=str(exc), state=ActionStatusCode.API_ERROR)

        # 3. 结果封装
        if isinstance(outputs, ActionReturn):
            action_return = outputs
            if not action_return.args:
                action_return.args = kwargs
            if not action_return.type:
                action_return.type = self.name
        else:
            # 尝试使用 JsonParser 解析结果（如果 MCP 返回的是 JSON 字符串）
            # 否则直接作为字符串返回
            try:
                result = self._parser.parse_outputs(outputs)
            except Exception as exc:
                logger.warning(f"Failed to parse MCP Action {self.name} output: {exc}")
                result = str(outputs)

            action_return = ActionReturn(fallback_args, type=self.name, result=result)

        return action_return
