import asyncio
import logging
import random
import threading
import time
from collections import deque
from contextlib import AsyncExitStack, nullcontext
from typing import Deque, Literal, Optional, TypeAlias

from lagent.actions.base_action import AsyncActionMixin, BaseAction
from lagent.actions.parser import JsonParser, ParseError
from lagent.schema import ActionReturn, ActionStatusCode

ServerType: TypeAlias = Literal["stdio", "sse", "http"]

logger = logging.getLogger(__name__)
_loop = None


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


class TokenBucket:
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit  # tokens per second
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            # Add new tokens based on time elapsed
            new_tokens = (now - self.last_update) * self.rate_limit
            self.tokens = min(self.rate_limit, self.tokens + new_tokens)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class AsyncTokenBucket:
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit
        self.capacity = rate_limit
        self.tokens = rate_limit
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_limit)
        self.last_update = now

    async def acquire(self):
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                missing = 1 - self.tokens
                wait_time = missing / self.rate_limit
            await asyncio.sleep(wait_time)


class FairAsyncTokenBucket:
    def __init__(self, rate_limit: float, capacity: Optional[float] = None):
        """
        rate_limit: 每秒生成多少个 token
        capacity: 桶容量（最大可累积多少 token），默认和 rate_limit 一样
        """
        self.rate_limit = float(rate_limit)
        self.capacity = float(capacity) if capacity is not None else float(rate_limit)

        self.tokens = self.capacity
        self.last_update = time.monotonic()

        self._lock = asyncio.Lock()
        self._waiters: Deque[asyncio.Future] = deque()
        self._drainer_running = False  # 是否已有后台协程在发 token

    # ---------- 内部工具方法 ----------

    def _refill_unlocked(self) -> None:
        """
        在不持锁的前提下不要调用。
        根据时间流逝计算当前 token 数。
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_limit)
        self.last_update = now

    async def _drain_waiters(self) -> None:
        """
        后台协程：按 FIFO 顺序给排队的协程发 token。
        - 没 token 时，就 sleep 到下一个 token 产生的时间点。
        - 有 token 且有排队，就唤醒队头的一个，再继续循环。
        """
        try:
            while True:
                fut_to_wake: Optional[asyncio.Future] = None
                sleep_time: Optional[float] = None

                async with self._lock:
                    self._refill_unlocked()

                    # 队列空了，没什么好做的了，退出 drainer
                    if not self._waiters:
                        self._drainer_running = False
                        return

                    if self.tokens >= 1:
                        # 有 token，按 FIFO 唤醒一个排队的协程
                        self.tokens -= 1
                        fut_to_wake = self._waiters.popleft()
                        sleep_time = 0.0
                    else:
                        # 没 token，算一下距离下一个 token 的时间
                        missing = 1.0 - self.tokens  # 还差多少 token 才能发下一枚
                        sleep_time = max(0.0, missing / self.rate_limit)

                # 出锁之后再唤醒，避免在锁里执行用户代码 / 回调
                if fut_to_wake is not None and not fut_to_wake.done():
                    fut_to_wake.set_result(None)

                # 如果刚刚唤醒了一个协程，立刻回到循环，看是否还能继续发
                if sleep_time == 0.0:
                    continue

                # 没 token，就等到有 token 再继续
                await asyncio.sleep(sleep_time)
        finally:
            # 兜底，避免异常时 drainer_running 一直是 True 导致无法重启
            async with self._lock:
                self._drainer_running = False

    # ---------- 对外接口 ----------

    async def acquire(self) -> None:
        """
        获取一个 token（公平：排队 FIFO）
        """
        loop = asyncio.get_running_loop()

        # 先尝试直接拿 token（快速路径）
        async with self._lock:
            self._refill_unlocked()

            # 如果有 token 且没有历史排队的协程，直接拿走返回
            if self.tokens >= 1 and not self._waiters:
                self.tokens -= 1
                return

            # 否则需要排队
            fut = loop.create_future()
            self._waiters.append(fut)

            # 启动 drainer（只要一个就够了）
            if not self._drainer_running:
                self._drainer_running = True
                asyncio.create_task(self._drain_waiters())

        # 等待被 drainer 唤醒，唤醒后说明自己拿到了 token
        await fut


# --- 复用你原本的辅助工具 ---
_loop = None


def _get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except Exception:
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

    is_stateful = False

    def __init__(
        self,
        server_type: ServerType,
        rate_limit: float = None,
        max_concurrency: int = None,
        # 注意：这里的 name 主要用于 Lagent 注册，但工具的实际元数据来自 MCP Server
        name: Optional[str] = None,
        extra_args: Optional[dict] = None,
        **server_params,
    ):
        self._is_toolkit = False
        self.server_type = server_type
        self.server_params = server_params
        self.extra_args = extra_args or {}

        # 并发控制组件
        self.rate_limiter = FairAsyncTokenBucket(rate_limit) if rate_limit is not None else None
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()

        # 1. 临时连接获取工具元数据 (Metadata)
        # 必须在 __init__ 完成，因为 Lagent 需要 self.description
        loop = _get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._fetch_tool_metadata(), loop)
            tools = fut.result()
        else:
            tools = loop.run_until_complete(self._fetch_tool_metadata())

        # Single Action 约束：一个 Action 实例对应一个 MCP 工具
        if len(tools) != 1:
            logger.warning(
                f"MCP Server returned {len(tools)} tools, but AsyncMCPAction is designed for a Single Action. "
                f"Using the first one: {tools[0].name}"
            )

        self.tool_info = tools[0]
        tool_name = self.tool_info.name

        # 2. 初始化父类 BaseAction
        super().__init__(
            description={
                'name': tool_name,
                'description': self.tool_info.description,
                'parameters': [
                    {'name': k, 'type': v['type'].upper(), 'description': v.get('description', '')}
                    for k, v in self.tool_info.inputSchema['properties'].items()
                    if k not in self.extra_args
                ],
                'required': self.tool_info.inputSchema.get('required', []),
            },
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
        if self.server_type == "stdio":
            from mcp.client.stdio import stdio_client

            logger.info(
                f"Connecting to stdio MCP server with command: {self.server_params['command']} "
                f"{self.server_params.get('args', [])}"
            )
            client_kwargs = {"command": self.server_params["command"]}
            for key in ["args", "env", "cwd"]:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]

            server_params_obj = StdioServerParameters(**client_kwargs)
            read, write = await stack.enter_async_context(stdio_client(server_params_obj))

        elif self.server_type == "sse":
            from mcp.client.sse import sse_client

            logger.info(f"Connecting to SSE MCP server at: {self.server_params['url']}")

            url = self.server_params["url"]
            target_url = random.choice(url) if isinstance(url, list) else url

            client_kwargs = {"url": target_url}
            for key in ["headers", "timeout", "sse_read_timeout"]:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]

            read, write = await stack.enter_async_context(sse_client(**client_kwargs))

        elif self.server_type == "http":
            from mcp.client.streamable_http import streamablehttp_client

            logger.info(f"Connecting to StreamableHTTP MCP server at: {self.server_params['url']}")

            url = self.server_params["url"]
            target_url = random.choice(url) if isinstance(url, list) else url

            client_kwargs = {"url": target_url}
            for key in ["headers", "timeout", "sse_read_timeout", "terminate_on_close"]:
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

                # 2. 执行逻辑 (Critical Resource Scope)
                # 使用 AsyncExitStack 确保本次请求结束后，HTTP连接/进程管道被彻底关闭
                async with AsyncExitStack() as stack:
                    session = await self._connect(stack)

                    # 调用 MCP 工具
                    # 注意：Lagent 传入的是 kwargs 字典，MCP call_tool 正好接受字典
                    outputs_obj = await session.call_tool(self.tool_info.name, {**kwargs, **self.extra_args})

                    # 提取文本结果
                    if outputs_obj.content and hasattr(outputs_obj.content[0], 'text'):
                        outputs = outputs_obj.content[0].text
                    else:
                        outputs = str(outputs_obj)

        except ParseError as exc:
            return ActionReturn(fallback_args, type=self.name, errmsg=exc.err_msg, state=ActionStatusCode.ARGS_ERROR)
        except Exception as exc:
            # 记录详细堆栈以便调试 RL 过程中的错误
            logger.warning(f"MCP Action {self.name} failed: {exc}")
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
