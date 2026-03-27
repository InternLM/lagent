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


logger = logging.getLogger(__file__)


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
        **server_params,
    ):
        self._is_toolkit = False
        self.server_type = server_type
        self.server_params = server_params

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
        description = {
                'name': tool_name,
                'description': self.tool_info.description,
                'parameters': [
                    {'name': k, 'type': v['type'].upper(), 'description': v.get('description', '')}
                    for k, v in self.tool_info.inputSchema['properties'].items()
                ],
                'required': self.tool_info.inputSchema.get('required', []),
            }
        if self.is_stateful:
            description['parameters'].append({'name': 'session_id', 'type': 'STRING', 'description': 'session id'})
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
                    outputs_obj = await session.call_tool(self.tool_info.name, kwargs)

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
            except:
                result = str(outputs)

            action_return = ActionReturn(fallback_args, type=self.name, result=result)

        return action_return



class AsyncMCPClientStatefulAction(AsyncMCPClient):
    """
    Stateful Lagent Action that wraps a SINGLE tool from an MCP Server.
    
    Maintains a persistent connection per session_id to support stateful tools
    like a persistent shell.
    """

    is_stateful = True

    def __init__(
        self,
        server_type: ServerType,
        rate_limit: float = None,
        max_concurrency: int = None,
        name: Optional[str] = None,
        init_dir: Optional[str] = None,
        **server_params,
    ):
        # 1. 初始化父类（会自动获取 metadata 并初始化 BaseAction）
        super().__init__(
            server_type=server_type,
            rate_limit=rate_limit,
            max_concurrency=max_concurrency,
            name=name,
            **server_params
        )

        # 2. 初始化 Session 池
        self.init_dir = init_dir
        self._session_clients = {}
        self._session_stacks = {}
        self._lock = asyncio.Lock()

    async def get_or_create_client(self, session_id: str):
        """获取或创建与 session_id 绑定的长连接"""
        session_id = str(session_id)
        async with self._lock:
            if session_id not in self._session_clients:
                logger.info(f"Creating new MCP connection for session {session_id}")

                # anyio 严格要求进入和退出 CancelScope 必须在同一个 Task 中。
                # 为了解决这个问题，我们创建一个后台长期运行的 Task 来专门负责这个连接的生命周期。
                # 通过 asyncio.Queue 来实现主 Task 和后台连接 Task 之间的通信。

                request_queue = asyncio.Queue()
                response_queue = asyncio.Queue()

                async def _connection_worker():
                    try:
                        async with AsyncExitStack() as stack:
                            session = await self._connect(stack)
                            # 告知主 Task 连接成功
                            await response_queue.put(session)

                            # 循环等待主 Task 发送关闭信号
                            while True:
                                msg = await request_queue.get()
                                if msg == "close":
                                    break
                    except Exception as e:
                        # 告知主 Task 连接失败
                        await response_queue.put(e)

                # 启动后台 Task
                worker_task = asyncio.create_task(_connection_worker())

                # 等待连接建立完成
                result = await response_queue.get()
                if isinstance(result, Exception):
                    raise result

                self._session_clients[session_id] = result
                self._session_stacks[session_id] = (request_queue, worker_task)

                # 如果配置了 init_dir，则在连接建立后立即初始化
                if self.init_dir:
                    await self._initialize_session_dir(session_id, result)

            return self._session_clients[session_id]

    async def _initialize_session_dir(self, session_id: str, session):
        """将本地目录打包并同步到远程 Session"""
        import tarfile
        import io
        import base64
        import os

        if not self.init_dir or not os.path.exists(self.init_dir):
            logger.warning(f"Init dir {self.init_dir} not found or not set, skipping initialization.")
            return

        logger.info(f"Initializing session {session_id} with directory {self.init_dir}")

        # 1. 在内存中打包目录
        buf = io.BytesIO()
        dir_name = os.path.basename(os.path.normpath(self.init_dir))
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            # 将目录内容添加到 tar，arcname 使用目录本身的名字，保留外层文件夹
            tar.add(self.init_dir, arcname=dir_name)

        # 2. 转为 Base64 字符串
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 3. 构造解压命令
        # 注意：这里假设 tool_info.name 是 run_command 且接受 command 参数
        init_cmd = f"echo '{encoded}' | base64 -d | tar -xz"

        try:
            # 直接调用 session 执行初始化命令
            await session.call_tool(self.tool_info.name, {"command": init_cmd})
            logger.info(f"Session {session_id} initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {e}")

    async def close_session(self, session_id: str):
        """关闭指定 session 的连接，释放服务端资源"""
        session_id = str(session_id)
        async with self._lock:
            if session_id in self._session_stacks:
                logger.info(f"Closing MCP connection for session {session_id}")
                request_queue, worker_task = self._session_stacks.pop(session_id)
                self._session_clients.pop(session_id, None)
                
                # 发送关闭信号给后台 Task
                await request_queue.put("close")
                
                # 等待后台 Task 优雅退出
                try:
                    await worker_task
                except Exception as e:
                    logger.warning(f"Error while closing MCP session {session_id}: {e}")

    async def run(self, session_id: Optional[str] = None, **kwargs) -> ActionReturn:
        """
        Standard Lagent Action Entrypoint for stateful execution.
        """
        fallback_args = kwargs.copy()
        
        # 兜底：如果外部没有传入 session_id，使用默认会话
        if session_id is None:
            session_id = "default_session"

        try:
            # 1. 并发/速率控制
            async with self._sem:
                if self.rate_limiter is not None:
                    await self.rate_limiter.acquire()

                # 2. 从 Session 池中获取或创建当前 session 的长连接
                session = await self.get_or_create_client(session_id)

                # 调用 MCP 工具
                outputs_obj = await session.call_tool(self.tool_info.name, kwargs)

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
            try:
                result = self._parser.parse_outputs(outputs)
            except:
                result = str(outputs)

            action_return = ActionReturn(fallback_args, type=self.name, result=result)

        return action_return
if __name__ == '__main__':
    import asyncio
    from lagent.utils import create_object
    def get_tool_prompt(actions: list, exclude_arguments: list = None) -> str:
        from copy import deepcopy
        exclude_arguments = exclude_arguments or ['session_id']

        def _convert_tool_schema(action_description: dict, name_pattern: str = '{}') -> dict:
            properties = {}
            for param in action_description['parameters']:
                param = deepcopy(param)
                param_name, param_type = param.pop('name'), param.pop('type')
                if param_name in exclude_arguments:
                    continue
                param_type = [t.lower() for t in param_type] if isinstance(param_type, list) else param_type.lower()
                properties[param_name] = {'type': param_type, **param}
            return {
                'type': 'function',
                'function': {
                    'name': name_pattern.format(action_description['name']),
                    'description': action_description['description'],
                    'parameters': {'type': 'object', 'properties': properties, 'required': action_description['required']},
                },
            }

        tools = []
        for action in actions if isinstance(actions, list) else [actions]:
            action = create_object(action)
            action_desc = action.description
            if action.is_toolkit:
                for api in action_desc['api_list']:
                    tools.append(_convert_tool_schema(api, f"{action.name}.{{}}"))
            else:
                tools.append(_convert_tool_schema(action_desc))
        return tools

    action = AsyncMCPClientStatefulAction("http", url='http://simple-shell.ailab.ailab.ai/mcp',  init_dir="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace/")
    print(get_tool_prompt([action]))
    import asyncio
    import json
    async def test():
        res = await action.run(command='ls -a') 
        home_path = json.loads(res.result[0]['content'])['cwd']
        print(res)
        res = await action.run(command=f"python - <<'PY'\nfrom pathlib import Path\nimport json\nroot = Path('{home_path}/workspace/skills')\nitems = []\nif root.exists():\n    for d in root.iterdir():\n        skill = d / 'SKILL.md'\n        if d.is_dir() and skill.exists():\n            items.append({{'name': d.name, 'path': str(skill), 'source': 'workspace'}})\nprint(json.dumps(items, ensure_ascii=False))\nPY")
        print(res)
    asyncio.run(test())
