import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Literal, TypeAlias

from lagent.actions.base_action import BaseAction
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


class AsyncMCPClient(BaseAction):
    """Model Context Protocol (MCP) Client for asynchronous communication with MCP servers.

    Args:
        name (str): The name of the action. Make sure it is unique among all actions.
        server_type (ServerType): The type of MCP server to connect to. Options are "stdio", "sse", or "http".
        **server_params: Additional parameters for the server connection, which may include:
          - For stdio servers:
            - command (str): The command to run the MCP server.
            - args (list, optional): Additional arguments for the command.
            - env (dict, optional): Environment variables for the command.
            - cwd (str, optional): Current working directory for the command.
          - For sse servers:
            - url (str): The URL of the MCP server.
            - headers (dict, optional): Headers to include in the request.
            - timeout (int, optional): Timeout for the request.
            - sse_read_timeout (int, optional): Timeout for reading SSE events.
          - For http servers:
            - url (str): The URL of the MCP server.
            - headers (dict, optional): Headers to include in the request.
            - timeout (int, optional): Timeout for the request.
            - sse_read_timeout (int, optional): Timeout for reading SSE events.
            - terminate_on_close (bool, optional): Whether to terminate the connection on close.
    """

    is_stateful = True

    def __init__(self, name: str, server_type: ServerType, **server_params):
        self._is_toolkit = True
        self._sessions: dict = {}
        self.server_type = server_type
        self.server_params = server_params
        self.exit_stack = AsyncExitStack()
        # get the list of tools from the MCP server
        loop = _get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.list_tools(), loop)
            tools = fut.result()
        else:
            tools = loop.run_until_complete(self.list_tools())
        self._api_names = {tool.name for tool in tools}
        super().__init__(
            description=dict(
                name=name,
                api_list=[
                    {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': [
                            {'name': k, 'type': v['type'].upper(), 'description': v.get('description', '')}
                            for k, v in tool.inputSchema['properties'].items()
                        ],
                        'required': tool.inputSchema.get('required', []),
                    }
                    for tool in tools
                ],
            ),
            parser=JsonParser,
        )

    async def initialize(self, session_id):
        """Initialize the MCP client and connect to the server."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        from mcp import ClientSession, StdioServerParameters

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
            server_params = StdioServerParameters(**client_kwargs)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        elif self.server_type == "sse":
            from mcp.client.sse import sse_client

            logger.info(f"Connecting to SSE MCP server at: {self.server_params['url']}")

            client_kwargs = {"url": self.server_params["url"]}
            for key in ["headers", "timeout", "sse_read_timeout"]:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]
            read, write = await self.exit_stack.enter_async_context(sse_client(**client_kwargs))
        elif self.server_type == "http":
            from mcp.client.streamable_http import streamablehttp_client

            logger.info(f"Connecting to StreamableHTTP MCP server at: {self.server_params['url']}")

            client_kwargs = {"url": self.server_params["url"]}
            for key in ["headers", "timeout", "sse_read_timeout", "terminate_on_close"]:
                if self.server_params.get(key) is not None:
                    client_kwargs[key] = self.server_params[key]
            read, write, _ = await self.exit_stack.enter_async_context(streamablehttp_client(**client_kwargs))
        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")

        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[session_id] = session
        return session

    async def cleanup(self):
        await self.exit_stack.aclose()

    async def list_tools(self, session_id=0) -> list:
        session = await self.initialize(session_id=session_id)
        return (await session.list_tools()).tools

    def __del__(self):
        loop = _get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.cleanup(), loop)
            fut.result()
        else:
            loop.run_until_complete(self.cleanup())

    async def __call__(self, inputs: str, name: str) -> ActionReturn:
        session_id = inputs.pop('session_id', 0) if isinstance(inputs, dict) else 0
        fallback_args = {'inputs': inputs, 'name': name}
        if name not in self._api_names:
            return ActionReturn(
                fallback_args, type=self.name, errmsg=f'invalid API: {name}', state=ActionStatusCode.API_ERROR
            )
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except ParseError as exc:
            return ActionReturn(fallback_args, type=self.name, errmsg=exc.err_msg, state=ActionStatusCode.ARGS_ERROR)
        try:
            session = await self.initialize(session_id)
            outputs = await session.call_tool(name, inputs)
            outputs = outputs.content[0].text
        except Exception as exc:
            return ActionReturn(inputs, type=self.name, errmsg=str(exc), state=ActionStatusCode.API_ERROR)
        if isinstance(outputs, ActionReturn):
            action_return = outputs
            if not action_return.args:
                action_return.args = inputs
            if not action_return.type:
                action_return.type = self.name
        else:
            result = self._parser.parse_outputs(outputs)
            action_return = ActionReturn(inputs, type=self.name, result=result)
        return action_return
