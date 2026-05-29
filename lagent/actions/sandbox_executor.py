"""SandboxActionExecutor — drop-in replacement for ``AsyncActionExecutor``
that routes action calls to an :class:`~lagent.serving.sandbox.daemon.ActionDaemon`
running inside a remote sandbox.

Communication goes through the sandbox's bash execution channel::

    SandboxActionExecutor.forward("shell", {"command": "ls"})
      → sandbox_client.execute('python -m lagent.serving.sandbox.daemon call ...')
      → daemon inside sandbox executes ShellAction locally
      → JSON result flows back through stdout

Usage::

    from lagent.serving.sandbox.executor import SandboxActionExecutor

    executor = SandboxActionExecutor(
        sandbox_client=sandbox_client,  # your SandboxClient instance
        actions_config=[
            {"type": "lagent.actions.shell.ShellAction"},
            {"type": "lagent.actions.ipython_interpreter.AsyncIPythonInterpreter"},
        ],
    )
    await executor.connect()
    # Now use as AsyncActionExecutor

The ``sandbox_client`` can be **sync or async**.  If ``execute()`` is a
regular (sync) method it will be called via ``asyncio.to_thread`` so it
never blocks the event loop.  The return value may be a ``str`` (raw
stdout) or a ``dict`` with a ``"stdout"`` key — both are handled.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

from lagent.actions.builtin_actions import FinishAction, InvalidAction, NoAction
from lagent.hooks import Hook, RemovableHandle
from lagent.schema import (
    ActionReturn,
    ActionStatusCode,
    ActionValidCode,
    AgentMessage,
    FunctionCall,
)
from lagent.utils import create_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deserialize_action_return(text: str) -> ActionReturn:
    """Deserialize a JSON string back into an ActionReturn."""
    data = json.loads(text)
    if data.get("state") is not None:
        data["state"] = ActionStatusCode(data["state"])
    if data.get("valid") is not None:
        data["valid"] = ActionValidCode(data["valid"])
    return ActionReturn(**data)


class _ToolDescriptionStub:
    """Minimal stub that quacks like a BaseAction for ``get_tool_prompt()``.

    ``create_object()`` returns non-dict inputs as-is, so these stubs
    pass through correctly.
    """

    def __init__(self, desc: dict):
        self._desc = desc

    @property
    def name(self) -> str:
        return self._desc["name"]

    @property
    def is_toolkit(self) -> bool:
        return False

    @property
    def description(self) -> dict:
        return self._desc


class _ToolDescriptionProxy:
    """Makes ``executor.actions`` behave like ``Dict[str, BaseAction]``
    for code that iterates ``.values()`` and reads ``.name`` / ``.description``.
    """

    def __init__(self, descriptions: Dict[str, dict]):
        self._descriptions = descriptions

    def __contains__(self, name: str) -> bool:
        return name in self._descriptions

    def values(self):
        return [_ToolDescriptionStub(d) for d in self._descriptions.values()]

    def keys(self):
        return list(self._descriptions.keys())

    def items(self):
        return [(k, _ToolDescriptionStub(v)) for k, v in self._descriptions.items()]

    def __getitem__(self, key: str):
        return _ToolDescriptionStub(self._descriptions[key])


# ---------------------------------------------------------------------------
# SandboxActionExecutor
# ---------------------------------------------------------------------------


class SandboxActionExecutor:
    """Drop-in replacement for ``AsyncActionExecutor`` that routes action
    calls to an :class:`ActionDaemon` running inside a sandbox.

    Parameters
    ----------
    sandbox_client
        Object with an ``execute(command, ...)`` method that runs bash
        inside the sandbox.  May be sync or async.  The return value can
        be a plain ``str`` (stdout) **or** a ``dict`` with a ``"stdout"``
        key (like the real ``SandboxClient.execute``).
    actions_config : list of dict
        Action configurations (same format as ActionDaemon's config file).
    sock_path : str
        Unix socket path inside the sandbox.
    cwd : str
        Working directory for bash commands inside the sandbox.
    daemon_module : str
        Python module path for the daemon entry point.
    """

    def __init__(
        self,
        sandbox_client,
        actions_config: List[Dict],
        sock_path: str = "/tmp/lagent_action.sock",
        cwd: str = "/root",
        daemon_module: str = "lagent.serving.sandbox.daemon",
        invalid_action=dict(type=InvalidAction),
        no_action=dict(type=NoAction),
        finish_action=dict(type=FinishAction),
        hooks: Optional[List[Dict]] = None,
    ):
        self.sandbox_client = sandbox_client
        self.actions_config = actions_config
        self.sock_path = sock_path
        self.cwd = cwd
        self.daemon_module = daemon_module

        # Built-in actions handled locally
        self.invalid_action = create_object(invalid_action)
        self.no_action = create_object(no_action)
        self.finish_action = create_object(finish_action)

        # Hooks
        self._hooks: Dict[int, Hook] = OrderedDict()
        if hooks:
            for hook in hooks:
                hook = create_object(hook)
                self.register_hook(hook)

        # Connection state
        self._connected = False
        self._lock = asyncio.Lock()

        # Tool descriptions (populated on connect)
        self._tool_descriptions: Dict[str, dict] = {}

    # -- sandbox communication helpers --

    async def _exec(self, command: str, **kwargs) -> str:
        """Execute a command via the sandbox client.

        Handles both sync and async clients, and normalises the return
        value to a plain stdout string.
        """
        kwargs.setdefault("cwd", self.cwd)
        execute_fn = self.sandbox_client.execute
        if inspect.iscoroutinefunction(execute_fn):
            result = await execute_fn(command, **kwargs)
        else:
            result = await asyncio.to_thread(execute_fn, command, **kwargs)

        # Normalise: dict with "stdout" → str
        if isinstance(result, dict):
            stdout = result.get("stdout", "")
            if not stdout.strip() and result.get("stderr", "").strip():
                raise RuntimeError(f"Command stderr: {result['stderr']}")
            return stdout
        return result

    # -- connection lifecycle --

    async def connect(self) -> None:
        """Start the daemon inside the sandbox (idempotent) and fetch tool list."""
        async with self._lock:
            if self._connected:
                return

            # 1. Write actions config to sandbox
            config_json = json.dumps(self.actions_config, ensure_ascii=False)
            escaped_config = config_json.replace("'", "'\\''")
            await self._exec(
                f"echo '{escaped_config}' > /tmp/lagent_actions_config.json"
            )

            # 2. Check if daemon already running, start if not
            check = await self._exec(
                f"pgrep -f '[l]agent.serving.sandbox.daemon.*--sock {self.sock_path}'"
                f" > /dev/null 2>&1 && echo 'running' || echo 'stopped'"
            )
            if "stopped" in check:
                await self._exec(
                    f"nohup python -m {self.daemon_module} start "
                    f"--sock {self.sock_path} "
                    f"--actions-config /tmp/lagent_actions_config.json "
                    f"> /tmp/lagent_daemon.log 2>&1 &"
                )

            # 3. Wait for socket to be ready
            for _ in range(30):
                try:
                    output = await self._exec(
                        f"test -S {self.sock_path} && echo 'ready' || echo 'waiting'"
                    )
                    if "ready" in output:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError(
                    f"ActionDaemon did not start within 15s. "
                    f"Check /tmp/lagent_daemon.log inside sandbox."
                )

            # 4. Fetch tool list from daemon
            result = await self._daemon_call({"cmd": "list_tools"})
            tools = result.get("tools", [])
            self._tool_descriptions = {t["name"]: t for t in tools}
            self._connected = True
            logger.info(
                "SandboxActionExecutor connected, %d tools: %s",
                len(tools),
                list(self._tool_descriptions.keys()),
            )

    async def close(self) -> None:
        """Shut down the daemon."""
        if not self._connected:
            return
        try:
            await self._daemon_call({"cmd": "shutdown"})
        except Exception:
            pass
        self._connected = False
        self._tool_descriptions.clear()
        logger.info("SandboxActionExecutor closed")

    # -- daemon communication --

    async def _daemon_call(self, request: dict, timeout_sec: int = 300) -> dict:
        """Send a JSON request to the daemon via bash and parse response."""
        request_json = json.dumps(request, ensure_ascii=False)
        escaped = request_json.replace("'", "'\\''")
        output = await self._exec(
            f"python -m {self.daemon_module} call "
            f"--sock {self.sock_path} "
            f"'{escaped}'",
            timeout_sec=timeout_sec,
        )
        if not output.strip():
            raise RuntimeError("Daemon returned empty response (may have crashed or timed out)")
        return json.loads(output.strip())

    # -- AsyncActionExecutor-compatible interface --

    @property
    def actions(self) -> _ToolDescriptionProxy:
        return _ToolDescriptionProxy(self._tool_descriptions)

    def __contains__(self, name: str) -> bool:
        return name in self._tool_descriptions

    def keys(self) -> List[str]:
        return list(self._tool_descriptions.keys())

    def description(self) -> List[Dict]:
        return list(self._tool_descriptions.values())

    async def forward(self, name: str, parameters: dict, **kwargs) -> ActionReturn:
        action_name = name.split(".")[0] if "." in name else name

        # Built-in actions stay local
        if action_name not in self:
            if name == self.no_action.name:
                return self.no_action(parameters)
            elif name == self.finish_action.name:
                return self.finish_action(parameters)
            else:
                return self.invalid_action(parameters)

        # Ensure daemon is running
        if not self._connected:
            await self.connect()

        # Route to daemon
        try:
            result = await self._daemon_call(
                {"name": name, "parameters": parameters}
            )
            if "error" in result:
                return ActionReturn(
                    args=parameters,
                    type=name,
                    errmsg=result["error"],
                    state=ActionStatusCode.API_ERROR,
                )
            action_return = _deserialize_action_return(json.dumps(result))
            action_return.valid = ActionValidCode.OPEN
            return action_return
        except Exception as exc:
            logger.warning("SandboxActionExecutor: %s failed: %s", name, exc)
            return ActionReturn(
                args=parameters,
                type=name,
                errmsg=str(exc),
                state=ActionStatusCode.API_ERROR,
            )

    async def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        for hook in self._hooks.values():
            if inspect.iscoroutinefunction(hook.before_action):
                result = await hook.before_action(self, message)
            else:
                result = hook.before_action(self, message)
            if result:
                message = result

        assert isinstance(message.content, FunctionCall) or (
            isinstance(message.content, dict)
            and "name" in message.content
            and "parameters" in message.content
        )
        if isinstance(message.content, dict):
            name = message.content.get("name")
            parameters = message.content.get("parameters")
        else:
            name = message.content.name
            parameters = message.content.parameters

        response_message = await self.forward(name=name, parameters=parameters, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.__class__.__name__,
                content=response_message,
            )

        for hook in self._hooks.values():
            if inspect.iscoroutinefunction(hook.after_action):
                result = await hook.after_action(self, response_message)
            else:
                result = hook.after_action(self, response_message)
            if result:
                response_message = result
        return response_message

    def register_hook(self, hook):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle
