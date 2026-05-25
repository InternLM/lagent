"""SandboxAgent — drop-in replacement for ``AsyncAgent`` that proxies
to an :class:`~lagent.serving.sandbox.daemon.AgentDaemon`
running inside a sandbox.

The daemon holds a full agent (LLM + Actions + Skills).  SandboxAgent
sends ``chat`` / ``state_dict`` / ``reset`` commands via the sandbox's
bash channel and returns the results as regular ``AgentMessage`` objects.

Usage::

    from lagent.serving.sandbox.agent import SandboxAgent

    agent = SandboxAgent(
        sandbox_client=my_sandbox_client,
        agent_config={
            "type": "lagent.agents.internclaw_agent.InternClawAgent",
            "policy_agent": {...},
            "env_agent": {...},
        },
    )
    await agent.connect()

    response = await agent("Fix the bug in main.py")
    print(response.content)

    state = await agent.get_state_dict()
    await agent.reset()
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Union

from lagent.schema import AgentMessage

logger = logging.getLogger(__name__)


class SandboxAgent:
    """Drop-in for ``AsyncAgent`` that proxies to a Level 2 daemon.

    Parameters
    ----------
    sandbox_client
        Object with ``execute(command, ...)`` that runs bash in the sandbox.
        May be sync or async.
    agent_config : dict
        Agent configuration (passed to daemon, which calls ``create_object``).
    sock_path : str
        Unix socket path inside the sandbox.
    cwd : str
        Working directory for bash commands.
    daemon_module : str
        Python module path for the daemon.
    """

    def __init__(
        self,
        sandbox_client,
        agent_config: Dict,
        sock_path: str = "/tmp/lagent_action.sock",
        cwd: str = "/root",
        daemon_module: str = "lagent.serving.sandbox.daemon",
        python_bin: str = "/mnt/llm-ai-infra/miniconda3/envs/train/bin/python",
    ):
        self.sandbox_client = sandbox_client
        self.agent_config = agent_config
        self.sock_path = sock_path
        self.cwd = cwd
        self.daemon_module = daemon_module
        self.python_bin = python_bin
        self._connected = False
        self._lock = asyncio.Lock()

    # -- sandbox communication --

    async def _exec(self, command: str, **kwargs) -> str:
        kwargs.setdefault("cwd", self.cwd)
        execute_fn = self.sandbox_client.execute
        if inspect.iscoroutinefunction(execute_fn):
            result = await execute_fn(command, **kwargs)
        else:
            result = await asyncio.to_thread(execute_fn, command, **kwargs)
        if isinstance(result, dict):
            stdout = result.get("stdout", "")
            # If stdout is empty but stderr has content, raise so caller can debug
            if not stdout.strip() and result.get("stderr", "").strip():
                raise RuntimeError(f"Command stderr: {result['stderr'][:500]}")
            return stdout
        return result

    async def _daemon_call(self, request: dict, timeout_sec: int = 600) -> dict:
        """Send a request to daemon. Uses longer timeout for chat commands."""
        request_json = json.dumps(request, ensure_ascii=False)
        escaped = request_json.replace("'", "'\\''")
        output = await self._exec(
            f"{self.python_bin} -m {self.daemon_module} call "
            f"--sock {self.sock_path} "
            f"'{escaped}'",
            timeout_sec=timeout_sec,
        )
        if not output.strip():
            raise RuntimeError("Daemon returned empty response (may have crashed or timed out)")
        return json.loads(output.strip())

    # -- lifecycle --

    async def connect(self) -> None:
        """Start the Level 2 daemon inside the sandbox (idempotent)."""
        async with self._lock:
            if self._connected:
                return

            # Write agent config
            config_json = json.dumps(self.agent_config, ensure_ascii=False)
            escaped = config_json.replace("'", "'\\''")
            await self._exec(
                f"echo '{escaped}' > /tmp/lagent_agent_config.json"
            )

            # Check if daemon already running, start if not
            check = await self._exec(
                f"pgrep -f '[l]agent.serving.sandbox.daemon.*--sock {self.sock_path}'"
                f" > /dev/null 2>&1 && echo 'running' || echo 'stopped'"
            )
            if "stopped" in check:
                await self._exec(
                    f"nohup {self.python_bin} -m {self.daemon_module} start "
                    f"--sock {self.sock_path} "
                    f"--agent-config /tmp/lagent_agent_config.json "
                    f"> /tmp/lagent_daemon.log 2>&1 &"
                )

            # Wait for socket
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
                    f"Daemon did not start within 15s. "
                    f"Check /tmp/lagent_daemon.log inside sandbox."
                )

            # Verify it's an AgentDaemon
            result = await self._daemon_call({"cmd": "ping"})
            assert result.get("type") == "agent", (
                f"Expected AgentDaemon, got: {result}"
            )
            self._connected = True
            logger.info("SandboxAgent connected (Level 2 daemon)")

    async def close(self) -> None:
        if not self._connected:
            return
        try:
            await self._daemon_call({"cmd": "shutdown"})
        except Exception:
            pass
        self._connected = False

    # -- Agent-compatible interface --

    async def __call__(self, *messages, **kwargs) -> AgentMessage:
        """Send messages to the agent and return its response."""
        if not self._connected:
            await self.connect()

        str_messages = []
        for m in messages:
            if isinstance(m, AgentMessage):
                str_messages.append(m.content if isinstance(m.content, str) else m.model_dump())
            else:
                str_messages.append(str(m))

        result = await self._daemon_call({
            "cmd": "chat",
            "messages": str_messages,
        })

        if "error" in result:
            return AgentMessage(
                sender="SandboxAgent",
                content=f"Agent error: {result['error']}",
            )

        return AgentMessage(**{
            k: v for k, v in result.items()
            if k in AgentMessage.model_fields
        })

    async def get_state_dict(self) -> Dict:
        """Get the agent's full state (memory + traces)."""
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({"cmd": "state_dict"})
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("state_dict", {})

    async def load_state_dict(self, state_dict: Dict) -> None:
        result = await self._daemon_call({
            "cmd": "load_state_dict",
            "state_dict": state_dict,
        })
        if "error" in result:
            raise RuntimeError(result["error"])

    async def get_messages(self) -> List[Dict[str, list]]:
        result = await self._daemon_call({"cmd": "get_messages"})
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("messages", [])

    async def reset(self, recursive: bool = True) -> None:
        result = await self._daemon_call({
            "cmd": "reset",
            "recursive": recursive,
        })
        if "error" in result:
            raise RuntimeError(result["error"])
