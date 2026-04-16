"""SandboxSkillsLoader — drop-in replacement for ``SkillsLoader`` that
routes all skill operations to a :class:`SkillsDaemon` running inside
a sandbox.

Usage::

    from lagent.skills.sandbox_skills import SandboxSkillsLoader

    skills = SandboxSkillsLoader(
        sandbox_client=client,
        workspace="/root/workspace",
    )
    await skills.connect()

    summary = await skills.build_skills_summary()
    always = await skills.get_always_skills()
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SandboxSkillsLoader:
    """Drop-in for ``SkillsLoader`` that proxies to a ``SkillsDaemon``.

    Implements the same async interface as ``SkillsLoader``.
    Manages its own daemon lifecycle (separate from ActionDaemon).

    Parameters
    ----------
    sandbox_client
        Object with ``execute(command, ...)`` method.
    workspace : str
        Workspace path inside the sandbox.
    sock_path : str
        Unix socket path for the SkillsDaemon inside sandbox.
    daemon_module : str
        Python module path for the daemon.
    """

    def __init__(
        self,
        sandbox_client,
        workspace: str = "/root/workspace",
        sock_path: str = "/tmp/lagent_skills.sock",
        daemon_module: str = "lagent.serving.sandbox.daemon",
    ):
        self.sandbox_client = sandbox_client
        self.workspace = workspace
        self.sock_path = sock_path
        self.daemon_module = daemon_module
        self._connected = False
        self._lock = asyncio.Lock()

    async def _exec(self, command: str, **kwargs) -> str:
        kwargs.setdefault("cwd", "/tmp")
        execute_fn = self.sandbox_client.execute
        if inspect.iscoroutinefunction(execute_fn):
            result = await execute_fn(command, **kwargs)
        else:
            result = await asyncio.to_thread(execute_fn, command, **kwargs)
        if isinstance(result, dict):
            return result.get("stdout", "")
        return result

    async def _daemon_call(self, request: dict) -> dict:
        request_json = json.dumps(request, ensure_ascii=False)
        escaped = request_json.replace("'", "'\\''")
        output = await self._exec(
            f"python -m {self.daemon_module} call "
            f"--sock {self.sock_path} "
            f"'{escaped}'"
        )
        return json.loads(output.strip())

    async def connect(self) -> None:
        """Start the SkillsDaemon inside sandbox (idempotent)."""
        async with self._lock:
            if self._connected:
                return

            # Write a small Python script that starts SkillsDaemon
            # with a SkillsLoader for the workspace
            start_script = (
                f"from lagent.skills.skills import SkillsLoader; "
                f"from lagent.serving.sandbox.daemon import SkillsDaemon; "
                f"import asyncio; "
                f"loader = SkillsLoader('{self.workspace}'); "
                f"daemon = SkillsDaemon(skills_loader=loader, sock_path='{self.sock_path}'); "
                f"asyncio.run(daemon.start())"
            )
            escaped_script = start_script.replace("'", "'\\''")
            await self._exec(
                f"echo '{escaped_script}' > /tmp/lagent_start_skills_daemon.py"
            )

            # Check if already running
            check = await self._exec(
                f"pgrep -f '[l]agent_start_skills_daemon' "
                f"> /dev/null 2>&1 && echo 'running' || echo 'stopped'"
            )
            if "stopped" in check:
                await self._exec(
                    f"nohup python /tmp/lagent_start_skills_daemon.py "
                    f"> /tmp/lagent_skills_daemon.log 2>&1 &"
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
                    f"SkillsDaemon did not start within 15s. "
                    f"Check /tmp/lagent_skills_daemon.log inside sandbox."
                )

            self._connected = True
            logger.info("SandboxSkillsLoader connected")

    async def close(self) -> None:
        if not self._connected:
            return
        try:
            await self._daemon_call({"cmd": "shutdown"})
        except Exception:
            pass
        self._connected = False

    # -- SkillsLoader-compatible interface --

    async def list_skills(self, filter_unavailable: bool = True) -> list:
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({
            "cmd": "list_skills",
            "filter_unavailable": filter_unavailable,
        })
        return result.get("skills", [])

    async def build_skills_summary(self) -> str:
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({"cmd": "skills_summary"})
        return result.get("summary", "")

    async def load_skill(self, name: str) -> Optional[str]:
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({"cmd": "load_skill", "name": name})
        return result.get("content")

    async def load_skills_for_context(self, skill_names: List[str]) -> str:
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({
            "cmd": "load_skills_for_context",
            "names": skill_names,
        })
        return result.get("content", "")

    async def get_always_skills(self) -> List[str]:
        if not self._connected:
            await self.connect()
        result = await self._daemon_call({"cmd": "get_always_skills"})
        return result.get("skills", [])
