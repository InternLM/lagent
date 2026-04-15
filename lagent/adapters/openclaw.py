"""OpenClaw CLI adapter — wraps the openclaw CLI as a lagent Agent.

OpenClaw is a personal AI assistant with multi-channel support.
This adapter uses its ``agent --local --message`` CLI interface,
similar to ClaudeCodeAdapter's approach.

Requires: Node 22+, ``npm install -g openclaw``

Usage::

    from lagent.adapters.openclaw import OpenClawAdapter

    agent = OpenClawAdapter(
        thinking='medium',
        timeout=120,
    )
    r1 = await agent("What is 2+2?")
    r2 = await agent("Now multiply by 3")  # multi-turn via --session-id
"""

import asyncio
import json
import os
import shlex
import shutil
from typing import Any, Dict, List, Optional

from .base import AsyncExternalAgent


class OpenClawAdapter(AsyncExternalAgent):
    """Wraps OpenClaw CLI as a lagent Agent.

    Uses ``openclaw agent --local --message`` for execution.
    Real multi-turn via ``--session-id`` (same session across calls).

    Args:
        thinking: Thinking level (off/minimal/low/medium/high/xhigh).
        agent_id: OpenClaw agent id. Default: None (use default agent).
        json_output: Return JSON output. Default: True.
        node_version: Node version to use via nvm. Default: "22".
        nvm_dir: NVM directory. Default: ~/.nvm.
        **kwargs: Passed to AsyncExternalAgent.
    """

    def __init__(
        self,
        thinking: str = 'medium',
        agent_id: Optional[str] = 'main',
        json_output: bool = True,
        node_version: str = '22',
        nvm_dir: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'openclaw')
        kwargs.setdefault('description', 'OpenClaw personal AI assistant')
        super().__init__(**kwargs)

        self.thinking = thinking
        self.agent_id = agent_id
        self.json_output = json_output
        self.node_version = node_version
        self.nvm_dir = nvm_dir or os.path.expanduser('~/.nvm')
        self._openclaw_session_id: Optional[str] = None
        self._call_count = 0

    def setup(self) -> None:
        """Verify openclaw is accessible (via nvm)."""
        # We'll use nvm to access the right node version at runtime
        # Just check nvm exists
        nvm_sh = os.path.join(self.nvm_dir, 'nvm.sh')
        if not os.path.exists(nvm_sh):
            raise RuntimeError(
                f"nvm not found at {nvm_sh}. "
                f"Install nvm and then: nvm install {self.node_version} && "
                f"npm install -g openclaw"
            )

    def _build_command(self, task: str) -> str:
        """Build the openclaw CLI command."""
        # Source nvm to get the right node version
        nvm_prefix = (
            f'export NVM_DIR="{self.nvm_dir}" && '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
            f'nvm use {self.node_version} > /dev/null 2>&1 && '
        )

        cmd_parts = ['openclaw', 'agent', '--local']
        cmd_parts.extend(['--message', shlex.quote(task)])
        cmd_parts.extend(['--thinking', self.thinking])

        if self.json_output:
            cmd_parts.append('--json')
        if self.agent_id:
            cmd_parts.extend(['--agent', self.agent_id])

        # Multi-turn: use same session-id across calls
        if self._openclaw_session_id:
            cmd_parts.extend(['--session-id', self._openclaw_session_id])

        return nvm_prefix + ' '.join(cmd_parts)

    async def run_external_async(self, task: str, **kwargs) -> str:
        cmd = self._build_command(task)
        env = self._build_env()

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"OpenClaw timed out after {self.timeout}s"
            )

        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')

        if process.returncode != 0:
            raise RuntimeError(
                f"OpenClaw exited with code {process.returncode}.\n"
                f"stderr: {stderr[:2000]}"
            )

        self._call_count += 1

        # Parse JSON output to extract session-id and content
        if self.json_output:
            return self._parse_json_output(stdout)
        return stdout.strip()

    def _parse_json_output(self, stdout: str) -> str:
        """Parse openclaw --json output."""
        try:
            data = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return stdout.strip()

        # Capture session-id for multi-turn
        if isinstance(data, dict):
            sid = data.get('sessionId') or data.get('session_id')
            if sid:
                self._openclaw_session_id = sid
            # Extract the reply text
            reply = data.get('reply') or data.get('message') or data.get('content', '')
            if isinstance(reply, str):
                return reply
            return json.dumps(data)

        return stdout.strip()
