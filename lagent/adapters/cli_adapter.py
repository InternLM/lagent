"""CLI Agent Adapter — wraps external CLI agents as lagent Agents.

Spawns external agent CLIs (claude, aider, codex, etc.) as subprocesses,
captures stdout/stderr, and converts the output into AgentMessages.

The ``command_template`` must contain a ``{task}`` placeholder::

    adapter = CLIAgentAdapter(
        name="claude-code",
        command_template="claude -p '{task}' --output-format text",
        timeout=300,
    )
    result = await adapter("Fix the bug in main.py")

A custom ``parse_output`` function can be injected to extract structured
results from the raw CLI output.
"""

import asyncio
import os
import shlex
import shutil
from typing import Callable, Dict, List, Optional

from .base import AsyncExternalAgent


class CLIAgentAdapter(AsyncExternalAgent):
    """Wraps an external agent accessible via CLI (subprocess).

    Args:
        command_template: Format string for the CLI command.
            Must contain ``{task}`` placeholder.
            Example: ``"claude -p '{task}' --output-format text"``
        parse_output: Callable ``(stdout, stderr) -> str`` that extracts
            the desired result from raw output. Default: return stdout.
        shell: Whether to use shell execution. Default: True.
        extra_args: Additional CLI arguments appended to the command.
        max_output_chars: Truncate output beyond this limit.
        **kwargs: Passed to AsyncExternalAgent (name, working_dir,
            env_vars, timeout, proxy, hooks).
    """

    def __init__(
        self,
        command_template: str,
        parse_output: Optional[Callable[[str, str], str]] = None,
        shell: bool = True,
        extra_args: Optional[List[str]] = None,
        max_output_chars: int = 50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.command_template = command_template
        self.parse_output = parse_output or self._default_parse
        self.shell = shell
        self.extra_args = extra_args or []
        self.max_output_chars = max_output_chars

    def setup(self) -> None:
        """Verify the CLI binary exists on PATH."""
        binary = self.command_template.split()[0]
        if not shutil.which(binary):
            raise RuntimeError(
                f"CLI binary '{binary}' not found on PATH. "
                f"Ensure the external agent is installed."
            )

    async def run_external_async(self, task: str, **kwargs) -> str:
        """Spawn subprocess, capture output, parse, return."""
        escaped_task = shlex.quote(task)
        cmd = self.command_template.format(task=task)
        if self.extra_args:
            cmd += ' ' + ' '.join(self.extra_args)

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
                f"CLI agent timed out after {self.timeout}s"
            )

        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')

        if process.returncode != 0:
            raise RuntimeError(
                f"CLI agent exited with code {process.returncode}.\n"
                f"stderr: {stderr[:2000]}"
            )

        if len(stdout) > self.max_output_chars:
            stdout = stdout[:self.max_output_chars] + '\n...(truncated)'

        return self.parse_output(stdout, stderr)

    @staticmethod
    def _default_parse(stdout: str, stderr: str) -> str:
        """Default parser: return stdout, append stderr if non-empty."""
        result = stdout.strip()
        if stderr.strip():
            result += f'\n[stderr]: {stderr.strip()}'
        return result
