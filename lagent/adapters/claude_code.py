"""Claude Code CLI adapter — wraps the claude CLI as a lagent Agent.

Supports real multi-turn via Claude Code's ``--continue`` flag:
each ``forward()`` call after the first automatically resumes
the previous session.

Usage::

    from lagent.adapters.claude_code import ClaudeCodeAdapter

    agent = ClaudeCodeAdapter(timeout=120)
    r1 = await agent("Read main.py and explain what it does")
    r2 = await agent("Now refactor the error handling")  # continues same session
    print(r2.content)

    # With proxy for trajectory capture:
    from lagent.adapters.proxy import LLMProxyRecorder
    proxy = LLMProxyRecorder(real_api_key="...", real_base_url="...")
    agent = ClaudeCodeAdapter(proxy=proxy, timeout=120)
    result = await agent("Fix the bug")
    trace = agent.state_dict()['llm_trace']
"""

import asyncio
import json
import os
import shlex
import shutil
from typing import Callable, Dict, List, Optional

from .base import AsyncExternalAgent


def _parse_claude_stream_json(stdout: str, stderr: str) -> str:
    """Parse claude --output-format=stream-json output."""
    result_parts = []
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg_type = data.get('type', '')
        if msg_type == 'result':
            text = data.get('result', '')
            if text:
                result_parts.append(text)
        elif msg_type == 'assistant':
            msg = data.get('message', {})
            content = msg.get('content', [])
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get('type') == 'text':
                    result_parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    result_parts.append(block)
    if result_parts:
        return '\n'.join(result_parts)
    return stdout.strip()


def _parse_claude_text(stdout: str, stderr: str) -> str:
    """Parse claude --output-format=text output."""
    return stdout.strip()


class ClaudeCodeAdapter(AsyncExternalAgent):
    """Wraps Claude Code CLI as a lagent Agent with real multi-turn support.

    Uses Claude Code's ``--continue`` flag to resume sessions across
    ``forward()`` calls. Each call spawns a new subprocess but the
    conversation history is preserved by Claude Code internally.

    Args:
        output_format: "text" or "stream-json". Default: "text".
        max_turns: Maximum number of agent turns per call. Default: None.
        permission_mode: Permission mode. Default: "default".
        extra_flags: Additional CLI flags as a list of strings.
        model: Model name override.
        parse_output: Custom output parser ``(stdout, stderr) -> str``.
        max_output_chars: Truncate output beyond this limit.
        **kwargs: Passed to AsyncExternalAgent (name, working_dir,
            env_vars, timeout, proxy, hooks).
    """

    def __init__(
        self,
        output_format: str = 'text',
        max_turns: Optional[int] = None,
        permission_mode: str = 'default',
        extra_flags: Optional[List[str]] = None,
        model: Optional[str] = None,
        parse_output: Optional[Callable[[str, str], str]] = None,
        max_output_chars: int = 50000,
        **kwargs,
    ):
        # Build env vars
        env_vars = kwargs.pop('env_vars', {}) or {}
        if model:
            env_vars['ANTHROPIC_MODEL'] = model
        for key in ('ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY',
                    'ANTHROPIC_BASE_URL'):
            if key not in env_vars and os.environ.get(key):
                env_vars[key] = os.environ[key]

        kwargs.setdefault('name', 'claude-code')
        kwargs.setdefault('description', 'Claude Code CLI agent')

        super().__init__(env_vars=env_vars, **kwargs)

        self.output_format = output_format
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.extra_flags = extra_flags or []
        self.max_output_chars = max_output_chars
        self._call_count = 0

        if parse_output:
            self.parse_output = parse_output
        elif output_format == 'stream-json':
            self.parse_output = _parse_claude_stream_json
        else:
            self.parse_output = _parse_claude_text

    def setup(self) -> None:
        if not shutil.which('claude'):
            raise RuntimeError(
                "Claude Code CLI ('claude') not found on PATH. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )

    def _build_command(self, task: str) -> str:
        """Build the CLI command, adding --continue for subsequent calls."""
        cmd_parts = ['claude', '--print', '-p', shlex.quote(task)]
        cmd_parts.append(f'--output-format={self.output_format}')
        if self.permission_mode != 'default':
            cmd_parts.append(f'--permission-mode={self.permission_mode}')
        if self.max_turns is not None:
            cmd_parts.append(f'--max-turns={self.max_turns}')

        # Real multi-turn: --continue on subsequent calls
        if self._call_count > 0:
            cmd_parts.append('--continue')

        if self.extra_flags:
            cmd_parts.extend(self.extra_flags)

        return ' '.join(cmd_parts)

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
                f"Claude Code timed out after {self.timeout}s"
            )

        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')

        if process.returncode != 0:
            raise RuntimeError(
                f"Claude Code exited with code {process.returncode}.\n"
                f"stderr: {stderr[:2000]}"
            )

        if len(stdout) > self.max_output_chars:
            stdout = stdout[:self.max_output_chars] + '\n...(truncated)'

        self._call_count += 1
        return self.parse_output(stdout, stderr)
