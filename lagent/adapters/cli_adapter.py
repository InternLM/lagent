"""CLI Agent Adapters — wrap external agent CLIs as lagent Agents.

Three classes share one ``AsyncExternalAgent`` lineage:

* :class:`CLIAgentAdapter` — shared subprocess lifecycle (spawn, timeout,
  decode, error-handle). Usable directly with ``command_template=...`` for
  ad-hoc CLIs.
* :class:`ClaudeCodeCLIAdapter` — subclass for the ``claude`` CLI
  (``@anthropic-ai/claude-code``) with multi-turn via ``--resume``.
* :class:`CodexCLIAdapter` — subclass for the ``codex exec`` CLI
  (``@openai/codex``).

Subclasses customize behavior via two hooks:

* ``_build_argv(task)`` — return an argv list to spawn (no shell, no
  quoting hazards). Returning ``None`` falls back to the ``command_template``
  path.
* ``_default_parse(stdout, stderr)`` — extract the final result string
  (and stash any side state such as a session_id).

Trace capture works transparently through :class:`SessionClient`: the
proxy URL/key are injected into the subprocess env by
``AsyncExternalAgent._build_env()``, and the CLI subprocess (which reads
``ANTHROPIC_BASE_URL``/``OPENAI_BASE_URL``) routes its API calls through
the proxy.

Usage::

    from lagent.adapters.cli_adapter import ClaudeCodeCLIAdapter
    from lagent.adapters.proxy import SessionClient

    agent = ClaudeCodeCLIAdapter(
        model="claude-sonnet-4-5",
        permission_mode="bypassPermissions",
        proxy=SessionClient(real_base_url="...", real_api_key="...", port=8081),
    )
    r1 = await agent("Read README.md")
    r2 = await agent("Now summarize it")  # multi-turn via --resume
"""

import asyncio
import json
import shlex
import shutil
from typing import Any, Callable, Dict, List, Optional

from .base import AsyncExternalAgent


class CLIAgentAdapter(AsyncExternalAgent):
    """Shared subprocess machinery for CLI agent wrappers.

    Two ways to use:

    1. **Direct (template mode)** — ad-hoc one-shot CLIs::

           agent = CLIAgentAdapter(command_template="my-cli -p {task}")

       ``{task}`` is shell-escaped, so do not wrap it in extra quotes.

    2. **Subclass mode** — override :meth:`_build_argv` to return an
       argv list and (optionally) :meth:`_default_parse` to extract the
       result. See :class:`ClaudeCodeCLIAdapter` for an example.

    Args:
        command_template: Format string with ``{task}`` placeholder for
            template mode. Ignored if a subclass overrides ``_build_argv``.
        binary: Path to the CLI binary, used by ``setup()`` for the
            existence check. Inferred from ``command_template`` if omitted.
        parse_output: Optional ``(stdout, stderr) -> str`` extractor.
            Overrides the default ``_default_parse`` method.
        extra_args: Additional flags appended to the command.
        max_output_chars: Truncate the final result beyond this limit.
        **kwargs: Passed to AsyncExternalAgent.
    """

    def __init__(
        self,
        command_template: Optional[str] = None,
        binary: Optional[str] = None,
        parse_output: Optional[Callable[[str, str], str]] = None,
        extra_args: Optional[List[str]] = None,
        max_output_chars: int = 200000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.command_template = command_template
        self.binary = binary or (
            command_template.split()[0] if command_template else None
        )
        self.parse_output: Callable[[str, str], str] = (
            parse_output if parse_output is not None else self._default_parse
        )
        self.extra_args = extra_args or []
        self.max_output_chars = max_output_chars

    def setup(self) -> None:
        if self.binary and not shutil.which(self.binary):
            raise RuntimeError(
                f"CLI binary '{self.binary}' not found on PATH. "
                f"Ensure the external agent is installed."
            )

    def _build_argv(self, task: str) -> Optional[List[str]]:
        """Subclass hook: return argv list (no shell) to spawn.

        Default returns ``None``, which falls back to the
        ``command_template`` (shell) path.
        """
        return None

    def _default_parse(self, stdout: str, stderr: str) -> str:
        """Default extractor: stdout, with stderr appended if non-empty."""
        result = stdout.strip()
        if stderr.strip():
            result += f'\n[stderr]: {stderr.strip()}'
        return result

    async def run_external_async(self, task: str, **kwargs) -> str:
        argv = self._build_argv(task)
        env = self._build_env()
        cwd = self.working_dir

        if argv is not None:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        elif self.command_template:
            cmd = self.command_template.format(task=shlex.quote(task))
            if self.extra_args:
                cmd += ' ' + ' '.join(shlex.quote(a) for a in self.extra_args)
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        else:
            raise RuntimeError(
                "CLIAgentAdapter requires either `command_template` or "
                "an `_build_argv()` override."
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"`{self.binary or 'CLI agent'}` timed out after "
                f"{self.timeout}s"
            )

        stdout = stdout_b.decode('utf-8', errors='replace')
        stderr = stderr_b.decode('utf-8', errors='replace')

        if proc.returncode != 0:
            raise RuntimeError(
                f"`{self.binary or 'CLI agent'}` exited with code "
                f"{proc.returncode}.\nstderr: {stderr[:2000]}"
            )

        result = self.parse_output(stdout, stderr)
        if isinstance(result, str) and len(result) > self.max_output_chars:
            return result[: self.max_output_chars] + '\n...(truncated)'
        return result


class ClaudeCodeCLIAdapter(CLIAgentAdapter):
    """Wraps the ``claude`` CLI binary as an AsyncExternalAgent.

    Each call invokes::

        claude -p <task> --output-format json [...]

    Multi-turn: the first call captures ``session_id`` from the JSON
    output; subsequent calls add ``--resume <session_id>``. ``cwd`` must
    stay constant for resume to find the saved session file under
    ``~/.claude/projects/<cwd-hash>/<session-id>.jsonl``.

    Proxy: when a :class:`SessionClient` is attached, the base class
    injects ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` into the
    subprocess env. The CLI reads them and routes ``/v1/messages``
    through the proxy.

    Args:
        model: Model name or alias (e.g. ``"sonnet"``, ``"opus"``).
        system_prompt: Replace the default system prompt entirely.
        append_system_prompt: Append to the default system prompt.
        allowed_tools: Tool allowlist (``--allowed-tools``).
        disallowed_tools: Tool denylist (``--disallowed-tools``).
        permission_mode: One of ``"default"``, ``"acceptEdits"``,
            ``"auto"``, ``"bypassPermissions"``, ``"plan"``, ``"dontAsk"``.
        mcp_config: List of MCP config strings or paths
            (``--mcp-config``).
        effort: Reasoning effort (``"low"``…``"max"``).
        binary: Path to the ``claude`` binary. Default: resolved on PATH.
        **kwargs: Passed to :class:`CLIAgentAdapter`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        append_system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        permission_mode: str = "default",
        mcp_config: Optional[List[str]] = None,
        add_dirs: Optional[List[str]] = None,
        effort: Optional[str] = None,
        binary: str = "claude",
        **kwargs,
    ):
        kwargs.setdefault('name', 'claude-code-cli')
        kwargs.setdefault('description', 'Claude Code (CLI mode)')
        super().__init__(binary=binary, **kwargs)
        self.model = model
        self.system_prompt = system_prompt
        self.append_system_prompt = append_system_prompt
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.permission_mode = permission_mode
        self.mcp_config = mcp_config or []
        self.add_dirs = add_dirs or []
        self.effort = effort
        self._cli_session_id: Optional[str] = None

    def _build_argv(self, task: str) -> List[str]:
        argv = [self.binary, '-p', '--output-format', 'json']
        if self.model:
            argv += ['--model', self.model]
        if self.system_prompt:
            argv += ['--system-prompt', self.system_prompt]
        if self.append_system_prompt:
            argv += ['--append-system-prompt', self.append_system_prompt]
        if self.allowed_tools:
            argv += ['--allowed-tools', ' '.join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ['--disallowed-tools', ' '.join(self.disallowed_tools)]
        if self.permission_mode and self.permission_mode != 'default':
            argv += ['--permission-mode', self.permission_mode]
        if self.mcp_config:
            argv += ['--mcp-config', *self.mcp_config]
        for d in self.add_dirs:
            argv += ['--add-dir', d]
        if self.effort:
            argv += ['--effort', self.effort]
        if self._cli_session_id:
            argv += ['--resume', self._cli_session_id]
        argv += self.extra_args
        # `--` terminates option parsing so variadic flags
        # (e.g. --mcp-config, --allowed-tools) don't swallow the task.
        argv += ['--', task]
        return argv

    def reset_session(self) -> None:
        """Forget the captured CLI session_id; the next call starts fresh."""
        self._cli_session_id = None

    def _default_parse(self, stdout: str, stderr: str) -> str:
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return stdout
        if isinstance(payload, dict):
            sid = payload.get('session_id')
            if sid:
                self._cli_session_id = sid
            if payload.get('is_error'):
                raise RuntimeError(
                    f"claude CLI error: {str(payload.get('result', ''))[:2000]}"
                )
            return payload.get('result', '') or ''
        return str(payload)


class CodexCLIAdapter(CLIAgentAdapter):
    """Wraps the ``codex exec`` CLI as an AsyncExternalAgent.

    First call invokes::

        codex exec --json [...] -- <task>

    Subsequent calls use the ``resume`` subcommand for multi-turn::

        codex exec --json [...] resume <session_id> -- <task>

    The session_id is captured from the ``session_configured`` event
    in the NDJSON stream emitted on stdout. Codex (Rust impl) persists
    sessions to ``~/.codex/sessions/`` keyed by id.

    Proxy: when a :class:`SessionClient` is attached, the base class
    injects ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` into the subprocess
    env. For some configs you may also need ``-c model_provider=...``
    to point at the right backend.

    Args:
        model: Model name (e.g. ``"gpt-5.4-nano"``).
        config_overrides: ``{key: value}`` dict applied as
            ``-c key=value``. Values are parsed as TOML (fallback to
            string literal).
        profile: Codex profile name (``--profile``).
        sandbox_mode: One of ``"read-only"``, ``"workspace-write"``,
            ``"danger-full-access"``. Default: not set.
        bypass_approvals: If True, adds
            ``--dangerously-bypass-approvals-and-sandbox`` for fully
            unattended runs that may invoke shell commands.
        binary: Path to the ``codex`` binary.
        **kwargs: Passed to :class:`CLIAgentAdapter`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        profile: Optional[str] = None,
        sandbox_mode: Optional[str] = None,
        bypass_approvals: bool = False,
        skip_git_repo_check: bool = True,
        add_dirs: Optional[List[str]] = None,
        ephemeral: bool = False,
        ignore_user_config: bool = False,
        binary: str = "codex",
        **kwargs,
    ):
        kwargs.setdefault('name', 'codex-cli')
        kwargs.setdefault('description', 'OpenAI Codex (CLI mode)')
        super().__init__(binary=binary, **kwargs)
        self.model = model
        self.config_overrides = config_overrides or {}
        self.profile = profile
        self.sandbox_mode = sandbox_mode
        self.bypass_approvals = bypass_approvals
        self.skip_git_repo_check = skip_git_repo_check
        self.add_dirs = add_dirs or []
        self.ephemeral = ephemeral
        self.ignore_user_config = ignore_user_config
        self._cli_session_id: Optional[str] = None

    @staticmethod
    def _toml_render(value: Any) -> str:
        """Render a Python value as a TOML scalar for ``-c key=value``.

        codex parses the value as TOML; bools are case-sensitive
        (``true``/``false``). Other types fall back to ``str()`` —
        codex uses the raw string literal if TOML parsing fails.
        """
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    def _build_argv(self, task: str) -> List[str]:
        argv = [self.binary, 'exec', '--json']
        if self.model:
            argv += ['--model', self.model]
        if self.profile:
            argv += ['--profile', self.profile]
        if self.sandbox_mode:
            argv += ['--sandbox', self.sandbox_mode]
        if self.bypass_approvals:
            argv += ['--dangerously-bypass-approvals-and-sandbox']
        if self.skip_git_repo_check:
            argv += ['--skip-git-repo-check']
        for d in self.add_dirs:
            argv += ['--add-dir', d]
        if self.ephemeral:
            argv += ['--ephemeral']
        if self.ignore_user_config:
            argv += ['--ignore-user-config']
        for k, v in self.config_overrides.items():
            argv += ['-c', f'{k}={self._toml_render(v)}']
        if self._cli_session_id:
            argv += ['resume', self._cli_session_id]
        argv += self.extra_args
        argv += ['--', task]
        return argv

    def _default_parse(self, stdout: str, stderr: str) -> str:
        last_text = ''
        error_payload: Optional[dict] = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            # Some codex versions wrap payload as {"id": ..., "msg": {...}}
            payload = event.get('msg') if isinstance(event.get('msg'), dict) else event
            etype = payload.get('type', '')

            # First event in a fresh session carries the session_id
            if etype == 'session_configured':
                sid = (
                    payload.get('session_id')
                    or payload.get('id')
                    or (payload.get('session') or {}).get('id')
                )
                if sid:
                    self._cli_session_id = sid

            # Surface error events so a failed task doesn't silently
            # return raw NDJSON.
            if etype in ('error', 'session_error', 'task_failed', 'stream_error'):
                error_payload = payload

            if etype in ('agent_message', 'message') and isinstance(
                payload.get('message'), str
            ):
                last_text = payload['message']
            elif etype == 'task_complete' and isinstance(
                payload.get('last_agent_message'), str
            ):
                last_text = payload['last_agent_message']
            elif etype == 'item.completed':
                item = payload.get('item') or {}
                if isinstance(item, dict) and isinstance(item.get('text'), str):
                    last_text = item['text']

        if error_payload and not last_text:
            msg = error_payload.get('message') or json.dumps(error_payload)
            raise RuntimeError(f"codex error: {str(msg)[:2000]}")

        return last_text or stdout

    def reset_session(self) -> None:
        """Forget the captured CLI session_id; the next call starts fresh."""
        self._cli_session_id = None
