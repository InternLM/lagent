"""OpenClaw CLI adapter — wraps the ``openclaw`` CLI as a lagent Agent.

Each call invokes::

    openclaw agent --local --message <task> [--thinking ...] [--json] \\
        [--agent <id>] [--session-id <sid>]

Multi-turn: the first call captures ``sessionId`` from the JSON output;
subsequent calls add ``--session-id <sid>``. Set ``json_output=False`` to
get raw text (no session capture, single-turn only).

The ``openclaw`` binary is published via npm and typically installed
through nvm. Two execution modes:

* **argv mode (default)** — requires ``openclaw`` on PATH. Spawned
  directly via ``create_subprocess_exec`` (no shell escaping hazards).
* **nvm shell-wrap mode** — pass ``nvm_dir`` (and optionally
  ``node_version``); the adapter wraps the call in
  ``bash -lc 'source nvm.sh && nvm use ... && exec openclaw ...'`` so
  the right Node toolchain is loaded at runtime.

Proxy: when a :class:`SessionClient` is attached, the base class injects
``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` (and matching API keys)
into the subprocess env.

Usage::

    # Set OPENAI_BASE_URL / OPENAI_API_KEY, or pass ``proxy=...`` and let
    # the adapter inject them into the OpenClaw subprocess.
    from lagent.adapters.openclaw import OpenClawAdapter

    agent = OpenClawAdapter(model='gpt-4o-mini', thinking='medium', timeout=120)
    r1 = await agent("What is 2+2?")
    r2 = await agent("Now multiply by 3")  # multi-turn via --session-id
"""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .cli_adapter import CLIAgentAdapter


class OpenClawAdapter(CLIAgentAdapter):
    """Wraps the ``openclaw`` CLI as an :class:`AsyncExternalAgent`.

    Args:
        model: Model id exposed by the configured OpenAI-compatible backend.
        thinking: Thinking level (``off`` / ``minimal`` / ``low`` /
            ``medium`` / ``high`` / ``xhigh``).
        agent_id: OpenClaw agent id (``--agent``). Default: ``"main"``.
        json_output: Pass ``--json`` and parse the JSON envelope to
            extract ``sessionId`` (multi-turn) and the reply text.
            Default: True.
        skip_bootstrap: Skip OpenClaw's first-run identity ritual by
            removing ``BOOTSTRAP.md`` and pre-seeding ``IDENTITY.md`` /
            ``USER.md``. Default: True (recommended for headless /
            programmatic use).
        openclaw_home: OpenClaw state/config directory. Default:
            ``OPENCLAW_HOME`` / ``OPENCLAW_STATE_DIR`` / ``~/.openclaw``.
        nvm_dir: If set, wrap the spawn in ``bash -lc`` and source
            ``$nvm_dir/nvm.sh`` before invoking ``openclaw``. Use this
            when the binary is provided by nvm and not on PATH.
        node_version: Node version to ``nvm use`` (only when
            ``nvm_dir`` is set). Default: ``"22"``.
        binary: Path / name of the ``openclaw`` binary. Default:
            ``"openclaw"``.
        **kwargs: Passed to :class:`CLIAgentAdapter`.
    """

    def __init__(
        self,
        model: str,
        thinking: str = 'medium',
        agent_id: Optional[str] = 'main',
        json_output: bool = True,
        skip_bootstrap: bool = True,
        openclaw_home: Optional[str] = None,
        nvm_dir: Optional[str] = None,
        node_version: str = '22',
        binary: str = 'openclaw',
        **kwargs,
    ):
        kwargs.setdefault('name', 'openclaw')
        kwargs.setdefault('description', 'OpenClaw personal AI assistant')
        super().__init__(binary=binary, **kwargs)
        if not model:
            raise ValueError('OpenClawAdapter requires `model`.')
        self.model = model
        self.thinking = thinking
        self.agent_id = agent_id
        self.json_output = json_output
        self.skip_bootstrap = skip_bootstrap
        self.nvm_dir = nvm_dir
        self.node_version = node_version
        self.provider = 'custom-openai'
        self.openclaw_home = Path(
            openclaw_home
            or self.env_vars.get('OPENCLAW_HOME')
            or self.env_vars.get('OPENCLAW_STATE_DIR')
            or os.environ.get('OPENCLAW_HOME')
            or os.environ.get('OPENCLAW_STATE_DIR')
            or Path.home() / '.openclaw'
        ).expanduser()
        self.openclaw_config_path = Path(
            self.env_vars.get('OPENCLAW_CONFIG_PATH')
            or os.environ.get('OPENCLAW_CONFIG_PATH')
            or self.openclaw_home / 'openclaw.json'
        ).expanduser()
        self.env_vars.setdefault('OPENCLAW_HOME', str(self.openclaw_home))
        self.env_vars.setdefault('OPENCLAW_STATE_DIR', str(self.openclaw_home))
        self.env_vars.setdefault('OPENCLAW_CONFIG_PATH', str(self.openclaw_config_path))
        self.env_vars.setdefault('NO_COLOR', '1')
        self._cli_session_id: Optional[str] = None
        self._runtime_config_written = False
        self._openclaw_version: Optional[Tuple[int, int, int]] = None

    @staticmethod
    def _parse_openclaw_version(text: str) -> Optional[Tuple[int, int, int]]:
        match = re.search(r'OpenClaw\s+(\d+)\.(\d+)\.(\d+)', text)
        if not match:
            return None
        return tuple(int(part) for part in match.groups())

    def _detect_openclaw_version(self) -> Optional[Tuple[int, int, int]]:
        if self._openclaw_version is not None:
            return self._openclaw_version
        try:
            proc = subprocess.run(
                [self.binary, '--version'],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            text = (proc.stdout or proc.stderr or '').strip()
            print(f"openclaw version: {text}")
            self._openclaw_version = self._parse_openclaw_version(text)
        except (OSError, subprocess.SubprocessError):
            self._openclaw_version = None
        return self._openclaw_version

    def _supports_provider_timeout_seconds(self) -> bool:
        """``models.providers.*.timeoutSeconds`` landed in 2026.4.26."""
        version = self._detect_openclaw_version()
        return version is not None and version >= (2026, 4, 26)

    def setup(self) -> None:
        if self.nvm_dir:
            nvm_sh = os.path.join(self.nvm_dir, 'nvm.sh')
            if not os.path.exists(nvm_sh):
                raise RuntimeError(
                    f"nvm.sh not found at {nvm_sh}. Install nvm and run: "
                    f"nvm install {self.node_version} && "
                    f"npm install -g openclaw"
                )
            return
        super().setup()

    async def run_external_async(self, task: str, **kwargs) -> str:
        # 关键点：SessionClient 的端口在 forward() 里启动后才确定，因此配置必须运行前写。
        self._write_openclaw_config()
        return await super().run_external_async(task, **kwargs)

    def _build_argv(self, task: str) -> List[str]:
        cli_args = ['agent', '--local', '--message', task, '--thinking', self.thinking]
        if self.json_output:
            cli_args.append('--json')
        if self.agent_id:
            cli_args.extend(['--agent', self.agent_id])
        if self._cli_session_id:
            cli_args.extend(['--session-id', self._cli_session_id])
        if self.timeout:
            cli_args.extend(['--timeout', str(self.timeout)])
        cli_args.extend(self.extra_args)

        if self.nvm_dir:
            # Source nvm in a login shell, then exec openclaw with the
            # built argv. shlex.quote on every token keeps the task safe.
            quoted = ' '.join(shlex.quote(a) for a in [self.binary, *cli_args])
            wrap = (
                f'export NVM_DIR={shlex.quote(self.nvm_dir)} && '
                f'. "$NVM_DIR/nvm.sh" && '
                f'nvm use {shlex.quote(self.node_version)} > /dev/null && '
                f'exec {quoted}'
            )
            return ['bash', '-lc', wrap]

        return [self.binary, *cli_args]

    def reset_session(self) -> None:
        """Forget the captured session id; the next call starts fresh."""
        self._cli_session_id = None

    @staticmethod
    def _load_openclaw_json(text: str) -> Optional[dict]:
        """Load an OpenClaw JSON envelope from stdout or mixed stderr logs."""
        stripped = text.strip()
        if not stripped:
            return None
        try:
            data = json.loads(stripped)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        fallback = None
        for match in re.finditer(r'\{', stripped):
            try:
                data, _ = decoder.raw_decode(stripped[match.start():])
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if 'payloads' in data or 'meta' in data:
                return data
            fallback = data
        return fallback

    _IDENTITY_TEMPLATE_MARKER = 'Fill this in during your first conversation'
    _USER_TEMPLATE_MARKER = '_Learn about the person you'
    _MINIMAL_IDENTITY = """\
# IDENTITY.md - Who Am I?

- **Name:** OpenClaw
- **Creature:** AI assistant
- **Vibe:** Helpful and direct
- **Emoji:** 🤖
"""
    _MINIMAL_USER = """\
# USER.md - About Your Human

- **Name:** User
- **What to call them:** User
"""

    def _prepare_workspace(self, workspace: str) -> None:
        """Remove bootstrap ritual files so headless calls skip onboarding."""
        if not self.skip_bootstrap:
            return

        root = Path(workspace).expanduser()
        bootstrap = root / 'BOOTSTRAP.md'
        if bootstrap.is_file():
            bootstrap.unlink()

        identity = root / 'IDENTITY.md'
        if identity.is_file():
            text = identity.read_text(encoding='utf-8')
            if self._IDENTITY_TEMPLATE_MARKER in text:
                identity.write_text(self._MINIMAL_IDENTITY, encoding='utf-8')
        else:
            identity.write_text(self._MINIMAL_IDENTITY, encoding='utf-8')

        user = root / 'USER.md'
        if user.is_file():
            text = user.read_text(encoding='utf-8')
            if self._USER_TEMPLATE_MARKER in text:
                user.write_text(self._MINIMAL_USER, encoding='utf-8')
        else:
            user.write_text(self._MINIMAL_USER, encoding='utf-8')

    def _write_openclaw_config(self) -> None:
        env = self._build_env()
        base_url = (env.get('OPENAI_BASE_URL') or '').rstrip('/')
        api_key = env.get('OPENAI_API_KEY')
        if not base_url:
            raise RuntimeError(
                'OpenClawAdapter needs OPENAI_BASE_URL in subprocess env.'
            )
        if not api_key:
            raise RuntimeError(
                'OpenClawAdapter needs OPENAI_API_KEY in subprocess env.'
            )

        agent_id = self.agent_id or 'main'
        model_ref = f'{self.provider}/{self.model}'
        workspace = self.working_dir or os.environ.get('TASK_WORKSPACE') or os.getcwd()
        self._prepare_workspace(workspace)
        agents = {
            'defaults': {
                'model': {'primary': model_ref},
                'workspace': workspace,
            }
        }
        if agent_id != 'main':
            agents['list'] = [
                {
                    'id': agent_id,
                    'model': {'primary': model_ref},
                    'workspace': workspace,
                }
            ]

        provider_config = {
            'baseUrl': base_url,
            'apiKey': '$OPENAI_API_KEY',
            'api': os.environ.get('OPENCLAW_PROVIDER_API', 'openai-completions'),
            'models': [
                {
                    'id': self.model,
                    'name': self.model,
                    'reasoning': True,
                    'input': ['text'],
                    'contextWindow': int(os.environ.get('OPENCLAW_CONTEXT_WINDOW', '128000')),
                    'maxTokens': int(os.environ.get('OPENCLAW_MAX_TOKENS', '16384')),
                }
            ],
        }
        # Provider-scoped HTTP timeout is only valid on OpenClaw >= 2026.4.26.
        # Agent run ceiling is handled separately via ``--timeout`` in _build_argv.
        provider_timeout = os.environ.get('OPENCLAW_PROVIDER_TIMEOUT_SECONDS', "1200")
        if provider_timeout and self._supports_provider_timeout_seconds():
            provider_config['timeoutSeconds'] = int(provider_timeout)

        config = {
            'models': {
                'mode': 'merge',
                'providers': {
                    self.provider: provider_config,
                },
            },
            'agents': agents,
        }

        self.openclaw_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.openclaw_config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

        if not self._runtime_config_written:
            sessions_dir = self.openclaw_home / 'agents' / agent_id / 'sessions'
            shutil.rmtree(sessions_dir, ignore_errors=True)
            sessions_dir.mkdir(parents=True, exist_ok=True)
            self._runtime_config_written = True

    def _default_parse(self, stdout: str, stderr: str) -> str:
        if not self.json_output:
            return stdout.strip()

        # 低版本 OpenClaw 在某些错误路径会 exit 0、stdout 为空，并把 JSON envelope
        # 混在 stderr 日志后面；不能把这种情况静默解析成空字符串。
        data = self._load_openclaw_json(stdout)
        parsed_from_stderr = False
        if data is None and not stdout.strip():
            data = self._load_openclaw_json(stderr)
            parsed_from_stderr = data is not None
        if data is None:
            return stdout.strip()

        if not isinstance(data, dict):
            return stdout.strip()

        # OpenClaw envelope: sessionId lives under meta.agentMeta.
        meta = data.get('meta') if isinstance(data.get('meta'), dict) else {}
        agent_meta = meta.get('agentMeta') if isinstance(meta.get('agentMeta'), dict) else {}
        sid = (
            agent_meta.get('sessionId')
            or meta.get('sessionId')
            or data.get('sessionId')
            or data.get('session_id')
        )
        if sid:
            self._cli_session_id = sid

        # Reply text: prefer the canonical "final visible" field, then
        # the payloads[*].text array, then a handful of common fallbacks.
        text = meta.get('finalAssistantVisibleText') or meta.get('finalAssistantRawText')
        if isinstance(text, str) and text:
            return text

        payloads = data.get('payloads')
        if isinstance(payloads, list):
            parts = [
                p.get('text', '') for p in payloads
                if isinstance(p, dict) and isinstance(p.get('text'), str)
            ]
            joined = '\n'.join(p for p in parts if p)
            if joined:
                if parsed_from_stderr and 'isError=true' in stderr:
                    raise RuntimeError(f'OpenClaw error: {joined}')
                return joined

        for key in ('reply', 'message', 'response', 'output', 'text', 'result', 'content'):
            value = data.get(key)
            if isinstance(value, str) and value:
                if parsed_from_stderr and 'isError=true' in stderr:
                    raise RuntimeError(f'OpenClaw error: {value}')
                return value
            if isinstance(value, list):
                blocks = [
                    b.get('text', '') for b in value
                    if isinstance(b, dict) and b.get('type') == 'text'
                ]
                joined = '\n'.join(b for b in blocks if b)
                if joined:
                    if parsed_from_stderr and 'isError=true' in stderr:
                        raise RuntimeError(f'OpenClaw error: {joined}')
                    return joined
        if parsed_from_stderr and stderr.strip():
            raise RuntimeError(f'OpenClaw error: {stderr.strip()[:2000]}')
        return json.dumps(data, ensure_ascii=False)
