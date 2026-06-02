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

    from lagent.adapters.openclaw import OpenClawAdapter

    agent = OpenClawAdapter(thinking='medium', timeout=120)
    r1 = await agent("What is 2+2?")
    r2 = await agent("Now multiply by 3")  # multi-turn via --session-id
"""

import json
import os
import shlex
from typing import List, Optional

from .cli_adapter import CLIAgentAdapter


class OpenClawAdapter(CLIAgentAdapter):
    """Wraps the ``openclaw`` CLI as an :class:`AsyncExternalAgent`.

    Args:
        thinking: Thinking level (``off`` / ``minimal`` / ``low`` /
            ``medium`` / ``high`` / ``xhigh``).
        agent_id: OpenClaw agent id (``--agent``). Default: ``"main"``.
        json_output: Pass ``--json`` and parse the JSON envelope to
            extract ``sessionId`` (multi-turn) and the reply text.
            Default: True.
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
        thinking: str = 'medium',
        agent_id: Optional[str] = 'main',
        json_output: bool = True,
        nvm_dir: Optional[str] = None,
        node_version: str = '22',
        binary: str = 'openclaw',
        **kwargs,
    ):
        kwargs.setdefault('name', 'openclaw')
        kwargs.setdefault('description', 'OpenClaw personal AI assistant')
        super().__init__(binary=binary, **kwargs)
        self.thinking = thinking
        self.agent_id = agent_id
        self.json_output = json_output
        self.nvm_dir = nvm_dir
        self.node_version = node_version
        self._cli_session_id: Optional[str] = None

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

    def _build_argv(self, task: str) -> List[str]:
        cli_args = ['agent', '--local', '--message', task, '--thinking', self.thinking]
        if self.json_output:
            cli_args.append('--json')
        if self.agent_id:
            cli_args.extend(['--agent', self.agent_id])
        if self._cli_session_id:
            cli_args.extend(['--session-id', self._cli_session_id])
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

    def _default_parse(self, stdout: str, stderr: str) -> str:
        if not self.json_output:
            return stdout.strip()

        try:
            data = json.loads(stdout.strip())
        except json.JSONDecodeError:
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
                return joined

        for key in ('reply', 'message', 'response', 'output', 'text', 'result', 'content'):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list):
                blocks = [
                    b.get('text', '') for b in value
                    if isinstance(b, dict) and b.get('type') == 'text'
                ]
                joined = '\n'.join(b for b in blocks if b)
                if joined:
                    return joined
        return json.dumps(data, ensure_ascii=False)
