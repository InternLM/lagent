"""OpenCode CLI adapter for lagent."""

import asyncio
import json
import os
from typing import Dict, List, Optional

from .cli_adapter import CLIAgentAdapter


class OpenCodeCLIAdapter(CLIAgentAdapter):
    """Wraps the ``opencode run`` CLI as an AsyncExternalAgent.

    Each call invokes::

        opencode run --format json --dir <working_dir> --model <provider/model> -- <task>

    OpenCode emits JSON-lines events. The adapter captures ``sessionID`` from
    those events and resumes subsequent calls with ``--session <sessionID>``.

    The model endpoint is supplied via ``OPENCODE_CONFIG_CONTENT``. When a
    :class:`SessionClient` proxy is attached, the OpenCode ``openai`` provider
    is pointed at ``<proxy.url>/v1`` and uses the proxy session key.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "openai-compatible",
        provider_npm: Optional[str] = "@ai-sdk/openai-compatible",
        agent: Optional[str] = None,
        system_prompt: Optional[str] = None,
        dangerously_skip_permissions: bool = False,
        thinking: bool = False,
        share: str = "disabled",
        autoupdate: bool = False,
        opencode_home: str = "/tmp/opencode-home",
        cwd: Optional[str] = None,
        binary: str = "opencode",
        **kwargs,
    ):
        kwargs.setdefault('name', 'opencode-cli')
        kwargs.setdefault('description', 'OpenCode (CLI mode)')
        if cwd is not None:
            working_dir = kwargs.get('working_dir')
            if working_dir is not None and working_dir != cwd:
                raise ValueError(
                    'Specify only one of cwd and working_dir, or use the same value.'
                )
            kwargs['working_dir'] = cwd
        super().__init__(binary=binary, **kwargs)
        self.model = model
        self.provider = provider
        self.provider_npm = provider_npm
        self.agent = agent
        self.system_prompt = system_prompt
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.thinking = thinking
        self.share = share
        self.autoupdate = autoupdate
        self.opencode_home = opencode_home
        self._cli_session_id: Optional[str] = None
        self._last_events: List[dict] = []

    def _model_arg(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        env = env or os.environ
        model = self.model or env.get("RL_LLM_MODEL")
        if not model:
            return None
        if "/" in model:
            return model
        return f"{self.provider}/{model}"

    def _model_id(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        env = env or os.environ
        model = self.model or env.get("RL_LLM_MODEL")
        if not model:
            return None
        return model.split("/", 1)[1] if "/" in model else model

    @staticmethod
    def _with_v1(base_url: str) -> str:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url
        return f"{base_url}/v1"

    def _opencode_config(self, env: Dict[str, str]) -> dict:
        raw = env.get("OPENCODE_CONFIG_CONTENT")
        if raw:
            try:
                config = json.loads(raw)
                if not isinstance(config, dict):
                    config = {}
            except json.JSONDecodeError:
                config = {}
        else:
            config = {}

        model_arg = self._model_arg(env)
        model_id = self._model_id(env)
        provider = self.provider

        if self.proxy:
            base_url = self._with_v1(self.proxy.url)
            api_key = f"sk-proxy-{self.session_id}"
        else:
            base_url = env.get("RL_LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or ""
            api_key = env.get("RL_LLM_API_KEY") or env.get("OPENAI_API_KEY") or "sk-admin"

        if model_arg:
            config["model"] = model_arg
        config["enabled_providers"] = [provider]
        config["share"] = self.share
        config["autoupdate"] = self.autoupdate

        provider_cfg = config.setdefault("provider", {}).setdefault(provider, {})
        provider_cfg.setdefault("name", provider)
        if self.provider_npm:
            provider_cfg.setdefault("npm", self.provider_npm)
        options = provider_cfg.setdefault("options", {})
        if base_url:
            options["baseURL"] = base_url
        if api_key:
            options["apiKey"] = api_key

        if model_id:
            models = provider_cfg.setdefault("models", {})
            models.setdefault(
                model_id,
                {
                    "name": model_id,
                    "tool_call": True,
                    "reasoning": True,
                },
            )

        if self.system_prompt:
            agent_name = self.agent or "build"
            config.setdefault("agent", {}).setdefault(agent_name, {})[
                "prompt"
            ] = self.system_prompt
            config.setdefault("default_agent", agent_name)

        return config

    def _build_env(self) -> dict:
        env = super()._build_env()
        home = self.opencode_home
        env["HOME"] = home
        env["XDG_DATA_HOME"] = os.path.join(home, ".local", "share")
        env["XDG_CONFIG_HOME"] = os.path.join(home, ".config")
        env["XDG_CACHE_HOME"] = os.path.join(home, ".cache")
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            self._opencode_config(env), ensure_ascii=False
        )
        return env

    def _build_argv(
        self, task: str, env: Optional[Dict[str, str]] = None
    ) -> List[str]:
        argv = [self.binary, 'run', '--format', 'json']
        if self.dangerously_skip_permissions:
            argv += ['--dangerously-skip-permissions']
        if self.working_dir:
            argv += ['--dir', self.working_dir]
        env = env or self._build_env()
        model_arg = self._model_arg(env)
        if model_arg:
            argv += ['--model', model_arg]
        if self.agent:
            argv += ['--agent', self.agent]
        if self.thinking:
            argv += ['--thinking']
        if self._cli_session_id:
            argv += ['--session', self._cli_session_id]
        argv += self.extra_args
        argv += ['--', task]
        return argv

    async def run_external_async(self, task: str, **kwargs) -> str:
        env = self._build_env()
        argv = self._build_argv(task, env)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir,
            env=env,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"`{self.binary}` timed out after {self.timeout}s")

        stdout = stdout_b.decode('utf-8', errors='replace')
        stderr = stderr_b.decode('utf-8', errors='replace')

        try:
            result = self.parse_output(stdout, stderr)
        except Exception as exc:
            if stderr.strip():
                raise RuntimeError(f"{exc}\nstderr: {stderr[:2000]}") from exc
            raise

        if proc.returncode != 0:
            detail = result or stderr.strip() or stdout.strip()
            raise RuntimeError(
                f"`{self.binary}` exited with code {proc.returncode}: "
                f"{detail[:2000]}"
            )

        if isinstance(result, str) and len(result) > self.max_output_chars:
            return result[: self.max_output_chars] + '\n...(truncated)'
        return result

    @staticmethod
    def _extract_session_id(event: dict) -> Optional[str]:
        props = event.get('properties') if isinstance(event.get('properties'), dict) else {}
        for key in ('sessionID', 'session_id', 'sessionId'):
            if isinstance(event.get(key), str):
                return event[key]
            if isinstance(props.get(key), str):
                return props[key]
        session = event.get('session') or props.get('session')
        if isinstance(session, dict) and isinstance(session.get('id'), str):
            return session['id']
        return None

    @staticmethod
    def _error_message(payload: dict) -> str:
        error = payload.get('error')
        if isinstance(error, dict):
            data = error.get('data')
            if isinstance(data, dict) and data.get('message'):
                return str(data['message'])
            if error.get('message'):
                return str(error['message'])
            if error.get('name'):
                return json.dumps(error, ensure_ascii=False)
        if payload.get('message'):
            return str(payload['message'])
        return json.dumps(payload, ensure_ascii=False)

    def _default_parse(self, stdout: str, stderr: str) -> str:
        events: List[dict] = []
        text_by_id: Dict[str, str] = {}
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
            events.append(event)

            sid = self._extract_session_id(event)
            if sid:
                self._cli_session_id = sid

            payload = event.get('properties') if isinstance(event.get('properties'), dict) else event
            etype = event.get('type') or payload.get('type') or ''

            if etype in ('error', 'session.error', 'session.next.step.failed'):
                error_payload = payload

            if etype == 'message.part.updated':
                part = payload.get('part') if isinstance(payload.get('part'), dict) else {}
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    last_text = part['text']
            elif etype == 'message.part.delta':
                part_id = payload.get('partID') or payload.get('part_id')
                delta = payload.get('delta')
                if isinstance(part_id, str) and isinstance(delta, str):
                    text_by_id[part_id] = text_by_id.get(part_id, '') + delta
                    last_text = text_by_id[part_id]
            elif etype == 'session.next.text.delta':
                text_id = payload.get('textID') or payload.get('text_id')
                delta = payload.get('delta')
                if isinstance(text_id, str) and isinstance(delta, str):
                    text_by_id[text_id] = text_by_id.get(text_id, '') + delta
                    last_text = text_by_id[text_id]
            elif etype == 'session.next.text.ended' and isinstance(payload.get('text'), str):
                last_text = payload['text']
            elif isinstance(payload.get('message'), str):
                last_text = payload['message']
            elif isinstance(payload.get('text'), str) and etype not in (
                'session.next.reasoning.delta',
            ):
                last_text = payload['text']

        self._last_events = events

        if error_payload and not last_text:
            raise RuntimeError(
                f"opencode error: {self._error_message(error_payload)[:2000]}"
            )

        if last_text:
            return last_text
        if events:
            return stdout.strip()
        result = stdout.strip()
        if stderr.strip():
            result += f'\n[stderr]: {stderr.strip()}'
        return result

    def reset_session(self) -> None:
        """Forget the captured OpenCode session id; the next call starts fresh."""
        self._cli_session_id = None
