"""Terminus2 black-box adapter wrapping harbor's Terminus2.

This adapter wires :class:`harbor.agents.terminus_2.terminus_2.Terminus2` into
lagent's :class:`AsyncExternalAgent` contract.  Harbor owns the agent loop,
parser, prompt templates, LiteLLM call layer (with proper ``reraise=True``
retry semantics) and tmux-based terminal interaction; this adapter only
provides a local ``BaseEnvironment`` shim so harbor's ``TmuxSession`` can drive
the *current* container via subprocess instead of HTTP.

Harbor and its prompt templates are imported lazily inside ``setup()`` /
``run_external_async()`` so installing lagent without harbor on PYTHONPATH
keeps ``import lagent.adapters`` working.
"""

from __future__ import annotations
import asyncio
import copy
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from lagent.agents.agent import Agent
from lagent.utils import create_object
from .base import AsyncExternalAgent


class Terminus2Adapter(AsyncExternalAgent):
    """Wrap upstream ``harbor.agents.terminus_2.terminus_2.Terminus2``.

    Args:
        model (str | None): LiteLLM model name. Defaults to ``RL_LLM_MODEL`` or
            ``OPENAI_MODEL``.
        max_episodes (int | None): Max agent loop iterations.  Forwarded to
            harbor as ``max_turns``.
        parser_name (str): ``"json"`` or ``"xml"``.
        temperature (float): LiteLLM sampling temperature.
        reasoning_effort (str | None): One of ``"none"``, ``"minimal"``,
            ``"low"``, ``"medium"``, ``"high"``, ``"default"``.
        enable_summarize (bool): Allow harbor to summarize on
            ``ContextLengthExceededError``.  Default ``False`` so context
            overflow surfaces as a real error.
        record_terminal_session (bool): Record asciinema cast.  Default
            ``False`` to skip asciinema dependency.
        suppress_max_turns_warning (bool): Quiet harbor's per-turn warning.
        tmux_pane_width (int): tmux ``-x``.
        tmux_pane_height (int): tmux ``-y``.
        logging_dir (str | None): Directory for harbor's per-episode logs and
            ``trajectory.*.json``.  A fresh ``tempfile.mkdtemp`` is allocated
            per call when omitted.
        terminus2_kwargs (dict | None): Forwarded verbatim to harbor's
            ``Terminus2`` constructor.  Use this for advanced fields like
            ``model_info``, ``llm_call_kwargs``, ``max_thinking_tokens``,
            ``trajectory_config`` without growing the adapter signature.
        **kwargs: Forwarded to :class:`AsyncExternalAgent` (``timeout``,
            ``working_dir``, ``env_vars``, ``proxy``, ``hooks``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_episodes: Optional[int] = 100,
        parser_name: str = "json",
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
        enable_summarize: bool = False,
        record_terminal_session: bool = False,
        suppress_max_turns_warning: bool = True,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        logging_dir: Optional[str] = None,
        terminus2_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        proxy_cfg = kwargs.get("proxy")
        if isinstance(proxy_cfg, dict):
            kwargs["proxy"] = create_object(proxy_cfg)

        kwargs.setdefault("name", "terminus2")
        kwargs.setdefault("description", "Harbor Terminus2 black-box agent")
        kwargs.setdefault("working_dir", os.environ.get("TASK_WORKSPACE", "/app"))
        super().__init__(**kwargs)

        self.model = model or os.environ.get("RL_LLM_MODEL") or os.environ.get("OPENAI_MODEL", "")
        self.max_episodes = max_episodes
        self.parser_name = parser_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.enable_summarize = enable_summarize
        self.record_terminal_session = record_terminal_session
        self.suppress_max_turns_warning = suppress_max_turns_warning
        self.tmux_pane_width = tmux_pane_width
        self.tmux_pane_height = tmux_pane_height
        self.logging_dir = Path(logging_dir) if logging_dir else None
        self.terminus2_kwargs = dict(terminus2_kwargs or {})

        self._last_messages: list[dict[str, Any]] | None = None
        self._last_trajectory_steps: list[Any] | None = None
        self._last_context_metadata: dict[str, Any] | None = None
        self._last_failure_mode: str | None = None
        self._last_n_episodes: int | None = None

    def setup(self) -> None:
        try:
            import harbor.agents.terminus_2.terminus_2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Terminus2Adapter requires harbor on PYTHONPATH. "
                "Install harbor's source tree under PYTHONPATH (e.g., via the "
                "tb_pkg_install.sh build script that lagent's tb2-eval recipe "
                "uses) along with: pydantic, shortuuid, requests, pyyaml, "
                "tenacity, python-dotenv, litellm, jinja2, pathspec, packaging, "
                "logzero."
            ) from exc

        if not self.model:
            raise RuntimeError("Terminus2Adapter requires model or RL_LLM_MODEL.")

        self.model = _ensure_provider_prefix(self.model)

        if self.proxy is None:
            raise RuntimeError(
                "Terminus2Adapter requires a SessionClient proxy: all LLM calls are "
                "routed through it for token-level attribution. Pass `proxy=`."
            )

        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is required by Terminus2Adapter but was not found on PATH.")

    async def run_external_async(self, task: str, **kwargs) -> str:
        from harbor.agents.terminus_2.terminus_2 import Terminus2 as HarborTerminus2
        from harbor.models.agent.context import AgentContext

        local_env_cls = _get_local_sandbox_environment_cls()

        api_base = self.proxy.url
        api_key = f"sk-proxy-{self.session_id}"

        owns_logging_dir = self.logging_dir is None
        logs_dir = self.logging_dir or Path(tempfile.mkdtemp(prefix="harbor-terminus2-"))
        logs_dir.mkdir(parents=True, exist_ok=True)

        env = local_env_cls(
            workspace=self.working_dir or "/app",
            trial_dir=logs_dir,
            env_vars=self.env_vars or None,
        )

        env_updates: dict[str, str | None] = {
            **{k: v for k, v in (self.env_vars or {}).items()},
            "OPENAI_API_KEY": api_key,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "OPENAI_BASE_URL": api_base,
            "ANTHROPIC_BASE_URL": api_base,
        }

        try:
            with _temporary_env(env_updates):
                agent = HarborTerminus2(
                    logs_dir=logs_dir,
                    model_name=self.model,
                    api_base=api_base,
                    max_turns=self.max_episodes,
                    parser_name=self.parser_name,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    enable_summarize=self.enable_summarize,
                    record_terminal_session=self.record_terminal_session,
                    suppress_max_turns_warning=self.suppress_max_turns_warning,
                    tmux_pane_width=self.tmux_pane_width,
                    tmux_pane_height=self.tmux_pane_height,
                    session_id=self.session_id,
                    **self.terminus2_kwargs,
                )

                await env.start(force_build=False)
                await agent.setup(env)

                context = AgentContext()
                try:
                    run_coro = agent.run(task, env, context)
                    if self.timeout is not None:
                        await asyncio.wait_for(run_coro, timeout=self.timeout)
                    else:
                        await run_coro
                finally:
                    # Capture state even on timeout/exception so a partial
                    # trajectory still surfaces in ``state_dict`` / ``get_messages``.
                    self._capture_state(agent, context)
                return self._format_result(agent)
        finally:
            try:
                await env.stop(delete=False)
            except Exception:
                pass
            if owns_logging_dir:
                await asyncio.to_thread(shutil.rmtree, str(logs_dir), True)

    def state_dict(self, prefix: str = "", destination=None) -> Dict[str, Any]:
        dest = Agent.state_dict(self, prefix=prefix, destination=destination)
        try:
            dest[prefix + "llm_trace"] = _json_safe(self.get_messages())
        except Exception:
            pass
        dest[prefix + "terminus2.messages"] = _json_safe(self._last_messages)
        dest[prefix + "terminus2.trajectory_steps"] = _json_safe(self._last_trajectory_steps)
        dest[prefix + "terminus2.context_metadata"] = _json_safe(self._last_context_metadata)
        dest[prefix + "terminus2.failure_mode"] = self._last_failure_mode
        dest[prefix + "terminus2.n_episodes"] = self._last_n_episodes
        return dest

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _capture_state(self, agent: Any, context: Any) -> None:
        chat = getattr(agent, "_chat", None)
        if chat is not None:
            messages = getattr(chat, "messages", None)
            if isinstance(messages, list):
                self._last_messages = list(messages)
        trajectory = getattr(agent, "_trajectory_steps", None)
        if isinstance(trajectory, list):
            self._last_trajectory_steps = list(trajectory)
        metadata = getattr(context, "metadata", None)
        if metadata is not None:
            self._last_context_metadata = dict(metadata)
        self._last_n_episodes = getattr(agent, "_n_episodes", None)

    def _format_result(self, agent: Any) -> str:
        chat = getattr(agent, "_chat", None)
        if chat is not None:
            messages = getattr(chat, "messages", None) or []
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        return content
        return "Terminus2 finished."

    def get_messages(self) -> List[Dict[str, list]]:
        """Return the LLM trace recorded by the proxy (empty if no proxy)."""
        return self.proxy.get_messages() if self.proxy else []


# ─────────────────────────────────────────────────────────────────────
# Module-private helpers
# ─────────────────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _ensure_provider_prefix(model: str) -> str:
    """Prefix ``openai/`` when ``model`` carries no explicit LiteLLM provider route.

    LiteLLM dispatches by a ``provider/model`` prefix. A bare name
    (``deepseek-v4-flash``) or a HuggingFace-style id (``deepseek-ai/DeepSeek-V3``)
    has no provider, so it is routed through the OpenAI-compatible path that
    fronts the proxy. A model whose first ``/``-segment is already a known LiteLLM
    provider (``openai/...``, ``anthropic/...``) is returned unchanged, making
    this idempotent.

    Args:
        model (str): The configured model name.

    Returns:
        str: ``model`` unchanged when it already starts with a known LiteLLM
            provider, else ``"openai/" + model``.
    """
    if not model:
        return model
    import litellm

    providers = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    if model.split("/", 1)[0] in providers:
        return model
    return f"openai/{model}"


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old_values: dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            if value is None:
                continue
            old_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


_LOCAL_SANDBOX_ENVIRONMENT_CLS: type | None = None


def _get_local_sandbox_environment_cls() -> type:
    """Lazily build the harbor ``BaseEnvironment`` subclass.

    Defined as a factory so that ``import lagent.adapters.terminus2`` does not
    require harbor to be on PYTHONPATH; harbor is only needed when
    ``run_external_async`` actually fires.

    Returns:
        type: Cached ``LocalSandboxEnvironment`` class.
    """
    global _LOCAL_SANDBOX_ENVIRONMENT_CLS
    if _LOCAL_SANDBOX_ENVIRONMENT_CLS is not None:
        return _LOCAL_SANDBOX_ENVIRONMENT_CLS

    from harbor.environments.base import BaseEnvironment, ExecResult
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    class LocalSandboxEnvironment(BaseEnvironment):
        """Run harbor's ``BaseEnvironment`` operations as local subprocess calls.

        Harbor's :class:`~harbor.environments.base.BaseEnvironment` abstracts
        "run a command in the task container".  Concrete subclasses ship in
        harbor for Docker/GKE/Modal/e2b/Daytona/PJLab — all assume harbor runs
        *outside* the container and reaches in over an SDK or HTTP.

        When :class:`Terminus2Adapter` runs inside the lagent sandbox daemon,
        it is *already* in the task container.  Harbor's ``environment.exec``
        therefore degenerates to a local subprocess call,
        ``upload/download_file`` to ``shutil.copy`` (source and destination are
        both on this machine), and ``start/stop`` to no-ops.

        Args:
            workspace (str): Path the agent's terminal session starts in.
                Forwarded as ``cwd`` to subprocess calls when the caller does
                not override.
            trial_dir (Path | None): Directory used for harbor logging
                artefacts (``trial_paths.agent_dir`` etc.).  A fresh
                ``tempfile.mkdtemp`` dir is allocated when omitted.
            default_user (str | int | None): Forwarded to harbor's call sites
                that ask for ``user``; ignored by ``exec`` (subprocess inherits
                the daemon's effective user).
            env_vars (dict[str, str] | None): Persistent env injected into
                every ``exec``.
        """

        def __init__(
            self,
            workspace: str = "/app",
            *,
            trial_dir: Path | None = None,
            default_user: str | int | None = None,
            env_vars: dict[str, str] | None = None,
        ) -> None:
            if trial_dir is None:
                trial_dir = Path(tempfile.mkdtemp(prefix="harbor-local-env-"))
            trial_dir.mkdir(parents=True, exist_ok=True)

            super().__init__(
                environment_dir=trial_dir,
                environment_name="lagent-local",
                session_id=trial_dir.name,
                trial_paths=TrialPaths(trial_dir=trial_dir),
                task_env_config=EnvironmentConfig(),
                persistent_env=env_vars,
                suppress_override_warnings=True,
            )
            self._workspace = workspace
            self.default_user = default_user

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.DOCKER

        @property
        def is_mounted(self) -> bool:
            return False

        @property
        def supports_gpus(self) -> bool:
            return False

        @property
        def can_disable_internet(self) -> bool:
            return False

        async def start(self, force_build: bool = False) -> None:
            return None

        async def stop(self, delete: bool = False) -> None:
            return None

        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            merged_env = os.environ.copy()
            merged = self._merge_env(env)
            if merged:
                merged_env.update(merged)

            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd or self._workspace,
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                if timeout_sec is not None and timeout_sec > 0:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
                else:
                    stdout, stderr = await proc.communicate()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecResult(
                    stdout="",
                    stderr=f"command timed out after {timeout_sec}s: {command}",
                    return_code=124,
                )
            return ExecResult(
                stdout=(stdout or b"").decode("utf-8", errors="replace"),
                stderr=(stderr or b"").decode("utf-8", errors="replace"),
                return_code=proc.returncode if proc.returncode is not None else -1,
            )

        async def upload_file(self, source_path: Path | str, target_path: str) -> None:
            src = Path(source_path).resolve()
            dst = Path(target_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, src, dst)

        async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
            src = Path(source_dir).resolve()
            if not src.is_dir():
                raise NotADirectoryError(f"source is not a directory: {src}")
            dst = Path(target_dir)
            await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)

        async def download_file(self, source_path: str, target_path: Path | str) -> None:
            src = Path(source_path)
            dst = Path(target_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, src, dst)

        async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
            src = Path(source_dir)
            if not src.is_dir():
                raise NotADirectoryError(f"source is not a directory: {src}")
            dst = Path(target_dir)
            await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)

        def _validate_definition(self) -> None:
            return None

    _LOCAL_SANDBOX_ENVIRONMENT_CLS = LocalSandboxEnvironment
    return LocalSandboxEnvironment
