"""Thin Terminus2 adapter for lagent.

This module intentionally does not port Terminus2's parser, prompt, or agent
loop.  It wraps the upstream implementation from ``terminal-bench`` and exposes
it through lagent's black-box ``AsyncExternalAgent`` protocol.

The default terminal backend is a local tmux wrapper around
``lagent.actions.tmux_action.TmuxSession``.  That keeps the adapter usable
inside the lagent sandbox daemon, where the process is already running inside
the task container and does not need terminal-bench's Docker-backed session.
If you want the original terminal-bench Docker session, set
``terminal_backend="terminal_bench"`` and pass ``container_name``.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from lagent.actions.tmux_action import TmuxSession as LocalTmuxSession
from lagent.agents.agent import Agent
from lagent.utils import create_object

from .base import AsyncExternalAgent, _json_safe


class _LocalTerminusSession:
    """Sync, duck-typed session object consumed by terminal-bench Terminus2."""

    def __init__(
        self,
        session_name: str,
        *,
        pane_width: int = 160,
        pane_height: int = 40,
        working_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        kill_on_finish: bool = True,
    ) -> None:
        self._session_name = session_name
        self._kill_on_finish = kill_on_finish
        self._session = LocalTmuxSession(
            session_name=session_name,
            pane_width=pane_width,
            pane_height=pane_height,
            working_dir=working_dir,
            extra_env=extra_env,
        )

    def start(self) -> None:
        # LocalTmuxSession starts in its constructor.
        return None

    def stop(self) -> None:
        if not self._kill_on_finish:
            return
        subprocess.run(
            f"tmux kill-session -t {shlex.quote(self._session_name)}",
            shell=True,
            capture_output=True,
            text=True,
        )

    def is_session_alive(self) -> bool:
        return asyncio.run(self._session.is_session_alive())

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        asyncio.run(
            self._session.send_keys(
                keys,
                block=block,
                min_timeout_sec=min_timeout_sec,
                max_timeout_sec=max_timeout_sec,
            )
        )

    def capture_pane(self, capture_entire: bool = False) -> str:
        return asyncio.run(self._session.capture_pane(capture_entire=capture_entire))

    def get_incremental_output(self) -> str:
        return asyncio.run(self._session.get_incremental_output())

    def get_asciinema_timestamp(self) -> float:
        return 0.0


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


def _object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {k: _object_to_dict(v) for k, v in vars(value).items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_object_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: _object_to_dict(v) for k, v in value.items()}
    return value


class Terminus2Adapter(AsyncExternalAgent):
    """Wrap upstream ``terminal_bench.agents.terminus_2.Terminus2``.

    Args:
        model: LiteLLM model name. Defaults to ``RL_LLM_MODEL`` or
            ``OPENAI_MODEL``.
        base_url: LiteLLM ``api_base``. Defaults to ``RL_LLM_BASE_URL`` or
            ``OPENAI_BASE_URL``. If ``proxy`` is set, the proxy URL is used.
        api_key: API key exposed to LiteLLM through env vars. If ``proxy`` is
            set, a synthetic ``sk-proxy-<session>`` key is used.
        terminal_backend: ``"local"`` for lagent's in-container tmux session,
            or ``"terminal_bench"`` for terminal-bench's Docker session.
        container_name: Required when ``terminal_backend="terminal_bench"``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_episodes: Optional[int] = 30,
        parser_name: str = "json",
        temperature: float = 0.7,
        terminal_backend: Literal["local", "terminal_bench"] = "local",
        container_name: Optional[str] = None,
        commands_path: Optional[str] = None,
        disable_recording: bool = True,
        user: str = "",
        pane_width: int = 160,
        pane_height: int = 40,
        session_name: Optional[str] = None,
        auto_start_session: bool = True,
        kill_session_on_finish: bool = True,
        logging_dir: Optional[str] = None,
        require_tmux: bool = True,
        **kwargs,
    ) -> None:
        proxy_cfg = kwargs.get("proxy")
        if isinstance(proxy_cfg, dict):
            kwargs["proxy"] = create_object(proxy_cfg)

        kwargs.setdefault("name", "terminus2")
        kwargs.setdefault("description", "Terminal-Bench Terminus2 agent")
        kwargs.setdefault("working_dir", os.environ.get("TASK_WORKSPACE", "/app"))
        super().__init__(**kwargs)

        self.model = model or os.environ.get("RL_LLM_MODEL") or os.environ.get("OPENAI_MODEL", "")
        self.base_url = base_url or os.environ.get("RL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key = api_key or os.environ.get("RL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.max_episodes = max_episodes
        self.parser_name = parser_name
        self.temperature = temperature
        self.terminal_backend = terminal_backend
        self.container_name = container_name
        self.commands_path = Path(commands_path) if commands_path else None
        self.disable_recording = disable_recording
        self.user = user
        self.pane_width = pane_width
        self.pane_height = pane_height
        self.session_name = session_name or f"terminus2-{self.session_id}"
        self._explicit_session_name = session_name is not None
        self.auto_start_session = auto_start_session
        self.kill_session_on_finish = kill_session_on_finish
        self.logging_dir = Path(logging_dir) if logging_dir else None
        self.require_tmux = require_tmux
        self._run_index = 0

        self._last_result: Any = None
        self._last_result_dict: dict[str, Any] | None = None
        self._last_session_name: str | None = None

    def setup(self) -> None:
        try:
            import terminal_bench.agents.terminus_2.terminus_2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Terminus2Adapter requires terminal-bench to be installed or added to PYTHONPATH. "
                "Install with `pip install terminal-bench` or inject the terminal-bench repo into PYTHONPATH."
            ) from exc

        if not self.model:
            raise RuntimeError("Terminus2Adapter requires model or RL_LLM_MODEL.")

        if self.terminal_backend == "local":
            if self.require_tmux and shutil.which("tmux") is None:
                raise RuntimeError("tmux is required by Terminus2Adapter but was not found on PATH.")
        elif self.terminal_backend == "terminal_bench":
            if not self.container_name:
                raise RuntimeError("container_name is required when terminal_backend='terminal_bench'.")
        else:
            raise ValueError(f"Unsupported terminal_backend: {self.terminal_backend}")

    async def run_external_async(self, task: str, **kwargs) -> str:
        run_coro = asyncio.to_thread(self._run_task_sync, task)
        if self.timeout is None:
            return await run_coro
        return await asyncio.wait_for(run_coro, timeout=self.timeout)

    def _run_task_sync(self, instruction: str) -> str:
        from terminal_bench.agents.terminus_2.terminus_2 import Terminus2

        api_base = self.proxy.url if self.proxy is not None else (self.base_url or None)
        api_key = f"sk-proxy-{self.session_id}" if self.proxy is not None else self.api_key
        env_updates = {
            **self.env_vars,
            "OPENAI_API_KEY": api_key or None,
            "ANTHROPIC_API_KEY": api_key or None,
            "ANTHROPIC_AUTH_TOKEN": api_key or None,
            "OPENAI_BASE_URL": api_base,
            "ANTHROPIC_BASE_URL": api_base,
        }

        session = self._build_session()
        try:
            if self.auto_start_session and hasattr(session, "start"):
                session.start()

            if self.logging_dir is not None:
                self.logging_dir.mkdir(parents=True, exist_ok=True)

            with _temporary_env(env_updates):
                agent = Terminus2(
                    model_name=self.model,
                    max_episodes=self.max_episodes,
                    parser_name=self.parser_name,
                    api_base=api_base,
                    temperature=self.temperature,
                )
                result = agent.perform_task(
                    instruction=instruction,
                    session=session,
                    logging_dir=self.logging_dir,
                    time_limit_seconds=self.timeout,
                )

            self._last_result = result
            self._last_result_dict = _json_safe(_object_to_dict(result))
            return self._format_result(result)
        finally:
            if hasattr(session, "stop"):
                session.stop()

    def _build_session(self):
        self._last_session_name = (
            self.session_name if self._explicit_session_name else f"{self.session_name}-{self._run_index}"
        )
        self._run_index += 1

        if self.terminal_backend == "local":
            return _LocalTerminusSession(
                self._last_session_name,
                pane_width=self.pane_width,
                pane_height=self.pane_height,
                working_dir=self.working_dir,
                extra_env=self.env_vars,
                kill_on_finish=self.kill_session_on_finish,
            )

        import docker
        from terminal_bench.terminal.tmux_session import TmuxSession

        container = docker.from_env().containers.get(self.container_name)
        return TmuxSession(
            session_name=self._last_session_name,
            container=container,
            commands_path=self.commands_path,
            disable_recording=self.disable_recording,
            user=self.user,
        )

    @staticmethod
    def _format_result(result: Any) -> str:
        result_dict = _object_to_dict(result)
        if isinstance(result_dict, dict):
            failure_mode = result_dict.get("failure_mode")
            if isinstance(failure_mode, dict):
                failure_mode = failure_mode.get("value") or failure_mode.get("name") or str(failure_mode)
            return f"Terminus2 finished. failure_mode={failure_mode}"
        return "Terminus2 finished."

    def state_dict(self, prefix="", destination=None) -> Dict[str, Any]:
        dest = Agent.state_dict(self, prefix=prefix, destination=destination)
        if self.proxy is not None and hasattr(self.proxy, "get_messages"):
            try:
                dest[prefix + "llm_trace"] = _json_safe(self.proxy.get_messages())
            except Exception:
                pass
        dest[prefix + "terminus2.result"] = _json_safe(self._last_result_dict)
        dest[prefix + "terminus2.session_name"] = self._last_session_name
        return dest

    def get_messages(self, prefix="", destination=None) -> Dict[str, Any]:
        dest = super().get_messages(prefix=prefix, destination=destination)
        dest[prefix + "terminus2.result"] = _json_safe(self._last_result_dict)
        dest[prefix + "terminus2.session_name"] = self._last_session_name
        return dest
