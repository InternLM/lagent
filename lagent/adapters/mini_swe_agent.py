"""mini-SWE-agent adapter for lagent."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from .base import AsyncExternalAgent


class MiniSWEAgentAdapter(AsyncExternalAgent):
    """Wrap mini-SWE-agent as a lagent async external agent."""

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        cwd: Optional[str] = None,
        step_limit: int = 30,
        command_timeout: int = 300,
        trajectory_path: str = "/tmp/mini.traj.json",
        model_kwargs: Optional[dict[str, Any]] = None,
        mini_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        kwargs.setdefault("name", "mini-swe-agent")
        kwargs.setdefault("description", "mini-SWE-agent")
        super().__init__(**kwargs)
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.cwd = cwd or self.working_dir or os.getcwd()
        self.step_limit = step_limit
        self.command_timeout = command_timeout
        self.trajectory_path = Path(trajectory_path)
        self.model_kwargs = model_kwargs or {}
        self.mini_config = mini_config or {}

    def setup(self) -> None:
        os.environ.setdefault("MSWEA_CONFIGURED", "true")
        os.environ.setdefault("MSWEA_SILENT_STARTUP", "true")
        os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", "/tmp/mswea-config")
        try:
            import minisweagent  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("mini-swe-agent is required. Install with: pip install mini-swe-agent") from exc

    async def run_external_async(self, task: str, **kwargs) -> str:
        def run() -> str:
            from minisweagent.agents import get_agent
            from minisweagent.config import get_config_from_spec
            from minisweagent.environments import get_environment
            from minisweagent.models import get_model
            from minisweagent.utils.serialize import recursive_merge

            model_kwargs = {
                "drop_params": True,
                "custom_llm_provider": "openai",
                **self.model_kwargs,
                "api_base": self.proxy.url
                if self.proxy
                else self.api_base or os.environ.get("RL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", ""),
            }
            api_key = (
                f"sk-proxy-{self.session_id}"
                if self.proxy
                else self.api_key or os.environ.get("RL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
            )
            if api_key:
                model_kwargs["api_key"] = api_key

            config = recursive_merge(
                get_config_from_spec("mini"),
                {
                    "agent": {
                        "agent_class": "default",
                        "step_limit": self.step_limit,
                        "cost_limit": 0.0,
                        "output_path": self.trajectory_path,
                    },
                    "environment": {
                        "environment_class": "local",
                        "cwd": self.cwd,
                        "timeout": self.command_timeout,
                    },
                    "model": {
                        "model_class": "litellm",
                        "model_name": self.model,
                        "model_kwargs": model_kwargs,
                        "cost_tracking": "ignore_errors",
                    },
                },
                self.mini_config,
            )
            agent = get_agent(
                get_model(config=config["model"]),
                get_environment(config["environment"], default_type="local"),
                config["agent"],
                default_type="default",
            )
            result = agent.run(task, **kwargs)
            if result.get("submission"):
                return result["submission"]
            for message in reversed(agent.messages):
                if message.get("role") == "assistant" and message.get("content"):
                    return str(message["content"])
            return ""

        return await asyncio.to_thread(run)
