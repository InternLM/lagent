"""SDK Agent Adapter — wraps external Python SDK agents as lagent Agents.

For frameworks that expose a Python API (LangChain, CrewAI, OpenAI Agents,
etc.), subclass ``SDKAgentAdapter`` and implement two methods:
``create_sdk_agent()`` and ``invoke_sdk_agent()``.

Example::

    class LangchainAdapter(SDKAgentAdapter):
        def create_sdk_agent(self, config):
            from langchain.agents import AgentExecutor, create_react_agent
            ...
            return AgentExecutor(agent=agent, tools=tools)

        def invoke_sdk_agent(self, agent, task, **kwargs):
            result = agent.invoke({"input": task})
            return result["output"]

        async def invoke_sdk_agent_async(self, agent, task, **kwargs):
            result = await agent.ainvoke({"input": task})
            return result["output"]

    adapter = LangchainAdapter(
        name="langchain-react",
        sdk_module="langchain.agents",
        sdk_config={"model_name": "gpt-4", "tools": ["search"]},
    )
    result = await adapter("Research quantum computing trends")
"""

import asyncio
import importlib
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

from .base import AsyncExternalAgent


class SDKAgentAdapter(AsyncExternalAgent):
    """Wraps an external agent accessible via Python SDK.

    Subclasses must implement ``create_sdk_agent()`` and
    ``invoke_sdk_agent()``. Optionally override
    ``invoke_sdk_agent_async()`` for native async support.

    Args:
        sdk_module: Dotted module path to verify during setup.
            Example: ``"langchain.agents"``
        sdk_config: Configuration dict passed to ``create_sdk_agent()``.
        **kwargs: Passed to AsyncExternalAgent (name, working_dir,
            env_vars, timeout, proxy, hooks).
    """

    def __init__(
        self,
        sdk_module: Optional[str] = None,
        sdk_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sdk_module = sdk_module
        self.sdk_config = sdk_config or {}
        self._sdk_agent: Any = None

    def setup(self) -> None:
        """Verify SDK is importable, then create the agent instance."""
        if self.sdk_module:
            try:
                importlib.import_module(self.sdk_module)
            except ImportError as exc:
                raise RuntimeError(
                    f"SDK module '{self.sdk_module}' not importable: {exc}. "
                    f"Install the required package."
                ) from exc
        if self._sdk_agent is None:
            self._sdk_agent = self.create_sdk_agent(self.sdk_config)

    @abstractmethod
    def create_sdk_agent(self, config: Dict[str, Any]) -> Any:
        """Instantiate the external framework's agent object.

        Args:
            config: The ``sdk_config`` dict from ``__init__``.

        Returns:
            The external agent object (framework-specific type).
        """

    @abstractmethod
    def invoke_sdk_agent(self, agent: Any, task: str, **kwargs) -> str:
        """Run the external agent synchronously.

        Args:
            agent: Object returned by ``create_sdk_agent()``.
            task: The task/prompt string.

        Returns:
            Textual result from the external agent.
        """

    async def invoke_sdk_agent_async(
        self, agent: Any, task: str, **kwargs
    ) -> str:
        """Async version. Default: runs sync invoke in a thread executor.

        Override this if the SDK provides native async support.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke_sdk_agent(agent, task, **kwargs)
        )

    async def run_external_async(self, task: str, **kwargs) -> str:
        return await self.invoke_sdk_agent_async(
            self._sdk_agent, task, **kwargs
        )
