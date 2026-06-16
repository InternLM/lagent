"""Base classes for external agent adapters.

Provides ``BaseExternalAgent`` and ``AsyncExternalAgent`` — abstract base
classes that wrap external agent frameworks (CLI tools or Python SDKs)
into lagent's Agent protocol.

These adapters are fully compatible with lagent's ecosystem: they can be
used as ``agent_engine`` in InterclawApp, placed into Sequential chains,
managed by AgentService, and discovered by AgentLoader.

Key design:
    - ``forward()`` = execute external agent, return final output as AgentMessage
    - ``state_dict()`` = memory + LLM trace from Proxy (if enabled)
    - ``llm=None`` because external frameworks bring their own reasoning engine
    - ``setup()`` is lazy (called on first forward, not at init)
    - Proxy integration via ``_build_env()`` injects base_url + session key

Usage::

    class MyCLIAgent(AsyncExternalAgent):
        async def setup(self):
            ...  # verify binary exists

        async def run_external_async(self, task, **kwargs):
            ...  # subprocess call, return stdout

    agent = MyCLIAgent(name="my-agent", timeout=300)
    result = await agent("Fix the bug in main.py")
    trace = agent.state_dict().get('llm_trace', [])
"""

import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from lagent.agents.agent import Agent, AsyncAgentMixin
from lagent.schema import AgentMessage
from lagent.utils import create_object, ctx_session_id


class BaseExternalAgent(Agent):
    """Abstract base for wrapping external agent frameworks as lagent Agents.

    Subclasses implement ``setup()`` and ``run_external()``.
    The ``forward()`` method handles the lifecycle:
    setup → build env → run → wrap output as AgentMessage.

    This class does NOT require an LLM, memory, or aggregator from lagent.
    The external framework provides its own reasoning engine.

    Args:
        name: Agent name, used as AgentMessage.sender.
        description: Human-readable description.
        working_dir: Working directory for the external agent.
        env_vars: Extra environment variables for the external agent.
        timeout: Maximum execution time in seconds. None = no limit.
        proxy: Optional LLMProxyRecorder for trajectory capture.
        hooks: Optional hooks (same as Agent).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        working_dir: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        proxy: Any = None,
        **kwargs,
    ):
        # Don't pass llm, template, output_format, aggregator — not needed
        super().__init__(
            llm=None,
            name=name,
            description=description,
            hooks=kwargs.pop('hooks', None),
        )
        self.working_dir = working_dir
        self.env_vars = env_vars or {}
        self.timeout = timeout
        self.proxy = create_object(proxy)
        self.session_id = self.proxy and self.proxy.session_id or ctx_session_id.get() or uuid4().hex[:8]
        self._setup_done = False

    @abstractmethod
    def setup(self) -> None:
        """One-time initialization (verify binary / check SDK import).

        Called lazily on first ``forward()`` call. Must be idempotent.
        """

    @abstractmethod
    def run_external(self, task: str, **kwargs) -> str:
        """Execute the external agent synchronously.

        Args:
            task: The task/prompt string.

        Returns:
            The external agent's textual output.

        Raises:
            RuntimeError: If the external agent fails.
            TimeoutError: If execution exceeds self.timeout.
        """

    def _build_env(self) -> dict:
        """Build environment variables dict with proxy injection."""
        env = os.environ.copy()
        env.update(self.env_vars)
        if self.proxy:
            session_key = f"sk-proxy-{self.session_id}"
            env.update(
                {
                    'OPENAI_BASE_URL': self.proxy.openai_base_url,
                    'OPENAI_API_KEY': session_key,
                    'ANTHROPIC_BASE_URL': self.proxy.anthropic_base_url,
                    'ANTHROPIC_API_KEY': session_key,
                }
            )
        return env

    def _extract_task(self, messages: tuple) -> str:
        """Join multiple AgentMessage contents into a single task string."""
        parts = []
        for m in messages:
            if isinstance(m, AgentMessage):
                parts.append(str(m.content))
            else:
                parts.append(str(m))
        return '\n'.join(parts)

    def forward(self, *message: AgentMessage, **kwargs) -> Union[AgentMessage, str]:
        """Lagent Agent protocol implementation.

        Extracts task from messages, runs external agent, wraps result.
        """
        task = self._extract_task(message)
        if not self._setup_done:
            self.setup()
            self._setup_done = True

        try:
            output = self.run_external(task, **kwargs)
        except Exception as exc:
            return AgentMessage(
                sender=self.name,
                content=f"External agent failed: {exc}",
                extra_info={'error': str(exc), 'adapter': self.__class__.__name__},
            )

        return AgentMessage(
            sender=self.name,
            content=output,
            extra_info={'adapter': self.__class__.__name__, 'session_id': self.session_id},
        )

    def state_dict(self, prefix='', destination=None) -> Dict:
        raise NotImplementedError(
            "BaseExternalAgent does not implement state_dict. Subclasses should override if needed."
        )

    def load_state_dict(self, state_dict: Dict):
        raise NotImplementedError(
            "BaseExternalAgent does not implement load_state_dict. Subclasses should override if needed."
        )

    def get_messages(self) -> List[dict]:
        """Get the LLM trace from the proxy, if available."""
        return self.proxy.get_messages()


class AsyncExternalAgent(AsyncAgentMixin, BaseExternalAgent):
    """Async variant of BaseExternalAgent.

    Subclasses implement ``run_external_async()`` instead of
    ``run_external()``.
    """

    @abstractmethod
    async def run_external_async(self, task: str, **kwargs) -> str:
        """Async version of run_external."""

    def run_external(self, task: str, **kwargs) -> str:
        """Sync fallback — not used in async path."""
        raise NotImplementedError("Use run_external_async() for AsyncExternalAgent")

    async def forward(self, *message: AgentMessage, **kwargs) -> Union[AgentMessage, str]:
        task = self._extract_task(message)
        if not self._setup_done:
            self.setup()
            self._setup_done = True

        # Lazily start proxy if present
        if self.proxy and not self.proxy.is_running:
            await self.proxy.start()

        try:
            output = await self.run_external_async(task, **kwargs)
        except Exception as exc:
            return AgentMessage(
                sender=self.name,
                content=f"External agent failed: {exc}",
                extra_info={'error': str(exc), 'adapter': self.__class__.__name__},
            )

        return AgentMessage(
            sender=self.name,
            content=output,
            extra_info={'adapter': self.__class__.__name__, 'session_id': self.session_id},
        )
