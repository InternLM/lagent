"""OpenHands adapter — wraps the OpenHands Software Agent SDK as a lagent Agent.

Uses the ``openhands-sdk`` Python library (``openhands.sdk``) instead of a CLI
subprocess, providing:

- Real multi-turn via a persistent ``Conversation`` object (no ``--continue``
  hack — the conversation keeps its own event history across calls).
- Structured event access (MessageEvent, ActionEvent, ObservationEvent, ...).
- Built-in tools (terminal, file editor, task tracker, browser) and MCP servers.
- Full usage/cost tracking per turn via ``llm.metrics``.

Install with::

    pip install -U openhands-sdk openhands-tools

Usage::

    from lagent.adapters.openhands import OpenHandsAdapter

    agent = OpenHandsAdapter(model="anthropic/claude-sonnet-4-5-20250929")
    r1 = await agent("Read main.py and explain what it does")
    r2 = await agent("Now fix the bug")  # real multi-turn, same conversation
    print(r2.content)

With a proxy for trajectory capture, ``base_url`` / ``api_key`` are pointed at
the recorder so every LLM call is logged::

    from lagent.adapters.proxy import SessionClient

    proxy = SessionClient(real_api_key="...", real_base_url="...")
    agent = OpenHandsAdapter(model="openai/gpt-4o", proxy=proxy)
    await agent("Fix the bug")
    messages = agent.get_messages()
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from .base import AsyncExternalAgent

# Friendly tool name -> (module, attribute) for the built-in openhands-tools.
# Anything not listed here is passed straight to ``Tool(name=...)`` as an
# already-registered tool name.
_BUILTIN_TOOLS = {
    'terminal': ('openhands.tools.terminal', 'TerminalTool'),
    'bash': ('openhands.tools.terminal', 'TerminalTool'),
    'file_editor': ('openhands.tools.file_editor', 'FileEditorTool'),
    'str_replace_editor': ('openhands.tools.file_editor', 'FileEditorTool'),
    'task_tracker': ('openhands.tools.task_tracker', 'TaskTrackerTool'),
    'browser': ('openhands.tools.browser_use', 'BrowserToolSet'),
    'browser_use': ('openhands.tools.browser_use', 'BrowserToolSet'),
}

_DEFAULT_TOOLS = ['terminal', 'file_editor', 'task_tracker']


class OpenHandsAdapter(AsyncExternalAgent):
    """Wraps the OpenHands Software Agent SDK as a lagent Agent.

    Each ``forward()`` call reuses the same ``Conversation`` object, so
    OpenHands maintains full conversation history (its event log) internally.
    The blocking ``conversation.run()`` is executed in a thread executor so it
    plays nicely inside an async event loop (and alongside the proxy server).

    Args:
        model: LiteLLM model string (``provider/model``). Default:
            ``$LLM_MODEL`` or ``"anthropic/claude-sonnet-4-5-20250929"``.
        api_key: LLM API key. Default: ``$LLM_API_KEY``. Ignored when a
            ``proxy`` is attached (the session key is injected instead).
        base_url: Custom LLM base URL. Default: ``$LLM_BASE_URL``. Ignored
            when a ``proxy`` is attached (the proxy URL is injected instead).
        tools: Tools to enable — friendly names (``"terminal"``,
            ``"file_editor"``, ``"task_tracker"``, ``"browser"``), raw
            registered tool names, or pre-built ``Tool`` objects. Default:
            terminal + file_editor + task_tracker. Pass ``[]`` to disable all.
        mcp_config: MCP server config dict, e.g.
            ``{"mcpServers": {"fetch": {"command": "uvx", "args": [...]}}}``.
        usage_id: Identifier for LLM usage/metrics tracking. Default: name.
        max_iterations: Max agent iterations per ``run()``. Default: 500.
        stuck_detection: Enable OpenHands stuck detection. Default: True.
        persistence_dir: Directory to persist the conversation event log.
        conversation_id: Fixed conversation id — a ``uuid.UUID`` or a uuid
            string (coerced for you). Default: auto.
        verbose: If False (default), disable the rich console visualizer.
        llm_kwargs: Extra kwargs forwarded to ``LLM(...)``.
        agent_kwargs: Extra kwargs forwarded to ``Agent(...)``.
        conversation_kwargs: Extra kwargs forwarded to ``Conversation(...)``.
        **kwargs: Passed to AsyncExternalAgent (name, working_dir, timeout,
            proxy, hooks, env_vars).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        mcp_config: Optional[Dict[str, Any]] = None,
        usage_id: Optional[str] = None,
        max_iterations: int = 500,
        stuck_detection: bool = True,
        persistence_dir: Optional[str] = None,
        conversation_id: Optional[str] = None,
        verbose: bool = False,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        conversation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'openhands')
        kwargs.setdefault('description', 'OpenHands software agent')
        super().__init__(**kwargs)

        self.model = model or os.getenv('LLM_MODEL', 'anthropic/claude-sonnet-4-5-20250929')
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.base_url = base_url or os.getenv('LLM_BASE_URL')
        self.tools = _DEFAULT_TOOLS if tools is None else tools
        self.mcp_config = mcp_config
        self.usage_id = usage_id or self.name
        self.max_iterations = max_iterations
        self.stuck_detection = stuck_detection
        self.persistence_dir = persistence_dir
        self.conversation_id = conversation_id
        self.verbose = verbose
        self.llm_kwargs = llm_kwargs or {}
        self.agent_kwargs = agent_kwargs or {}
        self.conversation_kwargs = conversation_kwargs or {}

        self._llm = None
        self._conversation = None
        self._events: List[Any] = []  # raw openhands Event objects

    def setup(self) -> None:
        try:
            import openhands.sdk  # noqa: F401
        except ImportError:
            raise RuntimeError("openhands-sdk is required. Install with: pip install -U openhands-sdk openhands-tools")

    def _resolve_tools(self) -> list:
        """Turn the ``tools`` spec into a list of openhands ``Tool`` objects."""
        from importlib import import_module

        from openhands.sdk import Tool

        resolved = []
        for entry in self.tools:
            if isinstance(entry, Tool):
                resolved.append(entry)
                continue
            if not isinstance(entry, str):
                raise TypeError(f"Unsupported tool spec: {entry!r} ({type(entry).__name__})")

            spec = _BUILTIN_TOOLS.get(entry)
            if spec is not None:
                module, attr = spec
                tool_cls = getattr(import_module(module), attr)
                resolved.append(Tool(name=tool_cls.name))
            else:
                # Treat the string as an already-registered tool name.
                resolved.append(Tool(name=entry))
        return resolved

    def _build_llm(self):
        from openhands.sdk import LLM

        llm_args = dict(self.llm_kwargs)
        llm_args.setdefault('usage_id', self.usage_id)
        llm_args['model'] = self.model

        # Proxy takes precedence: route every LLM call through the recorder.
        if self.proxy:
            llm_args['base_url'] = self.proxy.url
            llm_args['api_key'] = f"sk-proxy-{self.session_id}"
        else:
            if self.api_key:
                llm_args['api_key'] = self.api_key
            if self.base_url:
                llm_args['base_url'] = self.base_url

        return LLM(**llm_args)

    def _build_conversation(self):
        from openhands.sdk import Agent, Conversation, Event

        self._llm = self._build_llm()

        agent_args = dict(self.agent_kwargs)
        if self.mcp_config is not None:
            agent_args.setdefault('mcp_config', self.mcp_config)
        agent = Agent(llm=self._llm, tools=self._resolve_tools(), **agent_args)

        def _callback(event: 'Event') -> None:
            self._events.append(event)

        conv_args = dict(self.conversation_kwargs)
        conv_args.setdefault('workspace', self.working_dir or os.getcwd())
        conv_args.setdefault('callbacks', [_callback])
        conv_args.setdefault('max_iteration_per_run', self.max_iterations)
        conv_args.setdefault('stuck_detection', self.stuck_detection)
        if not self.verbose:
            conv_args.setdefault('visualizer', None)
        if self.persistence_dir is not None:
            conv_args.setdefault('persistence_dir', self.persistence_dir)
        if self.conversation_id is not None:
            # Conversation expects a uuid.UUID; coerce a plain string id.
            cid = self.conversation_id
            if isinstance(cid, str):
                from uuid import UUID

                cid = UUID(cid)
            conv_args.setdefault('conversation_id', cid)

        return Conversation(agent=agent, **conv_args)

    async def run_external_async(self, task: str, **kwargs) -> str:
        from openhands.sdk.conversation import get_agent_final_response

        # Lazily build (and reuse for multi-turn) the conversation.
        if self._conversation is None:
            self._conversation = self._build_conversation()

        conversation = self._conversation
        start_idx = len(self._events)

        def _run() -> None:
            conversation.send_message(task)
            conversation.run()

        # run() blocks; offload to a thread so the event loop (and proxy) stay live.
        loop = asyncio.get_running_loop()
        if self.timeout:
            await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=self.timeout)
        else:
            await loop.run_in_executor(None, _run)

        # Final assistant text for *this* turn (events appended since start_idx).
        turn_events = self._events[start_idx:]
        result = get_agent_final_response(turn_events)
        if result:
            return result
        # Fallback: scan the whole history, then give up gracefully.
        result = get_agent_final_response(self._events)
        return result or '(no output)'

    def reset_session(self) -> None:
        """Forget the current conversation; the next call starts fresh."""
        self._conversation = None
        self._events = []

    def get_messages(self) -> List[dict]:
        """Return the LLM trace recorded by the proxy (empty if no proxy)."""
        return self.proxy.get_messages() if self.proxy else []
