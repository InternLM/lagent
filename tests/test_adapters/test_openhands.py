"""Tests for the OpenHands adapter.

``openhands-sdk`` is an optional dependency that may not be installed, so these
tests inject a fake ``openhands`` package tree into ``sys.modules`` and exercise
the adapter's wiring (tool resolution, proxy injection, multi-turn reuse,
trace/metrics capture) against it.
"""

import sys
import types

import pytest

from lagent.adapters.openhands import OpenHandsAdapter
from lagent.schema import AgentMessage

# ── Fake openhands SDK ──────────────────────────────────────────


class _FakeEvent:
    """Base fake event."""


class _FakeMessageEvent(_FakeEvent):
    """A message-bearing event; the adapter reads .source/.text off it."""

    def __init__(self, source, role, text):
        self.source = source
        self.role = role
        self.text = text


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAgent:
    def __init__(self, llm=None, tools=None, **kwargs):
        self.llm = llm
        self.tools = tools
        self.kwargs = kwargs


class _FakeTool:
    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _FakeTool) and other.name == self.name


class _FakeConversation:
    """Drives the registered callbacks with one user + one agent event per run."""

    instances = []

    def __init__(self, agent=None, callbacks=None, **kwargs):
        self.agent = agent
        self.callbacks = callbacks or []
        self.kwargs = kwargs
        self.id = 'conv-123'
        self._pending = None
        self.run_count = 0
        _FakeConversation.instances.append(self)

    def send_message(self, message):
        self._pending = message

    def run(self):
        self.run_count += 1
        # Echo the user message, then an agent reply referencing it.
        for cb in self.callbacks:
            cb(_FakeMessageEvent('user', 'user', self._pending))
            cb(_FakeMessageEvent('agent', 'assistant', f"done: {self._pending}"))


def _fake_get_agent_final_response(events):
    for event in reversed(list(events)):
        if getattr(event, 'source', None) == 'agent':
            return event.text
    return ''


@pytest.fixture
def fake_openhands(monkeypatch):
    """Install a fake ``openhands`` package tree into sys.modules."""
    _FakeConversation.instances = []

    sdk = types.ModuleType('openhands.sdk')
    sdk.LLM = _FakeLLM
    sdk.Agent = _FakeAgent
    sdk.Conversation = _FakeConversation
    sdk.Tool = _FakeTool
    sdk.Event = _FakeEvent

    conv_mod = types.ModuleType('openhands.sdk.conversation')
    conv_mod.get_agent_final_response = _fake_get_agent_final_response
    sdk.conversation = conv_mod

    openhands = types.ModuleType('openhands')
    openhands.sdk = sdk
    tools_pkg = types.ModuleType('openhands.tools')
    openhands.tools = tools_pkg

    def _tool_module(name, cls_name):
        mod = types.ModuleType(f'openhands.tools.{name}')
        cls = type(cls_name, (), {'name': name})
        setattr(mod, cls_name, cls)
        setattr(tools_pkg, name, mod)
        return mod

    modules = {
        'openhands': openhands,
        'openhands.sdk': sdk,
        'openhands.sdk.conversation': conv_mod,
        'openhands.tools': tools_pkg,
        'openhands.tools.terminal': _tool_module('terminal', 'TerminalTool'),
        'openhands.tools.file_editor': _tool_module('file_editor', 'FileEditorTool'),
        'openhands.tools.task_tracker': _tool_module('task_tracker', 'TaskTrackerTool'),
        'openhands.tools.browser_use': _tool_module('browser_use', 'BrowserToolSet'),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield


# ── Setup / init ────────────────────────────────────────────────


class TestSetup:

    def test_setup_raises_without_sdk(self, monkeypatch):
        # Ensure openhands is not importable.
        for name in list(sys.modules):
            if name == 'openhands' or name.startswith('openhands.'):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr('builtins.__import__', _block_openhands(__import__))
        adapter = OpenHandsAdapter(name='oh')
        with pytest.raises(RuntimeError, match='openhands-sdk is required'):
            adapter.setup()

    def test_init_defaults(self, monkeypatch):
        monkeypatch.delenv('LLM_MODEL', raising=False)
        monkeypatch.delenv('LLM_API_KEY', raising=False)
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        adapter = OpenHandsAdapter()
        assert adapter.name == 'openhands'
        assert adapter.model == 'anthropic/claude-sonnet-4-5-20250929'
        assert adapter.tools == ['terminal', 'file_editor', 'task_tracker']
        assert adapter.usage_id == 'openhands'

    def test_init_reads_env(self, monkeypatch):
        monkeypatch.setenv('LLM_MODEL', 'openai/gpt-4o')
        monkeypatch.setenv('LLM_API_KEY', 'env-key')
        monkeypatch.setenv('LLM_BASE_URL', 'http://env.example')
        adapter = OpenHandsAdapter()
        assert adapter.model == 'openai/gpt-4o'
        assert adapter.api_key == 'env-key'
        assert adapter.base_url == 'http://env.example'


def _block_openhands(real_import):
    def _imp(name, *args, **kwargs):
        if name == 'openhands' or name.startswith('openhands.'):
            raise ImportError('No module named openhands')
        return real_import(name, *args, **kwargs)

    return _imp


# ── Tool resolution ─────────────────────────────────────────────


class TestResolveTools:

    def test_default_tools(self, fake_openhands):
        adapter = OpenHandsAdapter()
        adapter.setup()
        resolved = adapter._resolve_tools()
        names = [t.name for t in resolved]
        assert names == ['terminal', 'file_editor', 'task_tracker']

    def test_friendly_aliases(self, fake_openhands):
        adapter = OpenHandsAdapter(tools=['bash', 'browser'])
        adapter.setup()
        names = [t.name for t in adapter._resolve_tools()]
        assert names == ['terminal', 'browser_use']

    def test_raw_tool_name_passthrough(self, fake_openhands):
        adapter = OpenHandsAdapter(tools=['some_custom_tool'])
        adapter.setup()
        names = [t.name for t in adapter._resolve_tools()]
        assert names == ['some_custom_tool']

    def test_tool_object_passthrough(self, fake_openhands):
        from openhands.sdk import Tool

        tool = Tool(name='prebuilt')
        adapter = OpenHandsAdapter(tools=[tool])
        adapter.setup()
        assert adapter._resolve_tools() == [tool]

    def test_empty_tools(self, fake_openhands):
        adapter = OpenHandsAdapter(tools=[])
        adapter.setup()
        assert adapter._resolve_tools() == []

    def test_invalid_tool_spec(self, fake_openhands):
        adapter = OpenHandsAdapter(tools=[123])
        adapter.setup()
        with pytest.raises(TypeError):
            adapter._resolve_tools()


# ── LLM build / proxy injection ─────────────────────────────────


class TestBuildLLM:

    def test_no_proxy_uses_credentials(self, fake_openhands):
        adapter = OpenHandsAdapter(model='openai/gpt-4o', api_key='k', base_url='http://x')
        adapter.setup()
        llm = adapter._build_llm()
        assert llm.kwargs['model'] == 'openai/gpt-4o'
        assert llm.kwargs['api_key'] == 'k'
        assert llm.kwargs['base_url'] == 'http://x'
        assert llm.kwargs['usage_id'] == 'openhands'

    def test_proxy_injection(self, fake_openhands):
        mock_proxy = types.SimpleNamespace(url='http://127.0.0.1:9999', session_id='sess123', is_running=True)
        adapter = OpenHandsAdapter(model='openai/gpt-4o', api_key='k', proxy=mock_proxy)
        adapter.setup()
        llm = adapter._build_llm()
        assert llm.kwargs['base_url'] == 'http://127.0.0.1:9999'
        assert llm.kwargs['api_key'] == 'sk-proxy-sess123'

    def test_no_api_key_omitted(self, fake_openhands, monkeypatch):
        monkeypatch.delenv('LLM_API_KEY', raising=False)
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        adapter = OpenHandsAdapter(model='openai/gpt-4o')
        adapter.setup()
        llm = adapter._build_llm()
        assert 'api_key' not in llm.kwargs
        assert 'base_url' not in llm.kwargs


# ── End-to-end forward / multi-turn ─────────────────────────────


class TestForward:

    @pytest.mark.asyncio
    async def test_forward_returns_final_response(self, fake_openhands):
        adapter = OpenHandsAdapter(api_key='k')
        result = await adapter('hello')
        assert isinstance(result, AgentMessage)
        assert result.content == 'done: hello'
        assert result.sender == 'openhands'

    @pytest.mark.asyncio
    async def test_multi_turn_reuses_conversation(self, fake_openhands):
        adapter = OpenHandsAdapter(api_key='k')
        await adapter('first')
        conv = adapter._conversation
        r2 = await adapter('second')
        assert adapter._conversation is conv  # same conversation object
        assert conv.run_count == 2
        assert r2.content == 'done: second'

    @pytest.mark.asyncio
    async def test_conversation_kwargs_passed(self, fake_openhands):
        adapter = OpenHandsAdapter(api_key='k', max_iterations=42, stuck_detection=False)
        await adapter('go')
        conv = adapter._conversation
        assert conv.kwargs['max_iteration_per_run'] == 42
        assert conv.kwargs['stuck_detection'] is False
        # visualizer disabled by default (verbose=False)
        assert conv.kwargs['visualizer'] is None

    @pytest.mark.asyncio
    async def test_reset_session(self, fake_openhands):
        adapter = OpenHandsAdapter(api_key='k')
        await adapter('first')
        conv = adapter._conversation
        adapter.reset_session()
        assert adapter._conversation is None
        assert adapter._events == []
        await adapter('second')
        assert adapter._conversation is not conv


# ── Trace / metrics ─────────────────────────────────────────────


class TestTrace:

    @pytest.mark.asyncio
    async def test_get_messages_without_proxy_is_empty(self, fake_openhands):
        # No proxy attached → trajectory capture is unavailable; return [].
        adapter = OpenHandsAdapter(api_key='k')
        await adapter('hello')
        assert adapter.get_messages() == []

    @pytest.mark.asyncio
    async def test_get_messages_with_proxy(self, fake_openhands):
        mock_proxy = types.SimpleNamespace(
            url='http://x',
            session_id='s',
            is_running=True,
            get_messages=lambda: [{'messages': ['proxied']}],
        )
        adapter = OpenHandsAdapter(api_key='k', proxy=mock_proxy)
        assert adapter.get_messages() == [{'messages': ['proxied']}]
