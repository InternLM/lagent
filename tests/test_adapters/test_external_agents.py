"""Tests for external agent adapters."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lagent.adapters.base import AsyncExternalAgent, BaseExternalAgent
from lagent.adapters.cli_adapter import CLIAgentAdapter
from lagent.adapters.sdk_adapter import SDKAgentAdapter
from lagent.actions.external_agent import ExternalAgentAction
from lagent.schema import AgentMessage


# ── Base adapter tests ──────────────────────────────────────────


class ConcreteExternalAgent(AsyncExternalAgent):
    """Minimal concrete implementation for testing."""

    def setup(self):
        pass

    async def run_external_async(self, task, **kwargs):
        return f"result: {task}"


class TestBaseExternalAgent:

    def test_init_assigns_session_id(self):
        agent = ConcreteExternalAgent(name="test")
        assert agent.session_id
        assert len(agent.session_id) == 8

    def test_init_no_llm(self):
        agent = ConcreteExternalAgent(name="test")
        assert agent.llm is None

    @pytest.mark.asyncio
    async def test_forward_returns_agent_message(self):
        agent = ConcreteExternalAgent(name="test-agent")
        result = await agent("hello world")
        assert isinstance(result, AgentMessage)
        assert result.content == "result: hello world"
        assert result.sender == "test-agent"

    @pytest.mark.asyncio
    async def test_forward_stores_in_memory(self):
        agent = ConcreteExternalAgent(name="test")
        await agent("task1")
        msgs = agent.memory.get_memory()
        assert len(msgs) == 2  # input + output

    def test_state_dict_without_proxy(self):
        agent = ConcreteExternalAgent(name="test")
        state = agent.state_dict()
        assert 'memory' in state
        assert 'llm_trace' not in state

    def test_state_dict_with_proxy(self):
        mock_proxy = MagicMock()
        mock_proxy.get_records.return_value = [{"test": "record"}]
        agent = ConcreteExternalAgent(name="test", proxy=mock_proxy)
        state = agent.state_dict()
        assert 'llm_trace' in state
        assert state['llm_trace'] == [{"test": "record"}]

    def test_build_env_without_proxy(self):
        agent = ConcreteExternalAgent(
            name="test",
            env_vars={"MY_VAR": "123"},
        )
        env = agent._build_env()
        assert env["MY_VAR"] == "123"
        assert "OPENAI_BASE_URL" not in env or "sk-proxy" not in env.get("OPENAI_API_KEY", "")

    def test_build_env_with_proxy(self):
        mock_proxy = MagicMock()
        mock_proxy.url = "http://127.0.0.1:9999"
        agent = ConcreteExternalAgent(name="test", proxy=mock_proxy)
        env = agent._build_env()
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999"
        assert env["OPENAI_API_KEY"].startswith("sk-proxy-")
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"


# ── CLI adapter tests ───────────────────────────────────────────


class TestCLIAgentAdapter:

    def test_setup_finds_echo(self):
        adapter = CLIAgentAdapter(
            name="test-echo",
            command_template="echo '{task}'",
        )
        adapter.setup()  # should not raise

    def test_setup_raises_for_missing_binary(self):
        adapter = CLIAgentAdapter(
            name="test-missing",
            command_template="nonexistent_binary_xyz '{task}'",
        )
        with pytest.raises(RuntimeError, match="not found on PATH"):
            adapter.setup()

    @pytest.mark.asyncio
    async def test_run_echo(self):
        adapter = CLIAgentAdapter(
            name="echo-agent",
            command_template="echo '{task}'",
            timeout=5,
        )
        result = await adapter("hello")
        assert isinstance(result, AgentMessage)
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self):
        adapter = CLIAgentAdapter(
            name="slow-agent",
            command_template="sleep 10 && echo '{task}'",
            timeout=0.5,
        )
        result = await adapter("test")
        assert "failed" in result.content.lower() or "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_custom_parse_output(self):
        def my_parser(stdout, stderr):
            return f"PARSED: {stdout.strip()}"

        adapter = CLIAgentAdapter(
            name="parse-test",
            command_template="echo '{task}'",
            parse_output=my_parser,
        )
        result = await adapter("data")
        assert "PARSED:" in result.content


# ── SDK adapter tests ───────────────────────────────────────────


class MockSDKAdapter(SDKAgentAdapter):
    """Mock SDK adapter for testing."""

    def create_sdk_agent(self, config):
        return {"model": config.get("model", "test")}

    def invoke_sdk_agent(self, agent, task, **kwargs):
        return f"SDK({agent['model']}): {task}"


class TestSDKAgentAdapter:

    @pytest.mark.asyncio
    async def test_basic_invoke(self):
        adapter = MockSDKAdapter(
            name="mock-sdk",
            sdk_config={"model": "gpt-4"},
        )
        result = await adapter("test task")
        assert isinstance(result, AgentMessage)
        assert "SDK(gpt-4): test task" in result.content

    def test_setup_with_invalid_module(self):
        adapter = MockSDKAdapter(
            name="bad-sdk",
            sdk_module="nonexistent_module_xyz",
        )
        with pytest.raises(RuntimeError, match="not importable"):
            adapter.setup()


# ── ExternalAgentAction tests ───────────────────────────────────


class TestExternalAgentAction:

    @pytest.mark.asyncio
    async def test_run_agent_success(self):
        adapter = ConcreteExternalAgent(name="test-ext")
        action = ExternalAgentAction(adapters={"test-ext": adapter})
        result = await action('{"agent_name": "test-ext", "task": "do stuff"}', 'run_agent')
        assert result.state == 0  # SUCCESS

    @pytest.mark.asyncio
    async def test_run_agent_unknown(self):
        action = ExternalAgentAction(adapters={})
        result = await action('{"agent_name": "nope", "task": "x"}', 'run_agent')
        assert result.errmsg
        assert "Unknown agent" in result.errmsg

    @pytest.mark.asyncio
    async def test_list_agents(self):
        adapter = ConcreteExternalAgent(name="a1", description="Agent One")
        action = ExternalAgentAction(adapters={"a1": adapter})
        result = await action('{}', 'list_agents')
        assert result.result
        assert "a1" in result.result[0]["content"]

    @pytest.mark.asyncio
    async def test_list_agents_empty(self):
        action = ExternalAgentAction(adapters={})
        result = await action('{}', 'list_agents')
        assert "No external agents" in result.result[0]["content"]
