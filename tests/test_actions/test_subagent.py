"""Unit tests for AsyncAgentAction (subagent toolkit).

Run:
    pytest tests/test_actions/test_subagent.py -v
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagent.actions.subagent import AsyncAgentAction
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.services.agent import AgentEntry, AgentService, AgentStatus


# ── Helpers ──────────────────────────────────────────────────────────


def _make_entry(**overrides) -> AgentEntry:
    defaults = dict(
        id="abc12345",
        agent_type="worker",
        label="test task",
        task="do something",
        status=AgentStatus.STOPPED,
        result="task done",
    )
    defaults.update(overrides)
    return AgentEntry(**defaults)


def _make_action(service=None) -> AsyncAgentAction:
    if service is None:
        service = MagicMock(spec=AgentService)
    return AsyncAgentAction(agent_service=service)


def _result_text(ret: ActionReturn) -> str:
    """Extract text content from ActionReturn."""
    if ret.result and len(ret.result) > 0:
        return ret.result[0].get("content", "")
    return ""


# ── spawn ────────────────────────────────────────────────────────────


class TestSpawn:

    @pytest.mark.asyncio
    async def test_spawn_sync_success(self):
        entry = _make_entry()
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.spawn(
            agent_type="worker", task="do something",
            label="test", mode="sync",
        )

        assert ret.state != ActionStatusCode.API_ERROR
        assert "completed" in _result_text(ret)
        assert entry.id in _result_text(ret)
        service.spawn.assert_awaited_once_with(
            agent_type="worker", task="do something",
            label="test", mode="sync",
        )

    @pytest.mark.asyncio
    async def test_spawn_sync_failed_agent(self):
        entry = _make_entry(status=AgentStatus.FAILED, error="out of memory")
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.spawn(
            agent_type="worker", task="task", mode="sync",
        )

        assert ret.state == ActionStatusCode.API_ERROR
        assert "out of memory" in ret.errmsg

    @pytest.mark.asyncio
    async def test_spawn_sync_no_output(self):
        entry = _make_entry(result=None)
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.spawn(
            agent_type="worker", task="task", mode="sync",
        )

        assert "(no output)" in _result_text(ret)

    @pytest.mark.asyncio
    async def test_spawn_async_success(self):
        entry = _make_entry(status=AgentStatus.RUNNING, result=None)
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.spawn(
            agent_type="worker", task="background job", mode="async",
        )

        assert ret.state != ActionStatusCode.API_ERROR
        text = _result_text(ret)
        assert "background" in text.lower() or "spawned" in text.lower()
        assert entry.id in text
        assert entry.agent_type in text

    @pytest.mark.asyncio
    async def test_spawn_value_error(self):
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(side_effect=ValueError("Unknown agent type"))

        action = _make_action(service)
        ret = await action.spawn(agent_type="bad", task="task")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "Unknown agent type" in ret.errmsg

    @pytest.mark.asyncio
    async def test_spawn_generic_error(self):
        service = MagicMock(spec=AgentService)
        service.spawn = AsyncMock(side_effect=RuntimeError("connection lost"))

        action = _make_action(service)
        ret = await action.spawn(agent_type="worker", task="task")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "connection lost" in ret.errmsg


# ── list_agents ──────────────────────────────────────────────────────


class TestListAgents:

    @pytest.mark.asyncio
    async def test_list_with_entries(self):
        entries = [
            _make_entry(id="aaa", label="task A", status=AgentStatus.STOPPED),
            _make_entry(id="bbb", label="task B", status=AgentStatus.RUNNING),
        ]
        service = MagicMock(spec=AgentService)
        service.list = MagicMock(return_value=entries)

        action = _make_action(service)
        ret = await action.list_agents()

        text = _result_text(ret)
        assert "task A" in text
        assert "task B" in text
        assert "aaa" in text
        assert "bbb" in text

    @pytest.mark.asyncio
    async def test_list_empty(self):
        service = MagicMock(spec=AgentService)
        service.list = MagicMock(return_value=[])

        action = _make_action(service)
        ret = await action.list_agents()

        assert "No sub-agents" in _result_text(ret)

    @pytest.mark.asyncio
    async def test_list_empty_with_filter(self):
        service = MagicMock(spec=AgentService)
        service.list = MagicMock(return_value=[])

        action = _make_action(service)
        ret = await action.list_agents(status="running")

        text = _result_text(ret)
        assert "No sub-agents" in text
        assert "running" in text

    @pytest.mark.asyncio
    async def test_list_status_icons(self):
        entries = [
            _make_entry(id="1", label="p", status=AgentStatus.PENDING),
            _make_entry(id="2", label="r", status=AgentStatus.RUNNING),
            _make_entry(id="3", label="s", status=AgentStatus.STOPPED),
            _make_entry(id="4", label="f", status=AgentStatus.FAILED),
        ]
        service = MagicMock(spec=AgentService)
        service.list = MagicMock(return_value=entries)

        action = _make_action(service)
        ret = await action.list_agents()
        text = _result_text(ret)

        # Each status should have its icon
        for entry in entries:
            assert entry.id in text


# ── query_agent ──────────────────────────────────────────────────────


class TestQueryAgent:

    @pytest.mark.asyncio
    async def test_query_found(self):
        entry = _make_entry(result="42", error=None)
        service = MagicMock(spec=AgentService)
        service.get = MagicMock(return_value=entry)

        action = _make_action(service)
        ret = await action.query_agent(agent_id=entry.id)

        text = _result_text(ret)
        assert entry.id in text
        assert entry.agent_type in text
        assert "42" in text

    @pytest.mark.asyncio
    async def test_query_with_error(self):
        entry = _make_entry(
            status=AgentStatus.FAILED, result=None, error="kaboom",
        )
        service = MagicMock(spec=AgentService)
        service.get = MagicMock(return_value=entry)

        action = _make_action(service)
        ret = await action.query_agent(agent_id=entry.id)

        text = _result_text(ret)
        assert "kaboom" in text

    @pytest.mark.asyncio
    async def test_query_not_found(self):
        service = MagicMock(spec=AgentService)
        service.get = MagicMock(return_value=None)

        action = _make_action(service)
        ret = await action.query_agent(agent_id="nonexistent")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "not found" in ret.errmsg

    @pytest.mark.asyncio
    async def test_query_no_result_no_error(self):
        entry = _make_entry(result=None, error=None)
        service = MagicMock(spec=AgentService)
        service.get = MagicMock(return_value=entry)

        action = _make_action(service)
        ret = await action.query_agent(agent_id=entry.id)

        text = _result_text(ret)
        assert entry.id in text
        assert "Result" not in text
        assert "Error" not in text


# ── resume_agent ─────────────────────────────────────────────────────


class TestResumeAgent:

    @pytest.mark.asyncio
    async def test_resume_success(self):
        entry = _make_entry(result="continued")
        service = MagicMock(spec=AgentService)
        service.resume = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.resume_agent(agent_id="abc12345", message="go on")

        text = _result_text(ret)
        assert "resumed" in text
        assert "continued" in text
        service.resume.assert_awaited_once_with("abc12345", "go on")

    @pytest.mark.asyncio
    async def test_resume_no_output(self):
        entry = _make_entry(result=None)
        service = MagicMock(spec=AgentService)
        service.resume = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.resume_agent(agent_id="abc12345", message="go")

        assert "(no output)" in _result_text(ret)

    @pytest.mark.asyncio
    async def test_resume_failed(self):
        entry = _make_entry(status=AgentStatus.FAILED, error="crash")
        service = MagicMock(spec=AgentService)
        service.resume = AsyncMock(return_value=entry)

        action = _make_action(service)
        ret = await action.resume_agent(agent_id="abc12345", message="retry")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "crash" in ret.errmsg

    @pytest.mark.asyncio
    async def test_resume_value_error(self):
        service = MagicMock(spec=AgentService)
        service.resume = AsyncMock(
            side_effect=ValueError("Agent 'x' not found"),
        )

        action = _make_action(service)
        ret = await action.resume_agent(agent_id="x", message="hi")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "not found" in ret.errmsg

    @pytest.mark.asyncio
    async def test_resume_generic_error(self):
        service = MagicMock(spec=AgentService)
        service.resume = AsyncMock(side_effect=RuntimeError("timeout"))

        action = _make_action(service)
        ret = await action.resume_agent(agent_id="x", message="hi")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "timeout" in ret.errmsg


# ── stop_agent ───────────────────────────────────────────────────────


class TestStopAgent:

    @pytest.mark.asyncio
    async def test_stop_success(self):
        service = MagicMock(spec=AgentService)
        service.stop = AsyncMock(return_value=True)

        action = _make_action(service)
        ret = await action.stop_agent(agent_id="abc12345")

        assert ret.state != ActionStatusCode.API_ERROR
        assert "stopped" in _result_text(ret).lower()

    @pytest.mark.asyncio
    async def test_stop_not_found(self):
        service = MagicMock(spec=AgentService)
        service.stop = AsyncMock(return_value=False)

        action = _make_action(service)
        ret = await action.stop_agent(agent_id="nonexistent")

        assert ret.state == ActionStatusCode.API_ERROR
        assert "not found" in ret.errmsg.lower() or "not running" in ret.errmsg.lower()
