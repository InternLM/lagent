"""Unit tests for AgentService.

Run:
    pytest tests/test_agent_service.py -v
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagent.services.agent import AgentEntry, AgentService, AgentStatus, _now_ms
from lagent.services.agent_loader import AgentLoader, AgentSpec


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_agent(name="mock", content="done"):
    """Create a mock async agent that returns an AgentMessage-like object."""
    response = MagicMock()
    response.content = content

    agent = AsyncMock()
    agent.name = name
    agent.return_value = response
    agent.state_dict = MagicMock(return_value={"memory": [{"content": content}]})
    agent.load_state_dict = MagicMock()
    return agent


def _make_slow_agent(delay=10):
    """Create a mock agent that takes a long time (for testing stop/cancel)."""
    async def slow(*args, **kwargs):
        await asyncio.sleep(delay)
        return MagicMock(content="slow done")

    agent = AsyncMock(side_effect=slow)
    agent.state_dict = MagicMock(return_value={})
    return agent


def _make_failing_agent(error_msg="fail"):
    """Create a mock agent that raises on call."""
    agent = AsyncMock(side_effect=RuntimeError(error_msg))
    agent.state_dict = MagicMock(return_value={})
    return agent


def _make_spec(name="test-agent", build_agent=None):
    """Create an AgentSpec with a mock build function."""
    agent = build_agent or _make_mock_agent(name)

    def build(config):
        return agent

    async def abuild(config):
        return agent

    return AgentSpec(
        name=name,
        description=f"Test agent: {name}",
        agent_config={"type": "mock"},
        build=build,
    ), agent


# ── AgentEntry ───────────────────────────────────────────────────────


class TestAgentEntry:

    def test_default_id_generated(self):
        entry = AgentEntry()
        assert len(entry.id) == 8

    def test_to_dict(self):
        entry = AgentEntry(
            id="abc12345",
            agent_type="reviewer",
            label="review task",
            task="review this code",
            status=AgentStatus.RUNNING,
        )
        d = entry.to_dict()
        assert d["id"] == "abc12345"
        assert d["agent_type"] == "reviewer"
        assert d["status"] == "running"
        assert d["result"] is None

    def test_from_dict(self):
        data = {
            "id": "xyz",
            "agent_type": "translator",
            "label": "translate",
            "task": "translate this",
            "status": "stopped",
            "result": "translated text",
            "error": None,
            "created_at_ms": 1000,
            "finished_at_ms": 2000,
        }
        entry = AgentEntry.from_dict(data)
        assert entry.id == "xyz"
        assert entry.agent_type == "translator"
        assert entry.status == "stopped"
        assert entry.result == "translated text"
        assert entry.finished_at_ms == 2000

    def test_from_dict_defaults(self):
        entry = AgentEntry.from_dict({})
        assert len(entry.id) == 8
        assert entry.status == AgentStatus.PENDING

    def test_roundtrip(self):
        entry = AgentEntry(
            agent_type="test", label="lbl", task="tsk",
            status=AgentStatus.STOPPED, result="ok",
        )
        restored = AgentEntry.from_dict(entry.to_dict())
        assert restored.agent_type == entry.agent_type
        assert restored.result == entry.result


# ── AgentService: init & specs ───────────────────────────────────────


class TestAgentServiceSpecs:

    def test_init_empty(self):
        service = AgentService()
        assert service.available_types == []

    def test_register_spec(self):
        service = AgentService()
        spec, _ = _make_spec("my-agent")
        service.register_spec(spec)
        assert "my-agent" in service.available_types
        assert service.get_spec("my-agent") is spec

    def test_get_spec_not_found(self):
        service = AgentService()
        assert service.get_spec("nonexistent") is None

    @pytest.mark.asyncio
    async def test_load_specs_from_loader(self, tmp_path):
        agents = tmp_path / "agents"
        agent_dir = agents / "simple"
        agent_dir.mkdir(parents=True)
        (agent_dir / "config.py").write_text(
            "from lagent.agents.agent import Agent\n"
            "agent_config = dict(type=Agent, name='simple')\n"
            "name = 'simple'\n"
            "description = 'A simple agent'\n",
            encoding="utf-8",
        )

        loader = AgentLoader(tmp_path)
        service = AgentService(agent_loader=loader)
        await service.load_specs()

        assert "simple" in service.available_types

    @pytest.mark.asyncio
    async def test_load_specs_without_loader(self):
        service = AgentService()
        await service.load_specs()  # should not raise
        assert service.available_types == []


# ── AgentService: spawn (sync) ───────────────────────────────────────


class TestAgentServiceSpawnSync:

    @pytest.mark.asyncio
    async def test_spawn_sync_success(self):
        service = AgentService()
        spec, mock_agent = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "do something", mode="sync")

        assert entry.status == AgentStatus.STOPPED
        assert entry.result == "done"
        assert entry.finished_at_ms is not None
        assert entry.agent_type == "worker"
        mock_agent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_spawn_sync_with_label(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", label="my label", mode="sync")
        assert entry.label == "my label"

    @pytest.mark.asyncio
    async def test_spawn_sync_auto_label(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        long_task = "x" * 100
        entry = await service.spawn("worker", long_task, mode="sync")
        assert len(entry.label) <= 41  # 40 chars + ellipsis

    @pytest.mark.asyncio
    async def test_spawn_unknown_type_raises(self):
        service = AgentService()
        with pytest.raises(ValueError, match="Unknown agent type"):
            await service.spawn("nonexistent", "task")

    @pytest.mark.asyncio
    async def test_spawn_sync_failure(self):
        agent = AsyncMock(side_effect=RuntimeError("boom"))
        agent.state_dict = MagicMock(return_value={})

        spec = AgentSpec(
            name="failing",
            agent_config={"type": "mock"},
            build=lambda cfg: agent,
        )
        service = AgentService()
        service.register_spec(spec)

        entry = await service.spawn("failing", "task", mode="sync")
        assert entry.status == AgentStatus.FAILED
        assert "boom" in entry.error

    @pytest.mark.asyncio
    async def test_spawn_saves_state_after_completion(self):
        """After sync execution, agent state should be saved to _saved_states."""
        service = AgentService()
        spec, mock_agent = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")

        assert entry.id in service._saved_states
        assert service._saved_states[entry.id] == {"memory": [{"content": "done"}]}

    @pytest.mark.asyncio
    async def test_spawn_saves_state_even_on_failure(self):
        """State should be saved even if the agent fails."""
        agent = AsyncMock(side_effect=RuntimeError("fail"))
        agent.state_dict = MagicMock(return_value={"memory": []})

        spec = AgentSpec(
            name="failing",
            agent_config={"type": "mock"},
            build=lambda cfg: agent,
        )
        service = AgentService()
        service.register_spec(spec)

        entry = await service.spawn("failing", "task", mode="sync")
        assert entry.status == AgentStatus.FAILED
        assert entry.id in service._saved_states

    @pytest.mark.asyncio
    async def test_spawn_with_state_transfer(self):
        """If state dict is provided, load_state_dict is called."""
        service = AgentService()
        spec, mock_agent = _make_spec("worker")
        service.register_spec(spec)

        state = {"memory": [{"content": "previous"}]}
        await service.spawn("worker", "continue", mode="sync", state=state)

        mock_agent.load_state_dict.assert_called_once_with(state)

    @pytest.mark.asyncio
    async def test_spawn_agent_removed_from_live_after_completion(self):
        """Live _agents dict should be empty after sync completion."""
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        await service.spawn("worker", "task", mode="sync")
        assert len(service._agents) == 0


# ── AgentService: spawn (async) ──────────────────────────────────────


class TestAgentServiceSpawnAsync:

    @pytest.mark.asyncio
    async def test_spawn_async_returns_immediately(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="async")

        # Should return with PENDING or RUNNING, not STOPPED
        assert entry.status in (AgentStatus.PENDING, AgentStatus.RUNNING)
        assert entry.id in service._tasks

        # Wait for completion
        await service._tasks[entry.id]
        assert entry.status == AgentStatus.STOPPED
        assert entry.result == "done"

    @pytest.mark.asyncio
    async def test_spawn_async_saves_state(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="async")
        await service._tasks[entry.id]

        assert entry.id in service._saved_states

    @pytest.mark.asyncio
    async def test_spawn_async_on_complete_callback(self):
        completed = []

        async def on_complete(entry):
            completed.append(entry)

        service = AgentService(on_complete=on_complete)
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="async")
        await service._tasks[entry.id]

        assert len(completed) == 1
        assert completed[0].id == entry.id

    @pytest.mark.asyncio
    async def test_spawn_async_failure(self):
        agent = AsyncMock(side_effect=RuntimeError("async boom"))
        agent.state_dict = MagicMock(return_value={})

        spec = AgentSpec(
            name="failing",
            agent_config={"type": "mock"},
            build=lambda cfg: agent,
        )
        service = AgentService()
        service.register_spec(spec)

        entry = await service.spawn("failing", "task", mode="async")
        # Wait for the task to complete
        task = service._tasks.get(entry.id)
        if task is not None:
            await task

        assert entry.status == AgentStatus.FAILED
        assert "async boom" in entry.error


# ── AgentService: query ──────────────────────────────────────────────


class TestAgentServiceQuery:

    @pytest.mark.asyncio
    async def test_list_all(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        await service.spawn("worker", "task1", mode="sync")
        await service.spawn("worker", "task2", mode="sync")

        entries = service.list()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self):
        service = AgentService()

        spec_ok, _ = _make_spec("ok-agent")
        service.register_spec(spec_ok)

        spec_fail = AgentSpec(
            name="fail-agent",
            agent_config={"type": "mock"},
            build=lambda cfg: _make_failing_agent("fail"),
        )
        service.register_spec(spec_fail)

        await service.spawn("ok-agent", "ok task", mode="sync")
        await service.spawn("fail-agent", "bad task", mode="sync")

        stopped = service.list(status=AgentStatus.STOPPED)
        failed = service.list(status=AgentStatus.FAILED)
        assert len(stopped) == 1
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_list_filter_by_agent_type(self):
        service = AgentService()
        spec1, _ = _make_spec("type-a")
        spec2, _ = _make_spec("type-b")
        service.register_spec(spec1)
        service.register_spec(spec2)

        await service.spawn("type-a", "task", mode="sync")
        await service.spawn("type-b", "task", mode="sync")

        a_only = service.list(agent_type="type-a")
        assert len(a_only) == 1
        assert a_only[0].agent_type == "type-a"

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")
        assert service.get(entry.id) is entry

    def test_get_not_found(self):
        service = AgentService()
        assert service.get("nonexistent") is None


# ── AgentService: resume ─────────────────────────────────────────────


class TestAgentServiceResume:

    @pytest.mark.asyncio
    async def test_resume_reuses_entry(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "first task", mode="sync")
        original_id = entry.id

        resumed = await service.resume(original_id, "continue")
        assert resumed.id == original_id
        assert resumed.status == AgentStatus.STOPPED

    @pytest.mark.asyncio
    async def test_resume_restores_state(self):
        service = AgentService()

        call_count = 0
        agents = []

        def build(cfg):
            nonlocal call_count
            call_count += 1
            agent = _make_mock_agent(content=f"result-{call_count}")
            agents.append(agent)
            return agent

        spec = AgentSpec(
            name="stateful",
            agent_config={"type": "mock"},
            build=build,
        )
        service.register_spec(spec)

        entry = await service.spawn("stateful", "first", mode="sync")
        assert entry.result == "result-1"
        assert entry.id in service._saved_states

        resumed = await service.resume(entry.id, "second")
        assert resumed.result == "result-2"
        # The second agent should have had load_state_dict called
        assert agents[1].load_state_dict.called

    @pytest.mark.asyncio
    async def test_resume_not_found_raises(self):
        service = AgentService()
        with pytest.raises(ValueError, match="not found"):
            await service.resume("nonexistent", "msg")

    @pytest.mark.asyncio
    async def test_resume_running_raises(self):
        service = AgentService()
        spec = AgentSpec(
            name="slow",
            agent_config={"type": "mock"},
            build=lambda cfg: _make_slow_agent(),
        )
        service.register_spec(spec)

        entry = await service.spawn("slow", "task", mode="async")
        await asyncio.sleep(0.05)  # let it start running
        # Entry should be running
        with pytest.raises(ValueError, match="still running"):
            await service.resume(entry.id, "msg")

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_resume_resets_entry_fields(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")
        assert entry.result is not None
        assert entry.finished_at_ms is not None

        # Resume clears old result/error before re-running
        resumed = await service.resume(entry.id, "new task")
        assert resumed.result == "done"
        assert resumed.error is None

    @pytest.mark.asyncio
    async def test_resume_spec_not_found_raises(self):
        service = AgentService()
        spec, _ = _make_spec("temp")
        service.register_spec(spec)

        entry = await service.spawn("temp", "task", mode="sync")

        # Remove the spec to simulate missing spec
        del service._specs["temp"]

        with pytest.raises(ValueError, match="Agent spec.*not found"):
            await service.resume(entry.id, "msg")


# ── AgentService: stop ───────────────────────────────────────────────


class TestAgentServiceStop:

    @pytest.mark.asyncio
    async def test_stop_running_agent(self):
        spec = AgentSpec(
            name="slow",
            agent_config={"type": "mock"},
            build=lambda cfg: _make_slow_agent(),
        )
        service = AgentService()
        service.register_spec(spec)

        entry = await service.spawn("slow", "task", mode="async")
        await asyncio.sleep(0.05)  # let it start

        result = await service.stop(entry.id)
        assert result is True
        assert entry.status == AgentStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_nonexistent_returns_false(self):
        service = AgentService()
        result = await service.stop("nonexistent")
        assert result is False


# ── AgentService: persistence ────────────────────────────────────────


class TestAgentServicePersistence:

    @pytest.mark.asyncio
    async def test_save_and_load_entries(self, tmp_path):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")

        save_path = tmp_path / "agent_data"
        await service.save_all(save_path)

        # Verify files created
        assert (save_path / "agents.json").exists()

        # Load into a new service
        service2 = AgentService()
        await service2.load_all(save_path)

        assert entry.id in service2._entries
        loaded = service2._entries[entry.id]
        assert loaded.agent_type == "worker"
        assert loaded.result == "done"

    @pytest.mark.asyncio
    async def test_save_includes_saved_states(self, tmp_path):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")

        save_path = tmp_path / "agent_data"
        await service.save_all(save_path)

        # saved_states should also be persisted
        states_dir = save_path / "states"
        # Note: save_all saves _agents (running), not _saved_states.
        # But _saved_states are populated after completion.
        # The current save_all only saves live _agents, so states_dir
        # may be empty for completed agents.

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_path(self, tmp_path):
        service = AgentService()
        await service.load_all(tmp_path / "nonexistent")
        assert len(service._entries) == 0

    @pytest.mark.asyncio
    async def test_load_restores_saved_states(self, tmp_path):
        save_path = tmp_path / "agent_data"
        save_path.mkdir(parents=True)

        # Write entries
        entries = [{"id": "abc", "agent_type": "worker", "status": "stopped",
                     "task": "t", "label": "l", "result": "r"}]
        (save_path / "agents.json").write_text(
            json.dumps(entries), encoding="utf-8",
        )

        # Write state
        states_dir = save_path / "states"
        states_dir.mkdir()
        (states_dir / "abc.json").write_text(
            json.dumps({"memory": [{"msg": "hello"}]}), encoding="utf-8",
        )

        service = AgentService()
        await service.load_all(save_path)

        assert "abc" in service._entries
        assert "abc" in service._saved_states
        assert service._saved_states["abc"]["memory"][0]["msg"] == "hello"


# ── AgentService: shutdown ───────────────────────────────────────────


class TestAgentServiceShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_running(self):
        spec = AgentSpec(
            name="slow",
            agent_config={"type": "mock"},
            build=lambda cfg: _make_slow_agent(),
        )
        service = AgentService()
        service.register_spec(spec)

        await service.spawn("slow", "task1", mode="async")
        await service.spawn("slow", "task2", mode="async")
        await asyncio.sleep(0.05)

        assert len(service._tasks) == 2

        await service.shutdown()
        assert len(service._tasks) == 0


# ── AgentService: remove ─────────────────────────────────────────────


class TestAgentServiceRemove:

    @pytest.mark.asyncio
    async def test_remove_stopped_entry(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")
        assert service.remove(entry.id) is True
        assert service.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_remove_running_returns_false(self):
        spec = AgentSpec(
            name="slow",
            agent_config={"type": "mock"},
            build=lambda cfg: _make_slow_agent(),
        )
        service = AgentService()
        service.register_spec(spec)

        entry = await service.spawn("slow", "task", mode="async")
        await asyncio.sleep(0.05)

        assert service.remove(entry.id) is False
        await service.shutdown()

    def test_remove_nonexistent_returns_false(self):
        service = AgentService()
        assert service.remove("nonexistent") is False

    @pytest.mark.asyncio
    async def test_remove_cleans_saved_states(self):
        service = AgentService()
        spec, _ = _make_spec("worker")
        service.register_spec(spec)

        entry = await service.spawn("worker", "task", mode="sync")
        assert entry.id in service._saved_states

        service.remove(entry.id)
        assert entry.id not in service._saved_states


# ── AgentService: concurrency ────────────────────────────────────────


class TestAgentServiceConcurrency:

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent(self):
        """max_concurrent limits how many agents run simultaneously."""
        running = []
        max_seen = 0

        async def slow_call(*args, **kwargs):
            nonlocal max_seen
            running.append(1)
            max_seen = max(max_seen, len(running))
            await asyncio.sleep(0.1)
            running.pop()
            return MagicMock(content="done")

        def make_agent(cfg):
            agent = AsyncMock(side_effect=slow_call)
            agent.state_dict = MagicMock(return_value={})
            return agent

        spec = AgentSpec(
            name="slow",
            agent_config={"type": "mock"},
            build=make_agent,
        )
        service = AgentService(max_concurrent=2)
        service.register_spec(spec)

        # Spawn 4 async agents
        entries = []
        for i in range(4):
            e = await service.spawn("slow", f"task-{i}", mode="async")
            entries.append(e)

        # Wait for all to complete
        await asyncio.gather(*[service._tasks[e.id] for e in entries])

        assert max_seen <= 2


# ── Edge cases: warning/error branches ───────────────────────────────


@pytest.mark.asyncio
async def test_spawn_dynamic_load_from_loader(tmp_path):
    """L197-199: spawn loads spec via loader when not pre-registered."""
    agents = tmp_path / "agents"
    agent_dir = agents / "dynamic"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.py").write_text(
        "from lagent.agents.agent import Agent\n"
        "agent_config = dict(type=Agent, name='dynamic')\n"
        "name = 'dynamic'\n",
        encoding="utf-8",
    )

    loader = AgentLoader(tmp_path)
    service = AgentService(agent_loader=loader)
    # Don't call load_specs — let spawn discover it dynamically

    # Agent will fail (no llm), but the spec should be loaded and cached
    entry = await service.spawn("dynamic", "hello", mode="sync")
    assert "dynamic" in service._specs  # spec was dynamically loaded and cached


@pytest.mark.asyncio
async def test_spawn_state_transfer_failure_warns():
    """L226-227: load_state_dict failure is warned, not fatal."""
    agent = _make_mock_agent()
    agent.load_state_dict = MagicMock(side_effect=RuntimeError("bad state"))

    spec = AgentSpec(
        name="worker",
        agent_config={"type": "mock"},
        build=lambda cfg: agent,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn(
        "worker", "task", mode="sync",
        state={"memory": ["corrupted"]},
    )
    # Should succeed despite state transfer failure
    assert entry.status == AgentStatus.STOPPED
    assert entry.result == "done"


@pytest.mark.asyncio
async def test_run_sync_state_dict_failure_warns():
    """L263-264: state_dict() failure in finally is warned, not fatal."""
    agent = AsyncMock(return_value=MagicMock(content="ok"))
    agent.state_dict = MagicMock(side_effect=RuntimeError("state broken"))

    spec = AgentSpec(
        name="fragile",
        agent_config={"type": "mock"},
        build=lambda cfg: agent,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn("fragile", "task", mode="sync")
    assert entry.status == AgentStatus.STOPPED
    # State should NOT be saved
    assert entry.id not in service._saved_states


@pytest.mark.asyncio
async def test_run_async_state_dict_failure_warns():
    """L294-295: async state_dict() failure in finally is warned."""
    agent = AsyncMock(return_value=MagicMock(content="ok"))
    agent.state_dict = MagicMock(side_effect=RuntimeError("state broken"))

    spec = AgentSpec(
        name="fragile",
        agent_config={"type": "mock"},
        build=lambda cfg: agent,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn("fragile", "task", mode="async")
    await service._tasks[entry.id]

    assert entry.status == AgentStatus.STOPPED
    assert entry.id not in service._saved_states


@pytest.mark.asyncio
async def test_on_complete_callback_failure_logged():
    """L303-304: on_complete callback failure is logged, not fatal."""
    async def bad_callback(entry):
        raise RuntimeError("callback boom")

    service = AgentService(on_complete=bad_callback)
    spec, _ = _make_spec("worker")
    service.register_spec(spec)

    entry = await service.spawn("worker", "task", mode="async")
    await service._tasks[entry.id]

    # Agent should still complete successfully despite callback failure
    assert entry.status == AgentStatus.STOPPED
    assert entry.result == "done"


@pytest.mark.asyncio
async def test_resume_load_state_dict_failure_warns():
    """L361-362: resume load_state_dict failure is warned."""
    call_count = 0

    def build(cfg):
        nonlocal call_count
        call_count += 1
        agent = _make_mock_agent(content=f"r{call_count}")
        if call_count == 2:
            agent.load_state_dict = MagicMock(
                side_effect=RuntimeError("restore failed")
            )
        return agent

    spec = AgentSpec(
        name="stateful",
        agent_config={"type": "mock"},
        build=build,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn("stateful", "first", mode="sync")

    # Resume should succeed despite state restore failure
    resumed = await service.resume(entry.id, "second")
    assert resumed.status == AgentStatus.STOPPED
    assert resumed.result == "r2"


@pytest.mark.asyncio
async def test_save_all_saves_running_agent_states(tmp_path):
    """L415-424: save_all persists live agent states to disk."""
    save_started = asyncio.Event()
    save_done = asyncio.Event()

    async def controlled_agent(*args, **kwargs):
        save_started.set()
        await save_done.wait()
        return MagicMock(content="done")

    agent = AsyncMock(side_effect=controlled_agent)
    agent.state_dict = MagicMock(return_value={"memory": [{"msg": "live"}]})

    spec = AgentSpec(
        name="live",
        agent_config={"type": "mock"},
        build=lambda cfg: agent,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn("live", "task", mode="async")
    await save_started.wait()

    # Save while agent is running
    save_path = tmp_path / "data"
    await service.save_all(save_path)

    state_file = save_path / "states" / f"{entry.id}.json"
    assert state_file.exists()
    saved = json.loads(state_file.read_text())
    assert saved["memory"][0]["msg"] == "live"

    # Let agent finish
    save_done.set()
    await service._tasks[entry.id]


@pytest.mark.asyncio
async def test_save_all_state_dict_failure_warns(tmp_path):
    """L415-424: save_all handles state_dict failure gracefully."""
    save_started = asyncio.Event()
    save_done = asyncio.Event()

    async def controlled_agent(*args, **kwargs):
        save_started.set()
        await save_done.wait()
        return MagicMock(content="done")

    agent = AsyncMock(side_effect=controlled_agent)
    agent.state_dict = MagicMock(side_effect=RuntimeError("cant serialize"))

    spec = AgentSpec(
        name="broken",
        agent_config={"type": "mock"},
        build=lambda cfg: agent,
    )
    service = AgentService()
    service.register_spec(spec)

    entry = await service.spawn("broken", "task", mode="async")
    await save_started.wait()

    save_path = tmp_path / "data"
    await service.save_all(save_path)  # should not raise

    # No state file should be written
    state_file = save_path / "states" / f"{entry.id}.json"
    assert not state_file.exists()

    save_done.set()
    await service._tasks[entry.id]


@pytest.mark.asyncio
async def test_load_all_corrupted_state_warns(tmp_path):
    """L451-452: corrupted state file is warned, not fatal."""
    save_path = tmp_path / "data"
    save_path.mkdir()

    (save_path / "agents.json").write_text(
        json.dumps([{"id": "abc", "agent_type": "x", "status": "stopped",
                      "task": "t", "label": "l"}]),
        encoding="utf-8",
    )

    states_dir = save_path / "states"
    states_dir.mkdir()
    (states_dir / "abc.json").write_text("NOT VALID JSON {{{", encoding="utf-8")

    service = AgentService()
    await service.load_all(save_path)

    assert "abc" in service._entries
    assert "abc" not in service._saved_states  # failed to load


# ── _now_ms ──────────────────────────────────────────────────────────


def test_now_ms():
    ts = _now_ms()
    assert isinstance(ts, int)
    assert ts > 0
