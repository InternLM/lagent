"""End-to-end integration test: AgentLoader → AgentService → AsyncAgentAction.

No mocks on the three core modules.  Agent projects live under
``tests/data/agents/`` and are loaded via real AgentLoader.

Two agent types:
  - ``simple-agent``: EchoAgent, no LLM, no network — always runs.
  - ``e2e-agent``: InternClawAgent with real LLM — needs network.

Run:
    pytest tests/test_e2e_subagent.py -v -s
"""

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagent.actions.subagent import AsyncAgentAction
from lagent.schema import ActionStatusCode
from lagent.services.agent import AgentService, AgentStatus
from lagent.services.agent_loader import AgentLoader

DATA_DIR = Path(__file__).parent / "data"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def service():
    """Real AgentService backed by tests/data/agents/."""
    loader = AgentLoader(DATA_DIR)
    svc = AgentService(agent_loader=loader)
    await svc.load_specs()
    yield svc
    await svc.shutdown()


@pytest.fixture
def action(service):
    """Real AsyncAgentAction wrapping the service."""
    return AsyncAgentAction(agent_service=service)


# ── Helpers ──────────────────────────────────────────────────────────


def _text(ret):
    if ret.result and len(ret.result) > 0:
        return ret.result[0].get("content", "")
    return ""


# ── Discovery ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loader_discovers_test_agents():
    loader = AgentLoader(DATA_DIR)
    entries = await loader.list()
    names = {e["name"] for e in entries}
    assert "simple-agent" in names
    assert "e2e-agent" in names


@pytest.mark.asyncio
async def test_service_loads_specs(service):
    assert "simple-agent" in service.available_types
    assert "e2e-agent" in service.available_types


# ── simple-agent (no LLM, no network) ───────────────────────────────


@pytest.mark.asyncio
async def test_simple_spawn_sync(service, action):
    ret = await action.spawn(
        agent_type="simple-agent",
        task="hello world",
        label="simple-sync",
        mode="sync",
    )
    assert ret.state != ActionStatusCode.API_ERROR, ret.errmsg
    assert "echo: hello world" in _text(ret)


@pytest.mark.asyncio
async def test_simple_spawn_async_and_query(service, action):
    ret = await action.spawn(
        agent_type="simple-agent",
        task="background task",
        label="simple-async",
        mode="async",
    )
    assert ret.state != ActionStatusCode.API_ERROR, ret.errmsg

    # Wait for completion
    entries = service.list(agent_type="simple-agent")
    for e in entries:
        task = service._tasks.get(e.id)
        if task:
            await task

    # Query result
    entry = [e for e in service.list() if e.label == "simple-async"][0]
    query_ret = await action.query_agent(agent_id=entry.id)
    assert "echo: background task" in _text(query_ret)


@pytest.mark.asyncio
async def test_simple_list_agents(service, action):
    await action.spawn(
        agent_type="simple-agent", task="t1", label="label-a", mode="sync",
    )
    await action.spawn(
        agent_type="simple-agent", task="t2", label="label-b", mode="sync",
    )

    ret = await action.list_agents()
    text = _text(ret)
    assert "label-a" in text
    assert "label-b" in text


@pytest.mark.asyncio
async def test_simple_resume(service, action):
    # First run
    ret = await action.spawn(
        agent_type="simple-agent", task="ping", label="resumable", mode="sync",
    )
    assert "echo: ping" in _text(ret)

    # Find the entry
    entry = [e for e in service.list() if e.label == "resumable"][0]

    # Resume
    resume_ret = await action.resume_agent(agent_id=entry.id, message="pong")
    assert resume_ret.state != ActionStatusCode.API_ERROR, resume_ret.errmsg
    assert "echo: pong" in _text(resume_ret)


@pytest.mark.asyncio
async def test_simple_stop(service, action):
    # Need a slow agent to test stop — simple-agent finishes instantly,
    # so spawn it async and try to stop (may already be done)
    ret = await action.spawn(
        agent_type="simple-agent", task="fast", label="stoppable", mode="async",
    )
    entry = [e for e in service.list() if e.label == "stoppable"][0]

    # Try stop — might already be done
    stop_ret = await action.stop_agent(agent_id=entry.id)
    # Either stopped or already finished — both are valid
    assert stop_ret.state != ActionStatusCode.API_ERROR or "not running" in (stop_ret.errmsg or "")


# ── Error paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_unknown_type(service, action):
    ret = await action.spawn(agent_type="no-such-agent", task="fail")
    assert ret.state == ActionStatusCode.API_ERROR
    assert "Unknown agent type" in ret.errmsg


@pytest.mark.asyncio
async def test_query_nonexistent(service, action):
    ret = await action.query_agent(agent_id="nonexistent")
    assert ret.state == ActionStatusCode.API_ERROR
    assert "not found" in ret.errmsg


@pytest.mark.asyncio
async def test_resume_nonexistent(service, action):
    ret = await action.resume_agent(agent_id="nonexistent", message="hi")
    assert ret.state == ActionStatusCode.API_ERROR
    assert "not found" in ret.errmsg


@pytest.mark.asyncio
async def test_stop_nonexistent(service, action):
    ret = await action.stop_agent(agent_id="nonexistent")
    assert ret.state == ActionStatusCode.API_ERROR


# ── e2e-agent (real LLM) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_spawn_sync_real_llm(service, action):
    """Full e2e with real LLM call."""
    ret = await action.spawn(
        agent_type="e2e-agent",
        task="Reply with exactly: HELLO_E2E",
        label="e2e-real-llm",
        mode="sync",
    )
    print(f"\n[e2e spawn] state={ret.state}, errmsg={ret.errmsg}")
    if ret.result:
        print(f"[e2e spawn] {_text(ret)[:200]}")

    assert ret.state != ActionStatusCode.API_ERROR, f"LLM call failed: {ret.errmsg}"
    assert "HELLO_E2E" in _text(ret)


@pytest.mark.asyncio
async def test_e2e_resume_real_llm(service, action):
    """Spawn then resume with real LLM."""
    ret = await action.spawn(
        agent_type="e2e-agent", task="Say PING", label="e2e-resume", mode="sync",
    )
    if ret.state == ActionStatusCode.API_ERROR:
        pytest.skip(f"LLM not available: {ret.errmsg}")

    entry = [e for e in service.list() if e.label == "e2e-resume"][0]

    resume_ret = await action.resume_agent(agent_id=entry.id, message="Now say PONG")
    print(f"\n[e2e resume] {_text(resume_ret)[:200]}")
    assert resume_ret.state != ActionStatusCode.API_ERROR, resume_ret.errmsg
