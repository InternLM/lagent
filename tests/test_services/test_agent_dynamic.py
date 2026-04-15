"""Unit tests for AgentService + SubAgentAction (spec-based + build kwargs).

Tests the new design where:
- AgentService only manages lifecycle (no default_llm)
- LLM/actions are passed via spec.acreate(**kwargs) → build(config, **kwargs)
- SubAgentAction holds default_llm + parent_actions
"""

import asyncio
import os
import sys
import types

import pytest

# --- bypass circular import ---
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)
for pkg_name, subdir in [
    ("lagent.services", "lagent/services"),
    ("lagent.agents", "lagent/agents"),
]:
    if pkg_name not in sys.modules:
        _pkg = types.ModuleType(pkg_name)
        _pkg.__path__ = [os.path.join(_here, *subdir.split("/"))]
        _pkg.__package__ = pkg_name
        sys.modules[pkg_name] = _pkg

from lagent.agents.agent import AsyncAgent
from lagent.services.agent import AgentService, AgentStatus
from lagent.services.agent_loader import AgentSpec


# ── Mock LLM ─────────────────────────────────────────────────────────

class MockLLM:
    def __init__(self, response: str = "mock response"):
        self._response = response
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        return {"content": self._response}


# ── Helper: register a "default" spec with build function ────────────

def _register_default_spec(svc: AgentService, default_llm: MockLLM = None):
    """Register a 'default' spec whose build accepts llm/actions kwargs."""

    def build(config, llm=None, actions=None, system_prompt=None, **kw):
        template = system_prompt or config.get("template", "")
        return AsyncAgent(llm=llm, template=template)

    spec = AgentSpec(
        name="default",
        agent_config={"type": "AsyncAgent", "template": "default agent"},
        build=build,
    )
    svc.register_spec(spec)


# ═══════════════════════════════════════════════════════════════════════
#  SPEC-BASED SPAWN
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestSpawnFromSpec:
    async def test_spawn_with_spec(self):
        llm = MockLLM(response="spec result")
        svc = AgentService()
        _register_default_spec(svc, llm)

        entry = await svc.spawn(
            task="do something",
            agent_type="default",
            llm=llm,
            mode="sync",
        )

        assert entry.status == AgentStatus.STOPPED
        assert entry.agent_type == "default"
        assert entry.result is not None

    async def test_spawn_unknown_type_raises(self):
        svc = AgentService()
        with pytest.raises(ValueError, match="Unknown agent type"):
            await svc.spawn(task="test", agent_type="nonexistent")

    async def test_kwargs_forwarded_to_build(self):
        """Build function receives llm and actions via **spec_kwargs."""
        received = {}

        def build(config, **kwargs):
            received.update(kwargs)
            return AsyncAgent(llm=kwargs.get("llm"), template="test")

        spec = AgentSpec(name="spy", agent_config={}, build=build)
        svc = AgentService()
        svc.register_spec(spec)

        llm = MockLLM()
        await svc.spawn(
            task="test", agent_type="spy",
            llm=llm, actions=["fake"], custom_param="hello",
            mode="sync",
        )

        assert received["llm"] is llm
        assert received["actions"] == ["fake"]
        assert received["custom_param"] == "hello"

    async def test_shared_llm_instance(self):
        """Multiple spawns share the same LLM reference."""
        llm = MockLLM()
        svc = AgentService()
        _register_default_spec(svc)

        await svc.spawn(task="t1", agent_type="default", llm=llm, mode="sync")
        await svc.spawn(task="t2", agent_type="default", llm=llm, mode="sync")

        assert llm.call_count == 2

    async def test_system_prompt_override(self):
        """system_prompt kwarg should override config template."""
        received_template = {}

        def build(config, llm=None, system_prompt=None, **kw):
            t = system_prompt or config.get("template", "")
            received_template["value"] = t
            return AsyncAgent(llm=llm, template=t)

        spec = AgentSpec(
            name="default",
            agent_config={"template": "original"},
            build=build,
        )
        svc = AgentService()
        svc.register_spec(spec)

        llm = MockLLM()
        await svc.spawn(
            task="test", agent_type="default",
            llm=llm, system_prompt="overridden prompt",
            mode="sync",
        )

        assert received_template["value"] == "overridden prompt"


# ═══════════════════════════════════════════════════════════════════════
#  SPAWN_AGENT (pre-built instance)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestSpawnAgent:
    async def test_spawn_prebuilt(self):
        llm = MockLLM(response="prebuilt result")
        agent = AsyncAgent(llm=llm, template="custom")
        svc = AgentService()

        entry = await svc.spawn_agent(agent=agent, task="do it", mode="sync")

        assert entry.status == AgentStatus.STOPPED
        assert entry.agent_type == "_custom"
        assert entry.result is not None

    async def test_spawn_prebuilt_custom_type(self):
        llm = MockLLM()
        agent = AsyncAgent(llm=llm, template="test")
        svc = AgentService()

        entry = await svc.spawn_agent(
            agent=agent, task="test", agent_type="my_type",
        )
        assert entry.agent_type == "my_type"

    async def test_spawn_prebuilt_async(self):
        llm = MockLLM()
        agent = AsyncAgent(llm=llm, template="test")
        svc = AgentService()

        entry = await svc.spawn_agent(agent=agent, task="bg", mode="async")
        await asyncio.sleep(0.5)
        assert entry.status == AgentStatus.STOPPED


# ═══════════════════════════════════════════════════════════════════════
#  LIFECYCLE (list, get, stop)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestLifecycle:
    async def test_list_entries(self):
        llm = MockLLM()
        svc = AgentService()
        _register_default_spec(svc)

        await svc.spawn(task="t1", agent_type="default", llm=llm, mode="sync")
        await svc.spawn(task="t2", agent_type="default", llm=llm, mode="sync")
        assert len(svc.list()) == 2

    async def test_get_entry(self):
        llm = MockLLM()
        svc = AgentService()
        _register_default_spec(svc)

        entry = await svc.spawn(task="t1", agent_type="default", llm=llm, mode="sync")
        found = svc.get(entry.id)
        assert found is not None
        assert found.id == entry.id

    async def test_stop_async_agent(self):
        llm = MockLLM()
        svc = AgentService()
        _register_default_spec(svc)

        entry = await svc.spawn(task="t", agent_type="default", llm=llm, mode="async")
        await asyncio.sleep(0.1)
        stopped = await svc.stop(entry.id)
        assert isinstance(stopped, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
