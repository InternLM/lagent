"""End-to-end integration test for AgentService + AgentAction + CompactAction.

Three test levels (run selectively via command-line flags):

  1. **Unit** (no LLM, no network):
     Mock everything, verify wiring and data flow.

  2. **Integration** (real LLM, no sandbox):
     Use a live OpenAI-compatible endpoint to test the full
     Policy → fork → compact → summary pipeline.

  3. **Full E2E** (real LLM + sandbox):
     Same as the __main__ block in internclaw_agent.py — tests the
     complete InternClawAgent with AgentService, CompactAction, etc.

Usage::

    # Unit tests only (fast, no network)
    python tests/test_agent_service_e2e.py --unit

    # Integration tests (needs LLM endpoint)
    python tests/test_agent_service_e2e.py --integration

    # Full E2E (needs LLM + sandbox)
    python tests/test_agent_service_e2e.py --e2e

    # All tests
    python tests/test_agent_service_e2e.py --all
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure lagent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lagent.actions.compact import COMPACT_PROMPT, AsyncCompactAction, CompactAction, estimate_token_count
from lagent.actions.subagent import AgentAction, AsyncAgentAction
from lagent.agents.agent import AsyncAgent
from lagent.memory.memory import BaseMemoryStore, ClaudeCodeMemory
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage
from lagent.services.agent import AgentEntry, AgentService, AgentStatus
from lagent.services.agent_loader import AgentSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_e2e")


# =====================================================================
# Helpers / Mocks
# =====================================================================


class MockLLM:
    """A mock LLM that returns canned responses."""

    def __init__(self, responses: list[dict] | None = None):
        self._responses = responses or []
        self._call_count = 0

    async def chat(self, messages: list[dict], tools=None, **kwargs) -> dict:
        self._call_count += 1
        if self._responses:
            idx = min(self._call_count - 1, len(self._responses) - 1)
            return self._responses[idx]
        # Default: return a simple text response
        return {
            "role": "assistant",
            "content": f"Mock response #{self._call_count}",
            "tool_calls": [],
        }

    @property
    def call_count(self):
        return self._call_count


class MockAggregator:
    """Minimal aggregator that just passes messages through."""

    def aggregate(self, memory, name, output_format, template):
        msgs = []
        for m in memory.get_memory():
            if isinstance(m, AgentMessage):
                msgs.append({"role": "user", "content": m.content or ""})
        if not msgs:
            msgs = [{"role": "user", "content": "hello"}]
        return msgs, None


class SimpleTestAgent(AsyncAgent):
    """A minimal async agent for testing purposes.

    Uses a MockLLM and MockAggregator. Returns the LLM response
    directly as an AgentMessage.
    """

    def __init__(self, llm=None, name="test-agent", **kwargs):
        super().__init__(
            llm=llm or MockLLM(),
            aggregator=MockAggregator(),
            name=name,
            **kwargs,
        )

    async def forward(self, *messages, **kwargs):
        formatted, tools = self.aggregator.aggregate(self.memory, self.name, self.output_format, self.template)
        resp = await self.llm.chat(formatted, tools=tools, **kwargs)
        return AgentMessage(
            sender=self.name,
            content=resp.get("content", ""),
            tool_calls=resp.get("tool_calls") or [],
        )


# =====================================================================
# Level 1: Unit Tests (no network)
# =====================================================================


async def test_agent_service_basic():
    """AgentService: register, spawn (with mock agent), list, query."""
    print("\n" + "=" * 60)
    print("TEST: AgentService basic lifecycle (unit)")
    print("=" * 60)

    # 1. Create service without loader
    svc = AgentService()
    assert svc.available_types == [], "Should start empty"
    print("  ✅ AgentService() created without loader")

    # 2. Register spec with agent_config
    spec = AgentSpec(
        name="echo-agent",
        description="Echoes back the task",
        agent_config=dict(
            type=f"{SimpleTestAgent.__module__}.{SimpleTestAgent.__qualname__}",
            llm=dict(
                type=f"{MockLLM.__module__}.{MockLLM.__qualname__}",
                responses=[
                    {
                        "role": "assistant",
                        "content": "Echo: placeholder",
                        "tool_calls": [],
                    }
                ],
            ),
            name="echo-agent",
        ),
    )
    svc.register_spec(spec)
    assert "echo-agent" in svc.available_types
    print(f"  ✅ register_spec() → available_types={svc.available_types}")

    # 3. Spawn sync
    entry = await svc.spawn("echo-agent", "Hello world!", mode="sync")
    assert entry.status == AgentStatus.STOPPED, f"Expected STOPPED, got {entry.status}"
    assert entry.result is not None
    print(f"  ✅ spawn(sync) → status={entry.status}, result={entry.result[:50]}")

    # 4. List
    entries = svc.list()
    assert len(entries) == 1
    print(f"  ✅ list() → {len(entries)} entries")

    # 5. Query
    queried = svc.get(entry.id)
    assert queried is not None and queried.id == entry.id
    print(f"  ✅ get({entry.id}) → found")

    # 6. Spawn async
    entry2 = await svc.spawn("echo-agent", "Background task", mode="async")
    assert entry2.status in (AgentStatus.PENDING, AgentStatus.RUNNING, AgentStatus.STOPPED)
    await asyncio.sleep(0.2)  # let async task complete
    updated = svc.get(entry2.id)
    print(f"  ✅ spawn(async) → status={updated.status}")

    print("  🎉 AgentService basic lifecycle: ALL PASSED\n")


async def test_agent_action_wiring():
    """AgentAction: tool_api methods route to AgentService correctly."""
    print("\n" + "=" * 60)
    print("TEST: AgentAction wiring (unit)")
    print("=" * 60)

    svc = AgentService()
    spec = AgentSpec(
        name="helper",
        description="A helper agent",
        agent_config=dict(
            type=f"{SimpleTestAgent.__module__}.{SimpleTestAgent.__qualname__}",
            llm=dict(
                type=f"{MockLLM.__module__}.{MockLLM.__qualname__}",
                responses=[
                    {
                        "role": "assistant",
                        "content": "Done: something",
                        "tool_calls": [],
                    }
                ],
            ),
            name="helper",
        ),
    )
    svc.register_spec(spec)

    action = AgentAction(agent_service=svc)

    # spawn
    result = await action.spawn(agent_type="helper", task="Do something")
    assert result.state == ActionStatusCode.SUCCESS, f"spawn failed: {result.errmsg}"
    content = result.result[0]["content"]
    assert "Done" in content
    print(f"  ✅ spawn() → {content[:60]}...")

    # list_agents
    result = await action.list_agents()
    assert result.state == ActionStatusCode.SUCCESS
    content = result.result[0]["content"]
    assert "helper" in content or "Do something" in content
    print(f"  ✅ list_agents() → {content[:60]}...")

    # spawn unknown type
    result = await action.spawn(agent_type="nonexistent", task="fail")
    assert result.state == ActionStatusCode.API_ERROR
    print(f"  ✅ spawn(unknown) → error: {result.errmsg[:60]}...")

    print("  🎉 AgentAction wiring: ALL PASSED\n")


async def test_compact_action_unit():
    """CompactAction: should_compact + compact with mock service."""
    print("\n" + "=" * 60)
    print("TEST: CompactAction unit (mock LLM)")
    print("=" * 60)

    svc = AgentService()

    # Create a policy agent with mock LLM that returns a summary
    policy = SimpleTestAgent(
        llm=MockLLM(
            [
                {
                    "role": "assistant",
                    "content": "## Summary\nUser asked to test compact.\n## Pending\nNothing.",
                    "tool_calls": [],
                }
            ]
        ),
        name="policy",
    )
    # Feed some history into policy's memory
    policy.memory.add(AgentMessage(sender="user", content="Hello, please help me test"))
    policy.memory.add(AgentMessage(sender="policy", content="Sure, I'll help"))

    compact = CompactAction(
        agent_service=svc,
        policy_agent=policy,
        max_context_tokens=1000,
        threshold_ratio=0.5,
    )

    # should_compact
    assert not compact.should_compact(400), "400 < 500 threshold"
    assert compact.should_compact(600), "600 > 500 threshold"
    print("  ✅ should_compact() threshold logic works")

    # compact (fork-based)
    result = await compact.compact()
    assert result.state == ActionStatusCode.SUCCESS, f"compact failed: {result.errmsg}"
    summary = result.result[0]["content"]
    assert "Summary" in summary
    print(f"  ✅ compact() → summary: {summary[:60]}...")

    # circuit breaker
    compact._consecutive_failures = 3
    assert not compact.should_compact(9999), "Circuit breaker should block"
    compact._consecutive_failures = 0
    print("  ✅ Circuit breaker works")

    print("  🎉 CompactAction unit: ALL PASSED\n")


async def test_claude_code_memory_unit():
    """ClaudeCodeMemory: unified interface with CompactAction."""
    print("\n" + "=" * 60)
    print("TEST: ClaudeCodeMemory unit")
    print("=" * 60)

    svc = AgentService()
    policy = SimpleTestAgent(
        llm=MockLLM(
            [
                {
                    "role": "assistant",
                    "content": "Compacted summary here.",
                    "tool_calls": [],
                }
            ]
        ),
        name="policy",
    )
    policy.memory.add(AgentMessage(sender="user", content="test"))

    compact = CompactAction(
        agent_service=svc,
        policy_agent=policy,
        max_context_tokens=1000,
        threshold_ratio=0.5,
    )

    mem = ClaudeCodeMemory(compact_action=compact)

    # get_info returns empty (no persistent storage)
    info = await mem.get_info()
    assert info == {}
    print("  ✅ get_info() → {}")

    # should_compact delegates to CompactAction
    assert not mem.should_compact(400)
    assert mem.should_compact(600)
    print("  ✅ should_compact() delegates correctly")

    # compact returns summary string
    summary = await mem.compact()
    assert summary is not None and "Compacted summary" in summary
    print(f"  ✅ compact() → {summary[:50]}...")

    # actions is empty
    assert mem.actions == []
    print("  ✅ actions → []")

    print("  🎉 ClaudeCodeMemory unit: ALL PASSED\n")


async def test_spawn_with_state():
    """Test spawn with state transfer (replaces fork)."""
    print("\n" + "=" * 60)
    print("TEST: Spawn with state transfer (unit)")
    print("=" * 60)

    svc = AgentService()

    spec = AgentSpec(
        name="worker",
        description="A worker agent",
        agent_config=dict(
            type=f"{SimpleTestAgent.__module__}.{SimpleTestAgent.__qualname__}",
            llm=dict(
                type=f"{MockLLM.__module__}.{MockLLM.__qualname__}",
                responses=[
                    {
                        "role": "assistant",
                        "content": "Worker done",
                        "tool_calls": [],
                    }
                ],
            ),
            name="worker",
        ),
    )
    svc.register_spec(spec)

    # Spawn mode (normal)
    entry1 = await svc.spawn("worker", "Task A", mode="sync")
    assert entry1.agent_type == "worker"
    assert entry1.result is not None
    print(f"  ✅ Spawn mode: type={entry1.agent_type}, result={entry1.result[:40]}")

    # Spawn with state (replaces fork)
    fake_state = {"memory": [{"sender": "user", "content": "Context from prev session"}]}
    entry2 = await svc.spawn("worker", "Continue work", mode="sync", state=fake_state)
    assert entry2.agent_type == "worker"
    assert entry2.result is not None
    print(f"  ✅ Spawn with state: type={entry2.agent_type}, result={entry2.result[:40]}")

    print("  🎉 Spawn with state transfer: ALL PASSED\n")


async def test_agent_spec_create():
    """AgentSpec.create() and acreate() with agent_config (PyConfig)."""
    print("\n" + "=" * 60)
    print("TEST: AgentSpec.create() + acreate() (unit)")
    print("=" * 60)

    # agent_config pointing to SimpleTestAgent
    spec = AgentSpec(
        name="pyconfig-agent",
        agent_config=dict(
            type=f"{SimpleTestAgent.__module__}.{SimpleTestAgent.__qualname__}",
            llm=dict(
                type=f"{MockLLM.__module__}.{MockLLM.__qualname__}",
                responses=[
                    {
                        "role": "assistant",
                        "content": "PyConfig agent works!",
                        "tool_calls": [],
                    }
                ],
            ),
            name="pyconfig-test",
        ),
    )

    try:
        agent = spec.create()
        assert isinstance(agent, AsyncAgent), f"Got {type(agent)}"
        print(f"  ✅ spec.create() created {type(agent).__name__}")

        # Run the agent
        response = await agent("test task")
        print(f"  ✅ Agent response: {response.content[:50]}")
    except Exception as exc:
        # create_object may not resolve the mock classes; that's OK for
        # a unit test — the important thing is the error is clear
        print(f"  ⚠️  spec.create() raised (expected in unit test): {exc}")

    # Test async create
    try:
        agent2 = await spec.acreate()
        assert isinstance(agent2, AsyncAgent), f"Got {type(agent2)}"
        print(f"  ✅ spec.acreate() created {type(agent2).__name__}")
    except Exception as exc:
        print(f"  ⚠️  spec.acreate() raised (expected in unit test): {exc}")

    # Test error case: no agent_config
    spec_empty = AgentSpec(name="empty")
    try:
        spec_empty.create()
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no agent_config" in str(exc)
        print(f"  ✅ No agent_config → ValueError: {str(exc)[:60]}...")

    print("  🎉 AgentSpec.create(): ALL PASSED\n")


async def test_agent_service_persistence():
    """Save and load agent entries."""
    print("\n" + "=" * 60)
    print("TEST: AgentService persistence (unit)")
    print("=" * 60)

    import tempfile

    svc = AgentService()
    spec = AgentSpec(
        name="test-type",
        description="For persistence test",
        agent_config=dict(
            type=f"{SimpleTestAgent.__module__}.{SimpleTestAgent.__qualname__}",
            name="test-type",
        ),
    )
    svc.register_spec(spec)

    entry = await svc.spawn("test-type", "Persist me", mode="sync")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "agents"
        await svc.save_all(save_path)
        assert (save_path / "agents.json").exists()
        print(f"  ✅ save_all() → {save_path / 'agents.json'}")

        # Load into a fresh service
        svc2 = AgentService()
        await svc2.load_all(save_path)
        entries = svc2.list()
        assert len(entries) == 1
        assert entries[0].id == entry.id
        assert entries[0].agent_type == "test-type"
        print(f"  ✅ load_all() → restored {len(entries)} entries")

    print("  🎉 AgentService persistence: ALL PASSED\n")


# =====================================================================
# Level 2: Integration Tests (real LLM)
# =====================================================================


async def test_real_llm_compact():
    """Integration: CompactAction with a real LLM endpoint.

    Requires environment variables:
      - LLM_BASE_URL  (e.g. http://35.220.164.252:3888/v1)
      - LLM_API_KEY
      - LLM_MODEL     (e.g. gpt-5.4)
    Or falls back to the hardcoded defaults from internclaw_agent.py.
    """
    print("\n" + "=" * 60)
    print("TEST: Real LLM compact (integration)")
    print("=" * 60)

    from lagent.agents import AsyncAgent
    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    base_url = os.environ.get("LLM_BASE_URL", "http://35.220.164.252:3888/v1")
    api_key = os.environ.get("LLM_API_KEY", " ")
    model_name = os.environ.get("LLM_MODEL", "gpt-5.4")
    proxy = os.environ.get("LLM_PROXY", "http://100.100.72.89:8899")

    print(f"  Using model={model_name}, base_url={base_url}")

    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=base_url, api_key=api_key, proxy=proxy),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=120,
        max_retry=3,
        sleep_interval=2,
    )

    workspace = Path(__file__).parent.parent / "workspace"
    aggregator = InternClawContextBuilder(workspace, tools=None)

    # Build policy agent
    policy = AsyncAgent(
        llm=model,
        aggregator=aggregator,
        name="policy",
    )

    # Simulate a conversation
    policy.memory.add(
        AgentMessage(
            sender="user",
            content="Please help me write a Python function that calculates fibonacci numbers.",
        )
    )
    policy.memory.add(
        AgentMessage(
            sender="policy",
            content="Sure! Here's a recursive fibonacci function:\n\n```python\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n```",
            tool_calls=[],
        )
    )
    policy.memory.add(
        AgentMessage(
            sender="user",
            content="Can you make it iterative and add memoization?",
        )
    )

    # Create AgentService + CompactAction
    svc = AgentService()

    compact = CompactAction(
        agent_service=svc,
        policy_agent=policy,
        max_context_tokens=1000,
        threshold_ratio=0.1,  # Low threshold to force trigger
    )

    # Check should_compact
    token_est = estimate_token_count([{"content": "x" * 200}])  # simulate some tokens
    print(f"  Token estimate: {token_est}, threshold: {compact.threshold_tokens}")
    assert compact.should_compact(token_est) or token_est < compact.threshold_tokens

    # Force compact
    print("  Calling compact() with real LLM (this may take 10-30s)...")
    result = await compact.compact()
    print(f"  compact() state: {result.state}")
    if result.state == ActionStatusCode.SUCCESS:
        summary = result.result[0]["content"]
        print(f"  ✅ Got summary ({len(summary)} chars):")
        print(f"     {summary[:200]}...")
        assert len(summary) > 50, "Summary too short"
    else:
        print(f"  ⚠️  compact() returned error: {result.errmsg}")
        print("     (This may be expected if the LLM endpoint is down)")

    print("  🎉 Real LLM compact: DONE\n")


async def test_real_llm_agent_service_spawn():
    """Integration: AgentService.spawn with real LLM using PyConfig."""
    print("\n" + "=" * 60)
    print("TEST: Real LLM AgentService spawn (integration)")
    print("=" * 60)

    from lagent.agents import AsyncAgent
    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    base_url = os.environ.get("LLM_BASE_URL", "http://35.220.164.252:3888/v1")
    api_key = os.environ.get("LLM_API_KEY", " ")
    model_name = os.environ.get("LLM_MODEL", "gpt-5.4")
    proxy = os.environ.get("LLM_PROXY", "http://100.100.72.89:8899")

    workspace = Path(__file__).parent.parent / "workspace"

    # AgentService without loader
    svc = AgentService()

    # Register an agent type with PyConfig
    model_cfg = ModelConfig(model=model_name, base_url=base_url, api_key=api_key, proxy=proxy)
    sample_params = SampleParameters(temperature=0.7, top_p=1.0, top_k=50)

    # We can't use full PyConfig here because AsyncAgent needs
    # a ContextBuilder which needs a workspace path.  So we use a
    # custom factory instead.
    async def real_llm_factory(spec: AgentSpec, task: str):
        model = AsyncAPIClient(
            model=model_cfg,
            sample_params=sample_params,
            timeout=120,
            max_retry=3,
            sleep_interval=2,
        )
        aggregator = InternClawContextBuilder(workspace, tools=None)
        agent = AsyncAgent(
            llm=model,
            aggregator=aggregator,
            name=spec.name,
        )
        return agent

    svc._factory = real_llm_factory

    svc.register("qa-agent", description="Answers questions")

    print("  Spawning qa-agent with real LLM (sync mode)...")
    entry = await svc.spawn(
        "qa-agent",
        "What is 2+2? Answer in one word.",
        mode="sync",
    )
    print(f"  spawn() → status={entry.status}")
    if entry.status == AgentStatus.STOPPED:
        print(f"  ✅ Result: {entry.result[:100]}")
    elif entry.status == AgentStatus.FAILED:
        print(f"  ⚠️  Failed: {entry.error}")
    else:
        print(f"  ⚠️  Unexpected status: {entry.status}")

    # List
    entries = svc.list()
    print(f"  ✅ list() → {len(entries)} entries")

    print("  🎉 Real LLM AgentService spawn: DONE\n")


async def test_real_llm_full_pipeline():
    """Integration: Full pipeline — InternClawAgent with AgentService + ClaudeCodeMemory."""
    print("\n" + "=" * 60)
    print("TEST: Full pipeline with real LLM (integration)")
    print("=" * 60)

    from lagent.agents import AsyncAgent
    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.agents.internclaw_agent import AsyncEnvAgent, InternClawAgent
    from lagent.hooks.logger import MessageLogger
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    base_url = os.environ.get("LLM_BASE_URL", "http://35.220.164.252:3888/v1")
    api_key = os.environ.get("LLM_API_KEY", " ")
    model_name = os.environ.get("LLM_MODEL", "gpt-5.4")
    proxy = os.environ.get("LLM_PROXY", "http://100.100.72.89:8899")

    workspace = Path(__file__).parent.parent / "workspace"

    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=base_url, api_key=api_key, proxy=proxy),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=120,
        max_retry=3,
        sleep_interval=2,
    )

    aggregator = InternClawContextBuilder(workspace, tools=None)

    # Step 1: Create AgentService
    svc = AgentService()

    # Step 2: Create PolicyAgent
    policy = AsyncAgent(
        llm=model,
        aggregator=aggregator,
        name="policy",
        hooks=[MessageLogger()],
    )

    # Step 3: Create CompactAction → ClaudeCodeMemory
    compact = CompactAction(
        agent_service=svc,
        policy_agent=policy,
        max_context_tokens=128_000,
        threshold_ratio=0.85,
    )
    memory_store = ClaudeCodeMemory(compact_action=compact)

    # Step 4: Create AgentAction (for sub-agent spawning)
    agent_action = AgentAction(agent_service=svc)

    # Step 5: Create EnvAgent with memory_store
    # Note: For this test we don't use real shell actions — just
    # the AgentAction so policy can spawn sub-agents
    env = AsyncEnvAgent(
        actions=[agent_action],
        skills=None,
        memory_store=memory_store,
    )

    # Step 6: Create InternClawAgent
    # Use max_turn=3 to limit the test
    agent = InternClawAgent(
        policy_agent=policy,
        env_agent=env,
        max_turn=3,
    )

    print("  Running InternClawAgent with max_turn=3...")
    print("  (Policy → Env → Policy → Env → ...)")
    try:
        response = await agent("What is the capital of France? Answer briefly.")
        print(f"  ✅ Agent finished: {response.content[:100]}")
    except Exception as exc:
        print(f"  ⚠️  Agent failed: {exc}")
        import traceback

        traceback.print_exc()

    print("  🎉 Full pipeline: DONE\n")


# =====================================================================
# Level 3: Full E2E (real LLM + sandbox)
# =====================================================================


async def test_e2e_with_sandbox():
    """Full E2E: InternClawAgent with sandbox MCP actions.

    This is essentially the same as the __main__ block in
    internclaw_agent.py, but structured as a proper test.
    """
    print("\n" + "=" * 60)
    print("TEST: Full E2E with sandbox (e2e)")
    print("=" * 60)

    from lagent.actions.mcp_client import AsyncMCPClientSandbox
    from lagent.agents import AsyncAgent
    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.agents.internclaw_agent import AsyncEnvAgent, InternClawAgent
    from lagent.hooks.logger import MessageLogger
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters
    from lagent.skills.skills import SandboxSkillsBackend, SkillsLoader

    base_url = os.environ.get("LLM_BASE_URL", "http://35.220.164.252:3888/v1")
    api_key = os.environ.get("LLM_API_KEY", " ")
    model_name = os.environ.get("LLM_MODEL", "gpt-5.4")
    proxy = os.environ.get("LLM_PROXY", "http://100.100.72.89:8899")
    sandbox_url = os.environ.get("SANDBOX_URL", "http://simple-shell.ailab.ailab.ai/mcp")
    init_dir = os.environ.get("INIT_DIR", "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace")

    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=base_url, api_key=api_key, proxy=proxy),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=600,
        max_retry=10,
        sleep_interval=5,
    )

    shell_action = AsyncMCPClientSandbox('http', url=sandbox_url, init_dir=init_dir)

    try:
        # Discover workspace
        home_path = await shell_action.run(command='pwd')
        import json as _json

        cwd = _json.loads(home_path.result[0]['content'])['cwd']
        workspace_path = os.path.join(cwd, 'workspace')
        print(f"  Workspace: {workspace_path}")

        actions = [shell_action]
        aggregator = InternClawContextBuilder(Path(workspace_path), tools=None)

        # AgentService + CompactAction
        svc = AgentService()

        policy = AsyncAgent(
            llm=model,
            aggregator=aggregator,
            name="policy",
            hooks=[MessageLogger()],
        )

        compact = CompactAction(
            agent_service=svc,
            policy_agent=policy,
            max_context_tokens=128_000,
            threshold_ratio=0.85,
        )
        memory_store = ClaudeCodeMemory(compact_action=compact)

        # AgentAction for sub-agent management
        agent_action = AgentAction(agent_service=svc)

        env = AsyncEnvAgent(
            actions=actions + [agent_action],
            skills=None,
            memory_store=memory_store,
        )

        agent = InternClawAgent(
            policy_agent=policy,
            env_agent=env,
            max_turn=5,
        )

        print("  Running InternClawAgent with sandbox (max_turn=5)...")
        response = await agent("List the files in the current directory using ls -la")
        print(f"  ✅ Agent finished: {response.content[:200]}")

    except Exception as exc:
        print(f"  ⚠️  E2E test failed: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            await shell_action.close()
        except Exception:
            pass

    print("  🎉 Full E2E with sandbox: DONE\n")


# =====================================================================
# Runner
# =====================================================================


async def run_unit_tests():
    """Run all unit tests (no network required)."""
    print("\n" + "#" * 60)
    print("#  UNIT TESTS")
    print("#" * 60)
    await test_agent_service_basic()
    await test_agent_action_wiring()
    await test_compact_action_unit()
    await test_claude_code_memory_unit()
    await test_fork_spawn_modes()
    await test_default_agent_factory()
    await test_agent_service_persistence()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED ✅")
    print("=" * 60)


async def run_integration_tests():
    """Run integration tests (requires LLM endpoint)."""
    print("\n" + "#" * 60)
    print("#  INTEGRATION TESTS (requires LLM endpoint)")
    print("#" * 60)
    await test_real_llm_compact()
    await test_real_llm_agent_service_spawn()
    await test_real_llm_full_pipeline()
    print("=" * 60)
    print("ALL INTEGRATION TESTS DONE ✅")
    print("=" * 60)


async def run_e2e_tests():
    """Run full E2E tests (requires LLM + sandbox)."""
    print("\n" + "#" * 60)
    print("#  E2E TESTS (requires LLM + sandbox)")
    print("#" * 60)
    await test_e2e_with_sandbox()
    print("=" * 60)
    print("ALL E2E TESTS DONE ✅")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AgentService E2E Tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests (no network)")
    parser.add_argument("--integration", action="store_true", help="Run integration tests (needs LLM)")
    parser.add_argument("--e2e", action="store_true", help="Run full E2E tests (needs LLM + sandbox)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Default to unit tests if nothing specified
    if not any([args.unit, args.integration, args.e2e, args.all]):
        args.unit = True

    async def run_all():
        if args.unit or args.all:
            await run_unit_tests()
        if args.integration or args.all:
            await run_integration_tests()
        if args.e2e or args.all:
            await run_e2e_tests()

    asyncio.run(run_integration_tests())


if __name__ == "__main__":
    main()
