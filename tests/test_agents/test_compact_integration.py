"""Integration tests for CompactAgent + Provider + InternClawAgent.

Tests the compact data flow end-to-end:
  1. Provider + CompactAgent interaction
  2. Full InternClawAgent loop with compact triggering
  3. ContextBuilder handling compact_boundary + summary

Usage:
    python tests/test_agents/test_compact_integration.py          # mock only
    python tests/test_agents/test_compact_integration.py --real    # real model
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from lagent.agents.compact_agent import AsyncCompactAgent, estimate_token_count
from lagent.agents.internclaw_agent import (
    AsyncEnvAgent, AsyncPolicyAgent, InternClawAgent,
)
from lagent.agents.aggregator.context import InternClawContextBuilder
from lagent.memory import Memory, OpenClawMemoryProvider
from lagent.actions.save_memory import SaveMemoryAction
from lagent.schema import AgentMessage


# ── Helpers ───────────────────────────────────────────────────────

class MockLLM:
    """Mock LLM that returns predictable responses."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    async def chat(self, messages, **kwargs):
        self._call_count += 1
        if self._responses:
            resp = self._responses.pop(0)
            if callable(resp):
                return resp(messages, kwargs)
            return resp
        # Default: echo back with tool call to keep loop going
        return {"content": f"Response #{self._call_count}"}


# ── Test 1: Provider + EnvAgent get_info injection ────────────────

async def test_provider_injects_into_env_info():
    """Verify provider.get_info() content appears in env_info['memory']."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        mem_dir = workspace / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("# Facts\n- User prefers Python\n- Project uses lagent")

        provider = OpenClawMemoryProvider(workspace)
        env = AsyncEnvAgent(actions=[], long_term_memory=provider)

        info = await env.get_env_info()
        assert info['memory']['available'] is True
        assert 'Python' in info['memory']['long_term']
        assert 'lagent' in info['memory']['long_term']


# ── Test 2: CompactAgent with formatted_messages ──────────────────

async def test_compact_formats_and_summarizes():
    """Verify CompactAgent correctly formats list[dict] input and produces summary."""
    received = {}

    class CaptureLLM:
        async def chat(self, messages, **kwargs):
            received['messages'] = messages
            return {"content": "## Summary\nUser is refactoring lagent memory system."}

    compact = AsyncCompactAgent(llm=CaptureLLM(), max_context_tokens=100, threshold_ratio=0.5)

    formatted_messages = [
        {'role': 'system', 'content': 'You are helpful.'},
        {'role': 'user', 'content': 'Refactor memory'},
        {'role': 'assistant', 'content': 'Sure, analyzing...'},
    ]

    input_msg = AgentMessage(
        sender='orchestrator',
        content=formatted_messages,
        extra_info={'context_tokens': 80},
    )

    # Should trigger compact (80 > 100 * 0.5 = 50)
    assert compact.should_compact(input_msg) is True

    result = await compact(input_msg)

    # Verify LLM received properly formatted messages
    assert received['messages'][0]['role'] == 'system'
    assert 'CRITICAL' in received['messages'][0]['content']  # COMPACT_PROMPT
    assert 'USER: Refactor memory' in received['messages'][1]['content']

    content = result.content
    if isinstance(content, dict):
        content = content.get('content', '')
    assert 'Summary' in content


# ── Test 3: ContextBuilder handles compact_boundary ───────────────

async def test_context_builder_with_compact_boundary():
    """Verify ContextBuilder skips messages before boundary and prepends summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        builder = InternClawContextBuilder(workspace)

        mem = Memory()
        # Simulate 6 messages, compact happened at index 4
        summary = "## Summary\nUser discussed memory refactoring."
        env_info_with_compact = {
            'conversation_summary': summary,
            'compact_boundary': 4,
        }

        mem.add(AgentMessage(sender='user', content='msg0', role='user'))
        mem.add(AgentMessage(sender='agent', content='msg1', role='assistant'))
        mem.add(AgentMessage(sender='user', content='msg2', role='user'))
        mem.add(AgentMessage(sender='agent', content='msg3', role='assistant'))
        # This message carries the compact info
        mem.add(AgentMessage(
            sender='user', content='msg4 (after compact)', role='user',
            env_info=env_info_with_compact,
        ))
        mem.add(AgentMessage(sender='agent', content='msg5', role='assistant'))

        messages, tools = builder.aggregate(mem, name='agent')

        # Should have: system + summary + msg4 + msg5
        # msg0-msg3 should be skipped (before boundary index 4)
        contents = [m.get('content', '') for m in messages]

        # First is system prompt
        assert messages[0]['role'] == 'system'

        # Second should be the injected summary
        assert 'Conversation Summary' in contents[1]
        assert 'memory refactoring' in contents[1]

        # Messages after boundary should be present
        assert any('msg4' in c for c in contents), f"msg4 not found in {contents}"
        assert any('msg5' in c for c in contents), f"msg5 not found in {contents}"

        # Messages before boundary should NOT be present
        assert not any('msg0' in c for c in contents), "msg0 should be skipped"
        assert not any('msg1' in c for c in contents), "msg1 should be skipped"


async def test_context_builder_without_compact():
    """Verify ContextBuilder works normally without compact info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        builder = InternClawContextBuilder(workspace)

        mem = Memory()
        mem.add(AgentMessage(sender='user', content='hello', role='user'))
        mem.add(AgentMessage(sender='agent', content='hi', role='assistant'))

        messages, tools = builder.aggregate(mem, name='agent')

        contents = [m.get('content', '') for m in messages]
        assert any('hello' in c for c in contents)
        assert any('hi' in c for c in contents)


# ── Test 4: Provider + SaveMemoryAction + ContextBuilder ──────────

async def test_provider_action_contextbuilder_flow():
    """Full flow: write via action → read via provider → inject into ContextBuilder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Write memory via action
        action = SaveMemoryAction(workspace)
        await action.save_memory(
            memory_update="# Facts\n- User likes Rust",
            history_entry="[2026-04-09 10:00] User stated preference for Rust",
        )

        # Read via provider
        provider = OpenClawMemoryProvider(workspace)
        info = await provider.get_info()
        assert 'Rust' in info['long_term']

        # Build context with provider info
        builder = InternClawContextBuilder(workspace)
        mem = Memory()
        mem.add(AgentMessage(
            sender='user', content='What do I like?', role='user',
            env_info={'memory': info},
        ))

        messages, tools = builder.aggregate(mem, name='agent')
        system_prompt = messages[0]['content']
        assert 'Rust' in system_prompt  # Memory injected into system prompt


# ── Test 5: Full InternClawAgent loop with compact ────────────────

async def test_internclaw_compact_triggers():
    """Verify compact triggers during InternClawAgent loop when tokens exceed threshold."""
    compact_called = {'count': 0}

    class PolicyLLM:
        """Simulates policy: returns tool_calls for first N turns, then stops."""
        def __init__(self):
            self._turn = 0

        async def chat(self, messages, **kwargs):
            self._turn += 1
            if self._turn <= 3:
                return {
                    'content': f'Let me check turn {self._turn}',
                    'tool_calls': [{'id': f'call_{self._turn}', 'function': {'name': 'test_tool', 'arguments': '{}'}}],
                }
            return {'content': 'All done, no more tools needed.'}

    class CompactLLM:
        async def chat(self, messages, **kwargs):
            compact_called['count'] += 1
            return {'content': '## Summary\nCompacted conversation.'}

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Very low threshold so compact triggers easily
        compact = AsyncCompactAgent(
            llm=CompactLLM(),
            max_context_tokens=50,
            threshold_ratio=0.1,  # triggers at 5 tokens
        )

        from lagent.agents.aggregator import DefaultAggregator
        policy = AsyncPolicyAgent(llm=PolicyLLM(), aggregator=DefaultAggregator())

        # Minimal env that just passes through
        env = AsyncEnvAgent(actions=[])

        agent = InternClawAgent(
            policy_agent=policy,
            env_agent=env,
            compact_agent=compact,
            max_turn=5,
        )

        result = await agent("Start a conversation about memory refactoring")

        # Compact should have been called at least once
        assert compact_called['count'] > 0, \
            f"Compact should have triggered, but was called {compact_called['count']} times"


# ── Real model test ───────────────────────────────────────────────

async def test_real_compact_with_provider():
    """Integration: real LLM + provider + compact."""
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    model_name = "gpt-5.4-mini"
    api_base = "http://35.220.164.252:3888/v1"
    api_key = "" 
    proxy = "http://100.100.72.89:8899"

    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxy),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=600,
        max_retry=500,
        sleep_interval=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Setup provider with some existing memory
        mem_dir = workspace / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text(
            "# Project Context\n- Working on lagent memory refactoring\n- User prefers minimal abstractions"
        )

        provider = OpenClawMemoryProvider(workspace)
        compact = AsyncCompactAgent(
            llm=model,
            max_context_tokens=128_000,
        )

        # Build formatted_messages that include provider content
        env_info = await provider.get_info()

        formatted_messages = [
            {'role': 'system', 'content': f"You are helpful.\n\nMemory:\n{env_info.get('long_term', '')}"},
            {'role': 'user', 'content': 'Help me design the memory system'},
            {'role': 'assistant', 'content': 'Based on the project context, I see you prefer minimal abstractions. Let me propose a design.'},
            {'role': 'user', 'content': 'Yes, Memory should be a pure list, no LTM base class'},
            {'role': 'assistant', 'content': 'Agreed. Provider=read, Action=write, both independent.'},
            {'role': 'user', 'content': 'What about compact?'},
            {'role': 'assistant', 'content': 'CompactAgent is a standard AsyncAgent with its own aggregator.'},
        ]

        input_msg = AgentMessage(sender='orchestrator', content=formatted_messages)

        print("\n  Calling real model for compact + provider integration...")
        result = await compact(input_msg)

        content = result.content
        if isinstance(content, dict):
            content = content.get('content', '')

        print(f"  Summary length: {len(content)} chars")
        print(f"  First 300 chars:\n    {content[:300]}")

        assert content and len(content) > 50, "Summary should be substantial"
        content_lower = content.lower()
        assert any(w in content_lower for w in ['memory', 'compact', 'provider']), \
            "Summary should mention key topics from the conversation"

        # Verify the provider's memory content influenced the summary
        assert any(w in content_lower for w in ['minimal', 'abstraction', 'lagent']), \
            "Summary should reflect project context from provider"

        print("  Real compact + provider: OK")


# ── Runner ────────────────────────────────────────────────────────

async def main():
    run_real = True

    mock_tests = [
        test_provider_injects_into_env_info,
        test_compact_formats_and_summarizes,
        test_context_builder_with_compact_boundary,
        test_context_builder_without_compact,
        test_provider_action_contextbuilder_flow,
        test_internclaw_compact_triggers,
    ]

    for test in mock_tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  Mock tests: ALL PASSED")

    if run_real:
        print("\n  --- Real model integration tests ---")
        await test_real_compact_with_provider()
        print("  Real integration tests: ALL PASSED")
    else:
        print("\n  (skip real model tests, use --real to enable)")

    print("\n  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
