"""End-to-end tests for InternClawAgent with all memory modules.

Tests:
  1. Consolidate agent (SaveMemoryAction writes to MEMORY.md + HISTORY.md)
  2. Full pipeline: policy + env + compact + consolidate + provider + contextbuilder
  3. Real model full pipeline

Usage:
    python tests/test_agents/test_internclaw_e2e.py          # mock only
    python tests/test_agents/test_internclaw_e2e.py --real    # real model
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from lagent.actions.save_memory import AsyncSaveMemoryAction, SaveMemoryAction
from lagent.agents import AsyncAgent
from lagent.agents.aggregator.context import InternClawContextBuilder
from lagent.agents.aggregator.default_aggregator import DefaultAggregator
from lagent.agents.compact_agent import AsyncCompactAgent, estimate_token_count
from lagent.agents.internclaw_agent import AsyncEnvAgent, InternClawAgent
from lagent.memory import Memory, OpenClawMemoryProvider
from lagent.schema import AgentMessage

# ── Test 1: Consolidate agent writes to LTM ──────────────────────


async def test_consolidate_agent_writes_memory():
    """Consolidate agent = InternClawAgent with SaveMemoryAction.

    Verify it calls save_memory tool and writes to MEMORY.md + HISTORY.md.
    """

    class ConsolidateLLM:
        """Mock LLM that calls save_memory tool."""

        async def chat(self, messages, **kwargs):
            return {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'function': {
                            'name': 'AsyncSaveMemoryAction',
                            'arguments': json.dumps(
                                {
                                    'history_entry': '[2026-04-09 12:00] Discussed memory refactoring. Decided to split compact and LTM.',
                                    'memory_update': '# Facts\n- User prefers minimal abstractions\n- Memory is a pure list\n- CompactAgent is a standard AsyncAgent',
                                }
                            ),
                        },
                    }
                ],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        save_action = AsyncSaveMemoryAction(workspace)

        consolidate_policy = AsyncAgent(llm=ConsolidateLLM())
        consolidate_env = AsyncEnvAgent(actions=[save_action])

        consolidate_agent = InternClawAgent(
            policy_agent=consolidate_policy,
            env_agent=consolidate_env,
            max_turn=1,
            finish_condition=None,
        )

        # Run consolidation
        input_msg = AgentMessage(
            sender='orchestrator',
            content='Consolidate the following conversation...',
        )
        await consolidate_agent(input_msg)

        # Verify MEMORY.md was written
        memory_file = workspace / "memory" / "MEMORY.md"
        assert memory_file.exists(), "MEMORY.md should exist"
        content = memory_file.read_text()
        assert 'minimal abstractions' in content
        assert 'pure list' in content

        # Verify HISTORY.md was appended
        history_file = workspace / "memory" / "HISTORY.md"
        assert history_file.exists(), "HISTORY.md should exist"
        history = history_file.read_text()
        assert '[2026-04-09 12:00]' in history
        assert 'memory refactoring' in history


# ── Test 2: Consolidate + Provider round-trip ─────────────────────


async def test_consolidate_then_provider_reads():
    """Consolidate writes → provider reads back the same content."""

    class ConsolidateLLM:
        async def chat(self, messages, **kwargs):
            return {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'function': {
                            'name': 'AsyncSaveMemoryAction',
                            'arguments': json.dumps(
                                {
                                    'history_entry': '[2026-04-09] Round-trip test',
                                    'memory_update': '# Memory\n- Round-trip test passed',
                                }
                            ),
                        },
                    }
                ],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Consolidate writes
        consolidate = InternClawAgent(
            policy_agent=AsyncAgent(llm=ConsolidateLLM()),
            env_agent=AsyncEnvAgent(actions=[AsyncSaveMemoryAction(workspace)]),
            max_turn=1,
            finish_condition=None,
        )
        await consolidate(AgentMessage(sender='test', content='consolidate'))

        # Provider reads
        provider = OpenClawMemoryProvider(workspace)
        info = await provider.get_info()
        assert info['available'] is True
        assert 'Round-trip test passed' in info['long_term']


# ── Test 3: Full pipeline mock — compact + consolidate ────────────


async def test_full_pipeline_compact_and_consolidate():
    """Full InternClawAgent pipeline: policy loops → compact triggers →
    consolidate writes LTM → compact compresses context.
    """
    consolidate_called = {'count': 0}
    compact_called = {'count': 0}

    class PolicyLLM:
        def __init__(self):
            self._turn = 0

        async def chat(self, messages, **kwargs):
            self._turn += 1
            if self._turn <= 2:
                return {
                    'content': f'Working on turn {self._turn}...',
                    'tool_calls': [
                        {
                            'id': f'call_{self._turn}',
                            'function': {'name': 'test_tool', 'arguments': '{}'},
                        }
                    ],
                }
            return {'content': 'All done.'}

    class ConsolidateLLM:
        async def chat(self, messages, **kwargs):
            consolidate_called['count'] += 1
            return {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'cons_1',
                        'function': {
                            'name': 'AsyncSaveMemoryAction',
                            'arguments': json.dumps(
                                {
                                    'history_entry': '[2026-04-09] Consolidated',
                                    'memory_update': '# Consolidated memory',
                                }
                            ),
                        },
                    }
                ],
            }

    class CompactLLM:
        async def chat(self, messages, **kwargs):
            compact_called['count'] += 1
            return {'content': '## Summary\nConversation was about testing.'}

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        provider = OpenClawMemoryProvider(workspace)

        # Consolidate agent
        consolidate = InternClawAgent(
            policy_agent=AsyncAgent(llm=ConsolidateLLM()),
            env_agent=AsyncEnvAgent(actions=[AsyncSaveMemoryAction(workspace)]),
            max_turn=1,
            finish_condition=None,
        )

        # Compact agent — very low threshold to trigger
        compact = AsyncCompactAgent(
            llm=CompactLLM(),
            max_context_tokens=50,
            threshold_ratio=0.1,
        )

        # Main agent
        agent = InternClawAgent(
            policy_agent=AsyncAgent(
                llm=PolicyLLM(),
                aggregator=DefaultAggregator(),
            ),
            env_agent=AsyncEnvAgent(
                actions=[],
                long_term_memory=provider,
            ),
            compact_agent=compact,
            consolidate_agent=consolidate,
            max_turn=5,
        )

        result = await agent("Test the full pipeline")

        # Both should have been called
        assert compact_called['count'] > 0, "Compact should have triggered"
        assert consolidate_called['count'] > 0, "Consolidate should have triggered"

        # MEMORY.md should have been written by consolidation
        memory_file = workspace / "memory" / "MEMORY.md"
        assert memory_file.exists(), "Consolidation should have written MEMORY.md"


# ── Test 4: ContextBuilder with compact + provider together ──────


async def test_context_builder_full_assembly():
    """ContextBuilder assembles: system prompt (with LTM) + compact summary + recent messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        mem_dir = workspace / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("# Facts\n- User likes Python")

        builder = InternClawContextBuilder(workspace)
        provider = OpenClawMemoryProvider(workspace)
        env_info = await provider.get_info()

        # Build a session with compact boundary
        mem = Memory()
        # Old messages (before compact)
        mem.add(AgentMessage(sender='user', content='old msg 0', role='user'))
        mem.add(AgentMessage(sender='agent', content='old msg 1', role='assistant'))
        mem.add(AgentMessage(sender='user', content='old msg 2', role='user'))
        mem.add(AgentMessage(sender='agent', content='old msg 3', role='assistant'))
        # Message carrying compact info + provider memory
        env_info_with_compact = {
            **env_info,
            'conversation_summary': '## Summary\nDiscussed Python preferences.',
            'compact_boundary': 4,
        }
        mem.add(
            AgentMessage(
                sender='user',
                content='new msg after compact',
                role='user',
                env_info=env_info_with_compact,
            )
        )
        mem.add(AgentMessage(sender='agent', content='response after compact', role='assistant'))

        messages, tools = builder.aggregate(mem, name='agent')

        # System prompt should contain LTM content
        system_prompt = messages[0]['content']
        assert 'Python' in system_prompt, "System prompt should include LTM facts"

        # Summary should be injected
        all_content = ' '.join(m.get('content', '') or '' for m in messages)
        assert 'Summary' in all_content, "Compact summary should be present"
        assert 'Python preferences' in all_content

        # New messages should be present
        assert 'new msg after compact' in all_content
        assert 'response after compact' in all_content

        # Old messages should NOT be present
        assert 'old msg 0' not in all_content, "Old messages should be skipped"
        assert 'old msg 3' not in all_content


# ── Test 5: Multiple compact rounds ──────────────────────────────


async def test_multiple_compact_rounds():
    """Verify compact can trigger multiple times in a long session."""
    compact_count = {'n': 0}

    class PolicyLLM:
        def __init__(self):
            self._turn = 0

        async def chat(self, messages, **kwargs):
            self._turn += 1
            if self._turn <= 6:
                return {
                    'content': f'Turn {self._turn} ' + 'x' * 100,  # padding for tokens
                    'tool_calls': [
                        {
                            'id': f'c_{self._turn}',
                            'function': {'name': 'noop', 'arguments': '{}'},
                        }
                    ],
                }
            return {'content': 'Done.'}

    class CompactLLM:
        async def chat(self, messages, **kwargs):
            compact_count['n'] += 1
            return {'content': f'## Summary round {compact_count["n"]}'}

    compact = AsyncCompactAgent(
        llm=CompactLLM(),
        max_context_tokens=30,
        threshold_ratio=0.1,
    )

    agent = InternClawAgent(
        policy_agent=AsyncAgent(
            llm=PolicyLLM(),
            aggregator=DefaultAggregator(),
        ),
        env_agent=AsyncEnvAgent(actions=[]),
        compact_agent=compact,
        max_turn=8,
    )

    await agent("Start a long conversation")
    assert compact_count['n'] >= 2, f"Expected multiple compacts, got {compact_count['n']}"


# ── Real model test ───────────────────────────────────────────────


async def test_real_full_pipeline():
    """Real model: consolidate + compact + provider full round-trip."""
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

        # Seed initial memory
        mem_dir = workspace / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("# Initial\n- Project: lagent memory refactoring")

        provider = OpenClawMemoryProvider(workspace)

        # Consolidate agent with real model
        consolidate = InternClawAgent(
            policy_agent=AsyncAgent(llm=model),
            env_agent=AsyncEnvAgent(actions=[AsyncSaveMemoryAction(workspace)]),
            max_turn=1,
            finish_condition=None,
        )

        # Compact agent with real model
        compact = AsyncCompactAgent(llm=model, max_context_tokens=128_000)

        # Main agent — just test consolidate independently first
        print("\n  Testing consolidation with real model...")
        consolidate_input = AgentMessage(
            sender='test',
            content=(
                'Consolidate this conversation:\n'
                'USER: I want to refactor the memory system\n'
                'ASSISTANT: Sure, we should split compact from LTM\n'
                'USER: Memory should be a pure list\n'
                'ASSISTANT: Agreed, no state tracking in Memory class\n'
                '\nCall the save_memory tool with your consolidation.'
            ),
        )
        await consolidate(consolidate_input)

        # Check if consolidation wrote to files
        memory_content = (workspace / "memory" / "MEMORY.md").read_text()
        print(f"  MEMORY.md after consolidation ({len(memory_content)} chars):")
        print(f"    {memory_content[:200]}...")

        # Provider should reflect the update
        info = await provider.get_info()
        print(f"  Provider get_info: available={info.get('available')}")

        # Now test compact with the same model
        print("\n  Testing compact with real model...")
        formatted_messages = [
            {'role': 'system', 'content': f"You are helpful.\n\nMemory:\n{info.get('long_term', '')}"},
            {'role': 'user', 'content': 'Refactor memory system'},
            {'role': 'assistant', 'content': 'Split compact from LTM'},
            {'role': 'user', 'content': 'Memory = pure list'},
            {'role': 'assistant', 'content': 'Done, no state tracking'},
        ]

        compact_result = await compact(
            AgentMessage(
                sender='test',
                content=formatted_messages,
            )
        )

        content = compact_result.content
        if isinstance(content, dict):
            content = content.get('content', '')
        print(f"  Compact summary ({len(content)} chars):")
        print(f"    {content[:200]}...")

        assert content and len(content) > 20, "Compact summary should be substantial"

        print("\n  Real full pipeline: OK")


# ── Runner ────────────────────────────────────────────────────────


async def main():
    run_real = True

    mock_tests = [
        test_consolidate_agent_writes_memory,
        test_consolidate_then_provider_reads,
        test_full_pipeline_compact_and_consolidate,
        test_context_builder_full_assembly,
        test_multiple_compact_rounds,
    ]

    for test in mock_tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  Mock tests: ALL PASSED")

    if run_real:
        print("\n  --- Real model E2E tests ---")
        await test_real_full_pipeline()
        print("  Real E2E tests: ALL PASSED")
    else:
        print("\n  (skip real model tests, use --real to enable)")

    print("\n  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
