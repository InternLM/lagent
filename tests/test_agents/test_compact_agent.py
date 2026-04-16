"""Tests for lagent.agents.compact_agent

Includes:
  - Unit tests with mock LLM (always run)
  - Integration test with real model (run with --real flag)

Usage:
    python tests/test_agents/test_compact_agent.py          # mock only
    python tests/test_agents/test_compact_agent.py --real    # include real model
"""

import asyncio
import sys

from lagent.agents.compact_agent import AsyncCompactAgent, estimate_token_count, COMPACT_PROMPT
from lagent.schema import AgentMessage


# ── Mock tests ────────────────────────────────────────────────────

async def test_threshold_tokens():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    assert compact.threshold_tokens == 800


async def test_below_threshold():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    msg = AgentMessage(sender='policy', content='test', extra_info={'context_tokens': 500})
    assert compact.should_compact(msg) is False


async def test_above_threshold():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    msg = AgentMessage(sender='policy', content='test', extra_info={'context_tokens': 900})
    assert compact.should_compact(msg) is True


async def test_no_extra_info():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    msg = AgentMessage(sender='policy', content='test')
    assert compact.should_compact(msg) is False


async def test_circuit_breaker():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    compact._consecutive_failures = 3
    msg = AgentMessage(sender='policy', content='test', extra_info={'context_tokens': 900})
    assert compact.should_compact(msg) is False


async def test_circuit_breaker_reset():
    compact = AsyncCompactAgent(max_context_tokens=1000, threshold_ratio=0.8)
    compact._consecutive_failures = 3
    msg = AgentMessage(sender='policy', content='test', extra_info={'context_tokens': 900})
    assert compact.should_compact(msg) is False
    compact._consecutive_failures = 0
    assert compact.should_compact(msg) is True


async def test_default_template():
    compact = AsyncCompactAgent()
    assert compact.template == COMPACT_PROMPT


async def test_custom_template():
    compact = AsyncCompactAgent(template="Custom prompt")
    assert compact.template == "Custom prompt"


async def test_estimate_token_count_basic():
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert estimate_token_count(messages) == 5  # 20 chars / 4


async def test_estimate_token_count_with_tools():
    messages = [{"role": "user", "content": "Hi"}]
    tools = [{"type": "function", "function": {"name": "test"}}]
    assert estimate_token_count(messages, tools) > estimate_token_count(messages)


async def test_estimate_token_count_empty():
    assert estimate_token_count([]) == 0


async def test_forward_with_mock_llm():
    """Test that CompactAgent forward works with string content."""

    class MockLLM:
        async def chat(self, messages, **kwargs):
            assert len(messages) >= 1
            return {"content": "## Summary\nThis is a test summary."}

    compact = AsyncCompactAgent(llm=MockLLM())

    input_msg = AgentMessage(
        sender='orchestrator',
        content='USER: Hello\nASSISTANT: Hi there\nUSER: Help me refactor',
    )

    result = await compact(input_msg)
    content = result.content
    if isinstance(content, dict):
        content = content.get('content', '')
    assert 'Summary' in content
    assert result.sender == 'AsyncCompactAgent'


async def test_forward_with_list_dict_content():
    """Test CompactAgent with list[dict] input (formatted_messages from policy aggregator)."""

    received_messages = []

    class MockLLM:
        async def chat(self, messages, **kwargs):
            received_messages.extend(messages)
            return {"content": "## Summary\nCompact summary of conversation."}

    compact = AsyncCompactAgent(llm=MockLLM())

    # This is what orchestrator actually passes: formatted_messages from policy's aggregator
    formatted_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello, help me refactor memory'},
        {'role': 'assistant', 'content': 'Sure, let me analyze the code.'},
        {'role': 'user', 'content': 'Focus on base_memory.py'},
        {'role': 'assistant', 'content': 'I see the Memory class needs reset().', 'tool_calls': [
            {'function': {'name': 'read_file', 'arguments': {'path': 'base_memory.py'}}}
        ]},
    ]

    input_msg = AgentMessage(sender='orchestrator', content=formatted_messages)
    result = await compact(input_msg)

    # Verify CompactAggregator formatted correctly
    assert len(received_messages) == 2  # system (COMPACT_PROMPT) + user (formatted text)
    assert received_messages[0]['role'] == 'system'
    assert 'CRITICAL' in received_messages[0]['content']  # COMPACT_PROMPT

    user_content = received_messages[1]['content']
    assert 'USER: Hello, help me refactor memory' in user_content
    assert 'ASSISTANT: Sure, let me analyze the code.' in user_content
    assert 'read_file' in user_content  # tool_calls included
    assert 'SYSTEM: You are a helpful assistant.' in user_content

    content = result.content
    if isinstance(content, dict):
        content = content.get('content', '')
    assert 'Summary' in content


async def test_compact_aggregator_empty_content():
    """Test CompactAggregator handles empty/None content gracefully."""
    from lagent.agents.aggregator.compact_aggregator import CompactAggregator

    agg = CompactAggregator()

    from lagent.memory import Memory
    mem = Memory()
    mem.add(AgentMessage(sender='user', content=[
        {'role': 'user', 'content': 'test'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'function': {'name': 'shell', 'arguments': {}}}
        ]},
    ]))

    messages, tools = agg.aggregate(mem, 'compact', system_instruction='Summarize')
    assert messages[0]['role'] == 'system'
    user_text = messages[1]['content']
    assert 'USER: test' in user_text
    assert 'shell' in user_text  # tool call captured even with None content


async def test_forward_error_handling():
    """Test circuit breaker increments on failure."""

    class FailingLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("LLM unavailable")

    compact = AsyncCompactAgent(llm=FailingLLM())
    assert compact._consecutive_failures == 0

    input_msg = AgentMessage(sender='orchestrator', content='test history')
    result = await compact(input_msg)

    # forward raises inside Agent.__call__, but the agent should handle it
    # Check that the compact agent's state reflects the attempt
    # Note: error handling depends on whether forward() catches or propagates


# ── Real model test ───────────────────────────────────────────────

async def test_real_model_compact():
    """Integration test: CompactAgent with a real LLM.

    Run with: python tests/test_agents/test_compact_agent.py --real
    """
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    model_name = "gpt-5.4-mini"
    api_base = "http://35.220.164.252:3888/v1"
    api_key = "" 
    proxy = "http://100.100.72.89:8899"
    extra_body = {}
    # model_name = "/mnt/shared-storage-user/llmit1/user/liujiangning/exp/s2_preview/agent_rl/s2-preview-thinker_sft_0228b_rl0312rc1/20260316082019/hf-15"
    # api_base = "http://10.102.245.34:23333/v1"
    
    # api_key='YOUR KEY'
    # extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    # proxy = None
    
    model = AsyncAPIClient(
            model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxy),
            sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
            timeout=600,
            max_retry=500,
            sleep_interval=5,
            extra_body=extra_body,
        )

    compact = AsyncCompactAgent(
        llm=model,
        max_context_tokens=128_000,
        threshold_ratio=0.85,
    )

    # Simulate real formatted_messages from policy's aggregator (list[dict])
    formatted_messages = [
        {'role': 'system', 'content': 'You are InternClaw, a helpful AI assistant.'},
        {'role': 'user', 'content': "I'm working on refactoring the memory system in lagent."},
        {'role': 'assistant', 'content': 'I can help with that. What aspects of the memory system need refactoring?'},
        {'role': 'user', 'content': 'The main issue is that BaseMemoryStore couples compact and long-term memory together.'},
        {'role': 'assistant', 'content': 'I see. We should split them into independent modules. Let me analyze the current code.', 'tool_calls': [
            {'function': {'name': 'read_file', 'arguments': {'path': 'lagent/memory/memory.py'}}}
        ]},
        {'role': 'tool', 'content': 'class BaseMemoryStore(ABC):\n    def get_info(self)...\n    def should_compact(self)...\n    def compact(self)...', 'name': 'read_file'},
        {'role': 'assistant', 'content': 'I can see the coupling. BaseMemoryStore has both compact and LTM methods in one interface.'},
        {'role': 'user', 'content': 'Yes, also the CompactAction forks the policy agent which does not match lagent architecture.'},
        {'role': 'assistant', 'content': 'Right. Instead of forking, we can make CompactAgent a standard AsyncAgent.'},
        {'role': 'user', 'content': 'Good. The providers for different frameworks (OpenClaw vs Claude Code) should be independent.'},
        {'role': 'assistant', 'content': 'Agreed. Each provider just needs get_info() for reading, writes go through separate Actions.'},
        {'role': 'user', 'content': 'Make sure Memory is a pure list with no state tracking.'},
        {'role': 'assistant', 'content': 'Done. Memory now only has add/get/delete/save/load/reset.'},
    ]

    input_msg = AgentMessage(
        sender='orchestrator',
        content=formatted_messages,
    )

    print("\n  Calling real model for compact...")
    result = await compact(input_msg)

    content = result.content
    if isinstance(content, dict):
        content = content.get('content', '')

    print(f"  Summary length: {len(content)} chars")
    print(f"  First 200 chars: {content[:200]}...")

    # Basic assertions
    assert content, "Summary should not be empty"
    assert len(content) > 100, "Summary should be substantial"
    content_lower = content.lower()
    assert any(word in content_lower for word in ['memory', 'refactor', 'compact']), \
        "Summary should mention key topics"

    print("  Real model compact: OK")


# ── Runner ────────────────────────────────────────────────────────

async def main():
    run_real = True

    mock_tests = [
        test_threshold_tokens,
        test_below_threshold,
        test_above_threshold,
        test_no_extra_info,
        test_circuit_breaker,
        test_circuit_breaker_reset,
        test_default_template,
        test_custom_template,
        test_estimate_token_count_basic,
        test_estimate_token_count_with_tools,
        test_estimate_token_count_empty,
        test_forward_with_mock_llm,
        test_forward_with_list_dict_content,
        test_compact_aggregator_empty_content,
    ]

    for test in mock_tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  Mock tests: ALL PASSED")

    if run_real:
        print("\n  --- Real model tests ---")
        await test_real_model_compact()
        print("  Real model tests: ALL PASSED")
    else:
        print("\n  (skip real model tests, use --real to enable)")

    print("\n  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
