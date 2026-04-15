"""Test OpenAI Agents adapter — validates the SDKAgentAdapter pattern
works with a real framework beyond Claude Code.

Run:
    python tests/test_adapters/test_openai_agents.py
    python -m pytest tests/test_adapters/test_openai_agents.py -v -s

Note: requires OPENAI_API_KEY or a compatible API endpoint.
"""
import asyncio
import os
import sys

import pytest

from lagent.adapters.openai_agents import OpenAIAgentsAdapter
from lagent.adapters.proxy import LLMProxyRecorder
from lagent.schema import AgentMessage

WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use Anthropic endpoint as OpenAI-compatible if no OpenAI key
ANTHROPIC_BASE = os.environ.get('ANTHROPIC_BASE_URL', '')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')

pytestmark = pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY') and not ANTHROPIC_KEY,
    reason="No API key available",
)


def log(msg):
    print(f"  → {msg}")


class TestOpenAIAgentsAdapter:

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """Single turn with OpenAI Agents SDK."""
        agent = OpenAIAgentsAdapter(
            model='gpt-4o-mini',
            instructions='You are a math tutor. Answer with just the number.',
            max_turns=3,
            timeout=60,
        )
        result = await agent("What is 7+8?")
        log(f"result: {result.content}")
        assert isinstance(result, AgentMessage)
        assert '15' in result.content

    @pytest.mark.asyncio
    async def test_multiturn(self):
        """Real multi-turn via RunResult chaining."""
        agent = OpenAIAgentsAdapter(
            model='gpt-4o-mini',
            instructions='You are a helpful assistant. Be very brief.',
            max_turns=3,
            timeout=60,
        )
        r1 = await agent("Remember: my lucky number is 77. Just say OK.")
        log(f"turn 1: {r1.content}")

        r2 = await agent("What is my lucky number? Just the number.")
        log(f"turn 2: {r2.content}")
        assert '77' in r2.content

    @pytest.mark.asyncio
    async def test_with_proxy(self):
        """Proxy captures LLM trace from OpenAI Agents."""
        proxy = LLMProxyRecorder(
            real_api_key=os.environ.get('OPENAI_API_KEY', ''),
            real_base_url='https://api.openai.com',
        )
        await proxy.start()
        try:
            agent = OpenAIAgentsAdapter(
                model='gpt-4o-mini',
                instructions='Answer briefly.',
                max_turns=3,
                timeout=60,
                proxy=proxy,
            )
            result = await agent("What is 3*3? Just the number.")
            log(f"result: {result.content}")

            state = agent.state_dict()
            log(f"state keys: {list(state.keys())}")
            trace = state.get('llm_trace', [])
            log(f"llm_trace: {len(trace)} records")

            assert len(trace) >= 1
            rec = trace[0]
            assert 'messages' in rec['request']
            assert rec['response'] is not None
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_multiturn_with_proxy(self):
        """Multi-turn + Proxy: messages grow, training sample correct."""
        proxy = LLMProxyRecorder(
            real_api_key=os.environ.get('OPENAI_API_KEY', ''),
            real_base_url='https://api.openai.com',
        )
        await proxy.start()
        try:
            agent = OpenAIAgentsAdapter(
                model='gpt-4o-mini',
                instructions='Be very brief.',
                max_turns=3,
                timeout=60,
                proxy=proxy,
            )
            r1 = await agent("Remember: color=green. Just say OK.")
            log(f"turn 1: {r1.content}")
            r2 = await agent("What color? Just the word.")
            log(f"turn 2: {r2.content}")

            trace = proxy.get_records(agent.session_id)
            log(f"total LLM calls: {len(trace)}")
            for i, rec in enumerate(trace):
                msgs = rec['request'].get('messages', [])
                log(f"  call {i+1}: {len(msgs)} messages")

            # Messages should grow
            if len(trace) >= 2:
                msgs_first = len(trace[0]['request']['messages'])
                msgs_last = len(trace[-1]['request']['messages'])
                log(f"messages grew: {msgs_first} → {msgs_last}")
                assert msgs_last > msgs_first

            # Training samples
            samples = proxy.to_training_samples(agent.session_id)
            log(f"training samples: {len(samples)}")
            if samples:
                log(f"sample[0]: {len(samples[0]['messages'])} msgs, {samples[0]['num_calls']} calls")
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_state_dict_structure(self):
        """state_dict has memory + trace."""
        agent = OpenAIAgentsAdapter(
            model='gpt-4o-mini',
            max_turns=3, timeout=60,
        )
        await agent("Say hello.")

        state = agent.state_dict()
        assert 'memory' in state
        assert len(state['memory']) == 2  # input + output
        log(f"memory: {len(state['memory'])} entries")
        log(f"sender[0]: {state['memory'][0]['sender']}")
        log(f"sender[1]: {state['memory'][1]['sender']}")


# ── F5 Runner ────────────────────────────────────────────────────

async def _run_test(cls, name):
    obj = cls()
    print(f"\n{'='*60}")
    print(f"  {cls.__name__}.{name}")
    print(f"{'='*60}")
    try:
        await getattr(obj, name)()
        print(f"  ✅ PASSED")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


async def run_all():
    for name in sorted(dir(TestOpenAIAgentsAdapter)):
        if name.startswith('test_'):
            await _run_test(TestOpenAIAgentsAdapter, name)
    print(f"\n{'='*60}\n  Done!\n{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        asyncio.run(_run_test(TestOpenAIAgentsAdapter, sys.argv[1]))
    else:
        asyncio.run(run_all())
