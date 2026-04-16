"""Test OpenAI Agents adapter with InternClaw's API endpoint."""
import asyncio
import json
import os
import sys

from lagent.adapters.openai_agents import OpenAIAgentsAdapter
from lagent.adapters.proxy import LLMProxyRecorder

API_KEY = ' '
BASE_URL = 'http://35.220.164.252:3888/v1'
HTTP_PROXY = 'http://100.100.72.89:8899'


def log(msg):
    print(f"  → {msg}")


async def test_single_turn():
    print("\n" + "=" * 60)
    print("  OpenAI Agents: Single Turn")
    print("=" * 60)

    agent = OpenAIAgentsAdapter(
        model='gpt-4o-mini',
        instructions='Answer with just the number, nothing else.',
        max_turns=3,
        timeout=30,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_proxy=HTTP_PROXY,
    )

    result = await agent("What is 7+8?")
    log(f"result: {result.content}")
    log(f"sender: {result.sender}")
    assert '15' in result.content
    print("  ✅ PASSED")


async def test_multiturn():
    print("\n" + "=" * 60)
    print("  OpenAI Agents: Multi-Turn")
    print("=" * 60)

    agent = OpenAIAgentsAdapter(
        model='gpt-4o-mini',
        instructions='You are helpful. Be very brief.',
        max_turns=3,
        timeout=30,
        api_key=API_KEY,
        base_url=BASE_URL,
        http_proxy=HTTP_PROXY,
    )

    r1 = await agent("Remember: my pet's name is Muffin. Just say OK.")
    log(f"turn 1: {r1.content}")

    r2 = await agent("What is my pet's name? Just the name.")
    log(f"turn 2: {r2.content}")
    assert 'muffin' in r2.content.lower()

    log(f"memory: {len(agent.memory.get_memory())} entries")
    print("  ✅ PASSED")


async def test_with_proxy():
    print("\n" + "=" * 60)
    print("  OpenAI Agents: With LLM Proxy")
    print("=" * 60)

    proxy = LLMProxyRecorder(
        real_api_key=API_KEY,
        real_base_url=BASE_URL,
    )
    await proxy.start()
    log(f"proxy on: {proxy.url}")

    try:
        agent = OpenAIAgentsAdapter(
            model='gpt-4o-mini',
            instructions='Be brief.',
            max_turns=3,
            timeout=30,
            api_key=API_KEY,
            base_url=BASE_URL,
            http_proxy=HTTP_PROXY,
            proxy=proxy,
        )

        r1 = await agent("Remember: city=Tokyo. Just say OK.")
        log(f"turn 1: {r1.content}")
        r2 = await agent("What city? Just the name.")
        log(f"turn 2: {r2.content}")

        state = agent.state_dict()
        log(f"state keys: {list(state.keys())}")

        trace = state.get('llm_trace', [])
        log(f"llm_trace: {len(trace)} LLM calls")
        for i, rec in enumerate(trace):
            msgs = rec['request'].get('messages', [])
            resp = rec.get('response') or {}
            usage = resp.get('usage', {})
            log(f"  call {i+1}: {len(msgs)} msgs, "
                f"in={usage.get('prompt_tokens', '?')} "
                f"out={usage.get('completion_tokens', '?')}")

        # Training samples
        samples = proxy.to_training_samples(agent.session_id)
        log(f"training samples: {len(samples)}")
        if samples:
            s = samples[0]
            log(f"  sample: {len(s['messages'])} msgs, {s['num_calls']} calls, model={s['model']}")

        log(f"memory: {len(agent.memory.get_memory())} entries")
        print("  ✅ PASSED")
    finally:
        await proxy.stop()


async def test_state_dict_comparison_with_claude():
    print("\n" + "=" * 60)
    print("  Comparison: OpenAI Agents vs Claude Code state_dict")
    print("=" * 60)

    from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter

    oai = OpenAIAgentsAdapter(
        model='gpt-4o-mini',
        instructions='Be brief.',
        max_turns=3, timeout=30,
        api_key=API_KEY, base_url=BASE_URL, http_proxy=HTTP_PROXY,
    )
    claude = ClaudeCodeSDKAdapter(
        max_turns=3, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    await oai("Say 'hello'.")
    await claude("Say 'hello'.")

    oai_state = oai.state_dict()
    claude_state = claude.state_dict()

    log(f"OpenAI keys: {sorted(oai_state.keys())}")
    log(f"Claude keys: {sorted(claude_state.keys())}")

    # Both have memory with same structure
    log(f"OpenAI memory: {len(oai_state['memory'])} entries")
    log(f"Claude memory: {len(claude_state['memory'])} entries")
    assert len(oai_state['memory']) == 2
    assert len(claude_state['memory']) == 2

    log(f"OpenAI sender: {oai_state['memory'][1]['sender']}")
    log(f"Claude sender: {claude_state['memory'][1]['sender']}")
    print("  ✅ PASSED")


async def main():
    await test_single_turn()
    await test_multiturn()
    await test_with_proxy()
    await test_state_dict_comparison_with_claude()
    print(f"\n{'='*60}")
    print("  All OpenAI Agents tests passed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())
