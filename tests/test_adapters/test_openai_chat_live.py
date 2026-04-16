"""Test OpenAI Chat adapter with InternClaw's API endpoint.

Run:
    python tests/test_adapters/test_openai_chat_live.py
"""
import asyncio
import json
import os
import sys

from lagent.adapters.openai_chat import OpenAIChatAdapter
from lagent.adapters.proxy import LLMProxyRecorder

API_KEY = ' '
BASE_URL = 'http://35.220.164.252:3888/v1'
HTTP_PROXY = 'http://100.100.72.89:8899'


def log(msg):
    print(f"  → {msg}")


async def test_single_turn():
    print("\n" + "=" * 60)
    print("  OpenAI Chat: Single Turn")
    print("=" * 60)

    agent = OpenAIChatAdapter(
        model='gpt-4o-mini',
        api_key=API_KEY, base_url=BASE_URL, http_proxy=HTTP_PROXY,
        system_prompt='Answer with just the number, nothing else.',
        timeout=30,
    )
    result = await agent("What is 7+8?")
    log(f"result: {result.content}")
    assert '15' in result.content
    print("  ✅ PASSED")


async def test_multiturn():
    print("\n" + "=" * 60)
    print("  OpenAI Chat: Multi-Turn (real)")
    print("=" * 60)

    agent = OpenAIChatAdapter(
        model='gpt-4o-mini',
        api_key=API_KEY, base_url=BASE_URL, http_proxy=HTTP_PROXY,
        system_prompt='You are helpful. Be very brief.',
        timeout=30,
    )

    r1 = await agent("Remember: my pet is called Muffin. Just say OK.")
    log(f"turn 1: {r1.content}")

    r2 = await agent("What is my pet's name? Just the name.")
    log(f"turn 2: {r2.content}")
    assert 'muffin' in r2.content.lower()

    r3 = await agent("Say it in uppercase.")
    log(f"turn 3: {r3.content}")
    assert 'MUFFIN' in r3.content.upper()

    log(f"memory: {len(agent.memory.get_memory())} entries")
    log(f"internal messages: {len(agent._messages)}")
    print("  ✅ PASSED")


async def test_with_proxy():
    print("\n" + "=" * 60)
    print("  OpenAI Chat: With LLM Proxy")
    print("=" * 60)

    proxy = LLMProxyRecorder(
        real_api_key=API_KEY,
        real_base_url=BASE_URL,
        http_proxy=HTTP_PROXY,
    )
    await proxy.start()
    log(f"proxy on: {proxy.url}")

    try:
        agent = OpenAIChatAdapter(
            model='gpt-4o-mini',
            api_key=API_KEY, base_url=BASE_URL, http_proxy=HTTP_PROXY,
            system_prompt='Be brief.',
            timeout=30,
            proxy=proxy,
        )

        r1 = await agent("Remember: city=Paris. Just say OK.")
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
                f"tokens: {json.dumps(usage)[:100]}")

        # Messages should grow
        if len(trace) >= 2:
            m1 = len(trace[0]['request']['messages'])
            m2 = len(trace[-1]['request']['messages'])
            log(f"messages grew: {m1} → {m2}")
            assert m2 > m1

        # Training samples
        samples = proxy.to_training_samples(agent.session_id)
        log(f"training samples: {len(samples)}")
        if samples:
            s = samples[0]
            log(f"  sample: {len(s['messages'])} msgs, {s['meta']['num_calls']} calls")
            log(f"  all_usage: {s['all_usage']}")

        print("  ✅ PASSED")
    finally:
        await proxy.stop()


async def test_comparison_with_claude():
    print("\n" + "=" * 60)
    print("  Comparison: OpenAI Chat vs Claude Code SDK state_dict")
    print("=" * 60)

    from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter

    oai = OpenAIChatAdapter(
        model='gpt-4o-mini',
        api_key=API_KEY, base_url=BASE_URL, http_proxy=HTTP_PROXY,
        system_prompt='Be brief.',
        timeout=30,
    )
    claude = ClaudeCodeSDKAdapter(
        max_turns=3, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    await oai("Say 'hello'.")
    await claude("Say 'hello'.")

    oai_state = oai.state_dict()
    claude_state = claude.state_dict()

    log(f"OpenAI state keys: {sorted(oai_state.keys())}")
    log(f"Claude state keys: {sorted(claude_state.keys())}")
    log(f"OpenAI memory: {len(oai_state['memory'])} entries")
    log(f"Claude memory: {len(claude_state['memory'])} entries")

    # Both have 2 memory entries (input + output)
    assert len(oai_state['memory']) == 2
    assert len(claude_state['memory']) == 2
    assert oai_state['memory'][0]['sender'] == 'user'
    assert claude_state['memory'][0]['sender'] == 'user'

    # Show content
    log(f"OpenAI output: {oai_state['memory'][1]['content']}")
    log(f"Claude output: {claude_state['memory'][1]['content']}")
    print("  ✅ PASSED")


async def main():
    await test_single_turn()
    await test_multiturn()
    await test_with_proxy()
    await test_comparison_with_claude()
    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())
