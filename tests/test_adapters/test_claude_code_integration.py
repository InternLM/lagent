"""Integration tests for Claude Code adapters (CLI + SDK + Proxy).

These tests hit the real Claude Code CLI / SDK, so they require:
  - claude CLI on PATH
  - ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL set
  - claude-agent-sdk installed

Run:
    python -m pytest tests/test_adapters/test_claude_code_integration.py -v -s

Debug (F5 in IDE):
    python tests/test_adapters/test_claude_code_integration.py
    python tests/test_adapters/test_claude_code_integration.py TestCLIAdapter.test_single_turn
"""
import asyncio
import json
import os
import sys

import pytest

from lagent.adapters.claude_code import ClaudeCodeAdapter
from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter
from lagent.adapters.proxy import LLMProxyRecorder
from lagent.actions.external_agent import ExternalAgentAction
from lagent.schema import AgentMessage

WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests if env not configured (only for pytest)
pytestmark = pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_AUTH_TOKEN'),
    reason="ANTHROPIC_AUTH_TOKEN not set",
)


# ── Helpers ──────────────────────────────────────────────────────

def make_proxy():
    return LLMProxyRecorder(
        real_api_key=os.environ.get('ANTHROPIC_AUTH_TOKEN', ''),
        real_base_url=os.environ.get('ANTHROPIC_BASE_URL', ''),
    )


def log(msg):
    print(f"  → {msg}")


# ── CLI Adapter Tests ────────────────────────────────────────────

class TestCLIAdapter:

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """CLI adapter returns correct result for a simple question."""
        agent = ClaudeCodeAdapter(
            max_turns=3, timeout=60, working_dir=WORKDIR,
        )
        result = await agent("What is 3+4? Answer with just the number.")
        log(f"result: {result.content}")
        assert isinstance(result, AgentMessage)
        assert '7' in result.content

    @pytest.mark.asyncio
    async def test_single_turn_with_proxy(self):
        """CLI adapter + Proxy captures LLM trace in state_dict."""
        proxy = make_proxy()
        await proxy.start()
        try:
            agent = ClaudeCodeAdapter(
                max_turns=3, timeout=60, proxy=proxy, working_dir=WORKDIR,
            )
            result = await agent("What is 5+5? Answer with just the number.")
            log(f"result: {result.content}")

            state = agent.state_dict()
            log(f"state keys: {list(state.keys())}")
            assert 'memory' in state
            assert 'llm_trace' in state

            trace = state['llm_trace']
            log(f"llm_trace records: {len(trace)}")
            assert len(trace) >= 1

            # Verify trace structure
            rec = trace[0]
            assert 'timestamp' in rec
            assert 'request' in rec
            assert 'response' in rec
            assert rec['request'].get('model') is not None
            assert rec['request'].get('messages') is not None
            assert rec['response'].get('content') is not None
            assert rec['response'].get('usage') is not None
            log(f"usage: {json.dumps(rec['response']['usage'])}")
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_multiturn_continue(self):
        """CLI adapter uses --continue for real multi-turn."""
        agent = ClaudeCodeAdapter(
            max_turns=3, timeout=60, working_dir=WORKDIR,
        )
        r1 = await agent("Remember the number 99. Just say 'OK'.")
        log(f"turn 1: {r1.content}")
        assert agent._call_count == 1

        r2 = await agent("What number did I say? Just the number.")
        log(f"turn 2: {r2.content}")
        assert agent._call_count == 2
        assert '99' in r2.content

    @pytest.mark.asyncio
    async def test_multiturn_with_proxy_trace(self):
        """CLI multi-turn + Proxy shows message history growing."""
        proxy = make_proxy()
        await proxy.start()
        try:
            agent = ClaudeCodeAdapter(
                max_turns=3, timeout=60, proxy=proxy, working_dir=WORKDIR,
            )
            r1 = await agent("Remember: color=red. Just say 'OK'.")
            log(f"turn 1: {r1.content}")
            r2 = await agent("What color? Just the color.")
            log(f"turn 2: {r2.content}")

            trace = agent.state_dict()['llm_trace']
            log(f"llm_trace: {len(trace)} calls")
            assert len(trace) >= 2

            # Second call should have more messages (history)
            msgs_first = len(trace[0]['request']['messages'])
            msgs_last = len(trace[-1]['request']['messages'])
            log(f"messages: first={msgs_first} last={msgs_last}")
            assert msgs_last > msgs_first
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_memory_accumulates(self):
        """Memory stores input+output for each turn."""
        agent = ClaudeCodeAdapter(
            max_turns=3, timeout=60, working_dir=WORKDIR,
        )
        r1 = await agent("Say 'hello'.")
        log(f"turn 1: {r1.content}")
        r2 = await agent("Say 'world'.")
        log(f"turn 2: {r2.content}")

        memory = agent.memory.get_memory()
        log(f"memory: {len(memory)} entries")
        assert len(memory) == 4  # 2 turns × (input + output)
        assert memory[0].sender == 'user'
        assert memory[1].sender == 'claude-code'
        assert memory[2].sender == 'user'
        assert memory[3].sender == 'claude-code'


# ── SDK Adapter Tests ────────────────────────────────────────────

class TestSDKAdapter:

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """SDK adapter returns correct result."""
        agent = ClaudeCodeSDKAdapter(
            max_turns=3, timeout=60, cwd=WORKDIR,
        )
        result = await agent("What is 6*7? Answer with just the number.")
        log(f"result: {result.content}")
        assert isinstance(result, AgentMessage)
        assert '42' in result.content

    @pytest.mark.asyncio
    async def test_sdk_trace_structure(self):
        """SDK trace contains structured events."""
        agent = ClaudeCodeSDKAdapter(
            max_turns=3, timeout=60, cwd=WORKDIR,
        )
        result = await agent("Say 'test'. Just the word.")
        log(f"result: {result.content}")

        state = agent.state_dict()
        assert 'sdk_trace' in state
        assert 'claude_session_id' in state
        log(f"session_id: {state['claude_session_id']}")

        trace = state['sdk_trace']
        types = [e['type'] for e in trace]
        log(f"event types: {types}")
        assert 'AssistantMessage' in types
        assert 'ResultMessage' in types

        # Check ResultMessage has cost info
        result_events = [e for e in trace if e['type'] == 'ResultMessage']
        assert len(result_events) >= 1
        log(f"cost: ${result_events[0].get('total_cost_usd')}")
        assert result_events[0].get('total_cost_usd') is not None
        assert result_events[0].get('session_id') is not None

    @pytest.mark.asyncio
    async def test_multiturn_resume(self):
        """SDK adapter uses session_id for real multi-turn."""
        agent = ClaudeCodeSDKAdapter(
            max_turns=3, timeout=60, cwd=WORKDIR,
        )
        r1 = await agent("Remember the word 'banana'. Just say 'OK'.")
        log(f"turn 1: {r1.content}")

        # session_id should be captured (from ResultMessage or AssistantMessage)
        assert agent._session_id is not None, (
            f"session_id not captured. sdk_trace types: "
            f"{[e.get('type') for e in agent._sdk_trace]}"
        )
        session_id = agent._session_id
        log(f"session_id: {session_id}")

        r2 = await agent("What word did I say? Just the word.")
        log(f"turn 2: {r2.content}")
        assert 'banana' in r2.content.lower()
        # Same session
        assert agent._session_id == session_id

    @pytest.mark.asyncio
    async def test_sdk_with_proxy(self):
        """SDK + Proxy gives both sdk_trace and llm_trace."""
        proxy = make_proxy()
        await proxy.start()
        try:
            agent = ClaudeCodeSDKAdapter(
                max_turns=3, timeout=60, proxy=proxy, cwd=WORKDIR,
            )
            result = await agent("What is 2+2? Just the number.")
            log(f"result: {result.content}")

            state = agent.state_dict()
            log(f"state keys: {list(state.keys())}")
            assert 'memory' in state
            assert 'sdk_trace' in state
            assert 'llm_trace' in state
            assert 'claude_session_id' in state

            # Both traces populated
            log(f"sdk_trace: {len(state['sdk_trace'])} events")
            log(f"llm_trace: {len(state['llm_trace'])} calls")
            assert len(state['sdk_trace']) >= 1
            assert len(state['llm_trace']) >= 1

            # LLM trace has full request/response
            rec = state['llm_trace'][0]
            assert 'messages' in rec['request']
            assert 'content' in rec['response']
            assert 'usage' in rec['response']
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_sdk_proxy_multiturn(self):
        """SDK + Proxy multi-turn: both traces grow, messages accumulate."""
        proxy = make_proxy()
        await proxy.start()
        try:
            agent = ClaudeCodeSDKAdapter(
                max_turns=3, timeout=60, proxy=proxy, cwd=WORKDIR,
            )
            r1 = await agent("Remember: animal=cat. Just say 'OK'.")
            log(f"turn 1: {r1.content}")
            r2 = await agent("What animal? Just the animal.")
            log(f"turn 2: {r2.content}")

            state = agent.state_dict()
            sdk_trace = state['sdk_trace']
            llm_trace = state['llm_trace']

            # SDK: 2 turns → at least 2 ResultMessages
            result_msgs = [e for e in sdk_trace if e['type'] == 'ResultMessage']
            log(f"ResultMessages: {len(result_msgs)}")
            assert len(result_msgs) >= 2
            assert result_msgs[0]['call_index'] == 0
            assert result_msgs[1]['call_index'] == 1

            # Proxy: messages grow across turns
            log(f"llm_trace calls: {len(llm_trace)}")
            assert len(llm_trace) >= 2
            msgs_first = len(llm_trace[0]['request']['messages'])
            msgs_last = len(llm_trace[-1]['request']['messages'])
            log(f"messages: first={msgs_first} last={msgs_last}")
            assert msgs_last > msgs_first

            # Training samples: multi-turn aggregated into one sample
            samples = proxy.to_training_samples(agent.session_id)
            log(f"training samples: {len(samples)}")
            assert len(samples) >= 1

            sample = samples[0]
            log(f"sample messages: {len(sample['messages'])}")
            log(f"sample response blocks: {len(sample['response'])}")
            log(f"sample num_calls: {sample['meta']['num_calls']}")
            log(f"sample total_usage: {sample['meta']['total_usage']}")

            # The sample's messages should be the full conversation
            assert len(sample['messages']) >= msgs_last
            # Should have aggregated usage
            assert sample['meta']['total_usage']['total_input_tokens'] > 0
            assert sample['meta']['total_usage']['total_output_tokens'] > 0
            assert sample['meta']['num_calls'] >= 2
        finally:
            await proxy.stop()

class TestExternalAgentAction:

    @pytest.mark.asyncio
    async def test_action_with_cli(self):
        """ExternalAgentAction wrapping CLI adapter."""
        adapter = ClaudeCodeAdapter(
            max_turns=3, timeout=60, working_dir=WORKDIR,
        )
        action = ExternalAgentAction(adapters={"claude": adapter})

        result = await action(
            '{"agent_name": "claude", "task": "What is 8+9? Just the number."}',
            'run_agent',
        )
        log(f"result: {result.result[0]['content'][:80] if result.result else result.errmsg}")
        assert result.state == 0
        assert '17' in result.result[0]['content']

    @pytest.mark.asyncio
    async def test_action_with_sdk(self):
        """ExternalAgentAction wrapping SDK adapter."""
        adapter = ClaudeCodeSDKAdapter(
            max_turns=3, timeout=60, cwd=WORKDIR,
        )
        action = ExternalAgentAction(adapters={"claude-sdk": adapter})

        result = await action(
            '{"agent_name": "claude-sdk", "task": "What is 3*5? Just the number."}',
            'run_agent',
        )
        log(f"result: {result.result[0]['content'][:80] if result.result else result.errmsg}")
        assert result.state == 0
        assert '15' in result.result[0]['content']

    @pytest.mark.asyncio
    async def test_action_list_agents(self):
        """list_agents returns registered adapters."""
        cli = ClaudeCodeAdapter(max_turns=3, timeout=60, working_dir=WORKDIR)
        sdk = ClaudeCodeSDKAdapter(max_turns=3, timeout=60, cwd=WORKDIR)
        action = ExternalAgentAction(adapters={"cli": cli, "sdk": sdk})

        result = await action('{}', 'list_agents')
        log(f"agents: {result.result[0]['content']}")
        assert result.state == 0
        assert 'cli' in result.result[0]['content']
        assert 'sdk' in result.result[0]['content']


# ── State Dict Comparison ────────────────────────────────────────

class TestStateDictComparison:

    @pytest.mark.asyncio
    async def test_cli_and_sdk_memory_structure_matches(self):
        """Both adapters produce same memory structure."""
        cli_agent = ClaudeCodeAdapter(
            max_turns=3, timeout=60, working_dir=WORKDIR,
        )
        sdk_agent = ClaudeCodeSDKAdapter(
            max_turns=3, timeout=60, cwd=WORKDIR,
        )

        r1 = await cli_agent("Say 'ping'.")
        log(f"CLI: {r1.content}")
        r2 = await sdk_agent("Say 'ping'.")
        log(f"SDK: {r2.content}")

        cli_state = cli_agent.state_dict()
        sdk_state = sdk_agent.state_dict()

        # Both have memory with same structure
        assert 'memory' in cli_state
        assert 'memory' in sdk_state
        log(f"CLI memory: {len(cli_state['memory'])} entries")
        log(f"SDK memory: {len(sdk_state['memory'])} entries")
        assert len(cli_state['memory']) == 2  # input + output
        assert len(sdk_state['memory']) == 2

        # Both have sender fields
        assert cli_state['memory'][0]['sender'] == 'user'
        assert sdk_state['memory'][0]['sender'] == 'user'

    @pytest.mark.asyncio
    async def test_proxy_trace_structure_same_for_both(self):
        """When both use Proxy, llm_trace has same structure."""
        proxy_cli = make_proxy()
        proxy_sdk = make_proxy()
        await proxy_cli.start()
        await proxy_sdk.start()
        try:
            cli_agent = ClaudeCodeAdapter(
                max_turns=3, timeout=60, proxy=proxy_cli, working_dir=WORKDIR,
            )
            sdk_agent = ClaudeCodeSDKAdapter(
                max_turns=3, timeout=60, proxy=proxy_sdk, cwd=WORKDIR,
            )

            r1 = await cli_agent("What is 1+1? Just the number.")
            log(f"CLI: {r1.content}")
            r2 = await sdk_agent("What is 1+1? Just the number.")
            log(f"SDK: {r2.content}")

            cli_trace = cli_agent.state_dict()['llm_trace']
            sdk_trace = sdk_agent.state_dict()['llm_trace']

            # Both have at least 1 record
            log(f"CLI trace: {len(cli_trace)} records")
            log(f"SDK trace: {len(sdk_trace)} records")
            assert len(cli_trace) >= 1
            assert len(sdk_trace) >= 1

            # Same keys in each record
            cli_keys = set(cli_trace[0].keys())
            sdk_keys = set(sdk_trace[0].keys())
            log(f"CLI keys: {sorted(cli_keys)}")
            log(f"SDK keys: {sorted(sdk_keys)}")
            assert cli_keys == sdk_keys

            # Both have request.messages and response.content
            for label, trace in [("CLI", cli_trace), ("SDK", sdk_trace)]:
                assert 'messages' in trace[0]['request']
                assert 'content' in trace[0]['response']
                assert 'usage' in trace[0]['response']
                log(f"{label} usage: {json.dumps(trace[0]['response']['usage'])}")
        finally:
            await proxy_cli.stop()
            await proxy_sdk.stop()


# ── F5 Debug Runner ──────────────────────────────────────────────

async def _run_test(test_cls, method_name):
    """Run a single test method with output."""
    obj = test_cls()
    method = getattr(obj, method_name)
    print(f"\n{'='*60}")
    print(f"  {test_cls.__name__}.{method_name}")
    print(f"  {method.__doc__}")
    print(f"{'='*60}")
    try:
        await method()
        print(f"  ✅ PASSED")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


async def run_all():
    """Run all tests sequentially for F5 debugging."""
    test_classes = [
        TestCLIAdapter,
        TestSDKAdapter,
        TestExternalAgentAction,
        TestStateDictComparison,
    ]
    for cls in test_classes:
        for name in sorted(dir(cls)):
            if name.startswith('test_'):
                await _run_test(cls, name)

    print(f"\n{'='*60}")
    print("  All tests completed!")
    print(f"{'='*60}")


async def run_one(spec: str):
    """Run a single test: 'TestCLIAdapter.test_single_turn'."""
    cls_name, method_name = spec.split('.')
    cls = {
        'TestCLIAdapter': TestCLIAdapter,
        'TestSDKAdapter': TestSDKAdapter,
        'TestExternalAgentAction': TestExternalAgentAction,
        'TestStateDictComparison': TestStateDictComparison,
    }[cls_name]
    await _run_test(cls, method_name)


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     # python test_claude_code_integration.py TestCLIAdapter.test_single_turn
    #     asyncio.run(run_one(sys.argv[1]))
    # else:
    #     # python test_claude_code_integration.py → run all
    #     asyncio.run(run_all())
    asyncio.run(run_one('TestSDKAdapter'))