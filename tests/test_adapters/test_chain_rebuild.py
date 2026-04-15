"""Test rebuild_chains and to_training_samples with complex scenarios.

Simulates sub-agents, context compression, and mixed cases using
mock proxy records — no real LLM calls needed.

Run:
    python tests/test_adapters/test_chain_rebuild.py
    python -m pytest tests/test_adapters/test_chain_rebuild.py -v -s
"""
import asyncio
import sys
from lagent.adapters.proxy import LLMProxyRecorder


def make_chain_records(turns, model="claude-opus-4-6", system="default_system",
                       prefix="main", input_tokens=100, output_tokens=50):
    """Build a realistic chain of LLM call records.

    Each turn adds a user message and the previous assistant response
    to the messages list, mimicking real multi-turn behavior where
    each request carries the full conversation history.

    Args:
        turns: List of (user_text, assistant_response) tuples.
        model: Model name for all records.
        system: System prompt for all records.
        prefix: Prefix for timestamps.

    Returns:
        List of record dicts.
    """
    records = []
    messages = []
    for i, (user_text, response_text) in enumerate(turns):
        messages.append({"role": "user", "content": user_text})
        record = {
            "timestamp": f"2026-01-01T{prefix}:{i:02d}",
            "request": {
                "model": model,
                "system": system,
                "messages": list(messages),  # copy
                "tools": [{"name": "tool1"}],
            },
            "response": {
                "content": [{"type": "text", "text": response_text}],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "model": model,
            },
            "path": "/v1/messages",
            "method": "POST",
            "stream": False,
        }
        records.append(record)
        messages.append({"role": "assistant", "content": response_text})
    return records


class TestRebuildChains:
    """Test chain rebuilding with various scenarios."""

    def setup_method(self):
        self.proxy = LLMProxyRecorder.__new__(LLMProxyRecorder)
        self.proxy._records = {}

    def _inject(self, session_id, records):
        self.proxy._records[session_id] = records

    def test_simple_multiturn(self):
        """Simple case: 3-turn conversation → 1 chain."""
        print("\n--- Simple multi-turn (3 turns) ---")
        records = make_chain_records([
            ("Hello", "Hi there!"),
            ("What is 2+2?", "4"),
            ("Thanks", "You're welcome!"),
        ])
        self._inject("s1", records)

        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        for i, c in enumerate(chains):
            print(f"  Chain {i+1}: {len(c)} records, msgs={[len(r['messages']) for r in c]}")
        assert len(chains) == 1
        assert len(chains[0]) == 3

        samples = self.proxy.to_training_samples("s1")
        print(f"Samples: {len(samples)}, messages in sample: {len(samples[0]['messages'])}")
        assert len(samples) == 1
        assert len(samples[0]['messages']) == 7  # system + 3 user + 3 assistant
        assert samples[0]['meta']['num_calls'] == 3

    def test_context_compression(self):
        """Context compression: history grows, then resets with summary."""
        print("\n--- Context compression ---")
        pre = make_chain_records([
            ("Hello", "Hi!"),
            ("Tell me about Python", "Python is great."),
            ("More details", "It has many libraries."),
        ], prefix="pre")

        # Post-compression: fresh start with summary (no previous response)
        post = make_chain_records([
            ("Summary: we discussed Python. Now explain decorators.", "Decorators wrap functions."),
            ("Example?", "@decorator syntax."),
        ], prefix="post")

        self._inject("s1", pre + post)

        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        for i, c in enumerate(chains):
            print(f"  Chain {i+1}: {len(c)} records, msgs={[len(r['messages']) for r in c]}")
        assert len(chains) == 2
        assert len(chains[0]) == 3  # pre-compression
        assert len(chains[1]) == 2  # post-compression

        samples = self.proxy.to_training_samples("s1")
        print(f"Samples: {len(samples)}")
        assert len(samples) == 2

    def test_subagent_different_model(self):
        """Sub-agent uses a different model → separate chain."""
        print("\n--- Sub-agent (different model) ---")
        main1 = make_chain_records([
            ("Fix the bug", "Let me look at the code."),
            ("Found it?", "Yes, line 42."),
        ], model="opus", prefix="m1")

        sub = make_chain_records([
            ("Search for similar bugs", "Found 3 results."),
        ], model="haiku", prefix="sub")

        main2 = make_chain_records([
            ("Apply the fix from the search", "Done, fixed."),
        ], model="opus", prefix="m2")

        self._inject("s1", main1 + sub + main2)

        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        for i, c in enumerate(chains):
            models = set(r['meta']['model'] for r in c)
            print(f"  Chain {i+1}: {len(c)} records, model={models}")
        assert len(chains) == 3

    def test_subagent_same_model(self):
        """Sub-agent uses same model — detected by response discontinuity."""
        print("\n--- Sub-agent (same model) ---")
        main = make_chain_records([
            ("Main task", "Working on it."),
            ("Continue", "Need to search first."),
        ], model="opus", prefix="main")

        # Sub-agent: fresh messages, previous response NOT in history
        sub = make_chain_records([
            ("Sub-task: search code", "Found relevant code."),
            ("Details?", "Function foo() at line 10."),
        ], model="opus", prefix="sub")

        self._inject("s1", main + sub)

        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        for i, c in enumerate(chains):
            print(f"  Chain {i+1}: {len(c)} records, msgs={[len(r['messages']) for r in c]}")
        assert len(chains) == 2
        assert len(chains[0]) == 2  # main
        assert len(chains[1]) == 2  # sub

    def test_compression_plus_subagent(self):
        """Complex: main agent compresses, then spawns sub-agent."""
        print("\n--- Compression + Sub-agent (same model) ---")

        main_pre = make_chain_records([
            ("Fix bug", "Looking..."),
            ("Status?", "Found issue in auth.py."),
            ("Fix it", "Applying patch..."),
        ], model="opus", prefix="pre")

        main_post = make_chain_records([
            ("Summary: fixed auth bug. Now add tests.", "Writing tests..."),
        ], model="opus", prefix="post")

        sub = make_chain_records([
            ("Find test examples", "Found 5 examples."),
            ("Show best one", "test_auth.py is the best."),
        ], model="opus", prefix="sub")

        main_resume = make_chain_records([
            ("Write test based on example", "Test written."),
            ("Run tests", "All 12 tests passed."),
        ], model="opus", prefix="resume")

        self._inject("s1", main_pre + main_post + sub + main_resume)

        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        for i, c in enumerate(chains):
            print(f"  Chain {i+1}: {len(c)} records, msgs={[len(r['messages']) for r in c]}")

        # main_pre(3) → main_post(1): response "Applying patch..." not in post → break
        # main_post(1) → sub(1): response "Writing tests..." not in sub → break
        # sub(2) → main_resume(1): response "test_auth.py..." not in resume → break
        assert len(chains) == 4
        assert len(chains[0]) == 3
        assert len(chains[1]) == 1
        assert len(chains[2]) == 2
        assert len(chains[3]) == 2

    def test_single_call(self):
        """Edge case: only one LLM call."""
        print("\n--- Single call ---")
        records = make_chain_records([("Hello", "Hi")])
        self._inject("s1", records)

        chains = self.proxy.rebuild_chains("s1")
        assert len(chains) == 1
        samples = self.proxy.to_training_samples("s1")
        assert len(samples) == 1
        assert samples[0]['meta']['num_calls'] == 1
        print(f"Chains: {len(chains)}, Samples: {len(samples)} ✓")

    def test_empty_session(self):
        """Edge case: no records."""
        print("\n--- Empty session ---")
        chains = self.proxy.rebuild_chains("nonexistent")
        assert len(chains) == 0
        samples = self.proxy.to_training_samples("nonexistent")
        assert len(samples) == 0
        print(f"Chains: {len(chains)}, Samples: {len(samples)} ✓")

    def test_usage_aggregation(self):
        """Verify training sample format: messages with meta + overall meta."""
        print("\n--- Training sample format ---")
        records = make_chain_records(
            [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")],
            input_tokens=100, output_tokens=20,
        )
        self._inject("s1", records)

        samples = self.proxy.to_training_samples("s1")
        assert len(samples) == 1
        s = samples[0]

        # Top-level structure
        assert 'messages' in s
        assert 'meta' in s
        print(f"messages: {len(s['messages'])}")
        print(f"meta keys: {sorted(s['meta'].keys())}")

        # Messages: system + 3 user + 3 assistant (including final response)
        msgs = s['messages']
        roles = [m['role'] for m in msgs]
        print(f"roles: {roles}")
        # system, user, assistant(A1), user, assistant(A2), user, assistant(A3)
        assert roles[0] == 'system'
        assert roles[-1] == 'assistant'

        # Assistant messages should have extra_info
        asst_msgs = [m for m in msgs if m['role'] == 'assistant']
        print(f"assistant messages: {len(asst_msgs)}")
        for i, m in enumerate(asst_msgs):
            has_extra = 'extra_info' in m
            print(f"  asst[{i}]: content='{m['content'][:30]}', has_extra_info={has_extra}")

        # The last assistant message (from final response) should have extra_info
        assert 'extra_info' in asst_msgs[-1]

        # Overall meta
        meta = s['meta']
        print(f"num_calls: {meta['num_calls']}")
        print(f"total_usage: {meta['total_usage']}")
        assert meta['num_calls'] == 3
        assert meta['total_usage']['total_input_tokens'] == 300
        assert meta['total_usage']['total_output_tokens'] == 60
        print("✓")

    def test_openai_response_format(self):
        """Chain detection works with OpenAI response format (choices)."""
        print("\n--- OpenAI response format ---")
        records = []
        messages = []
        for i, (user, asst) in enumerate([("Hi", "Hello!"), ("How?", "Fine!")]):
            messages.append({"role": "user", "content": user})
            records.append({
                "timestamp": f"T{i}",
                "request": {"model": "gpt-4o", "messages": list(messages)},
                "response": {
                    "choices": [{"message": {"content": asst}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "path": "/v1/chat/completions",
                "method": "POST",
                "stream": False,
            })
            messages.append({"role": "assistant", "content": asst})

        self._inject("s1", records)
        chains = self.proxy.rebuild_chains("s1")
        print(f"Chains: {len(chains)}")
        assert len(chains) == 1
        assert len(chains[0]) == 2
        print("✓")


# ── F5 Runner ────────────────────────────────────────────────────

async def _run_test(test_cls, method_name):
    obj = test_cls()
    obj.setup_method()
    method = getattr(obj, method_name)
    print(f"\n{'='*60}")
    print(f"  {test_cls.__name__}.{method_name}")
    print(f"{'='*60}")
    try:
        method()
        print(f"  ✅ PASSED")
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


async def run_all():
    for name in sorted(dir(TestRebuildChains)):
        if name.startswith('test_'):
            await _run_test(TestRebuildChains, name)
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        asyncio.run(_run_test(TestRebuildChains, sys.argv[1]))
    else:
        asyncio.run(run_all())
