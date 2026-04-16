"""Integration tests for OpenClaw adapter (CLI + Proxy).

Requires:
  - Node 22+ via nvm
  - openclaw installed globally (npm install -g openclaw)
  - ~/.openclaw/openclaw.json configured with a working provider
  - http proxy at http://100.100.72.89:8899 for outbound access

Run:
    python tests/test_adapters/test_openclaw_integration.py
    python -m pytest tests/test_adapters/test_openclaw_integration.py -v -s
"""
import asyncio
import json
import os
import pathlib
import shutil
import sys

import pytest

from lagent.adapters.openclaw import OpenClawAdapter
from lagent.adapters.proxy import LLMProxyRecorder
from lagent.schema import AgentMessage

NVM_DIR = '/mnt/shared-storage-user/liukuikun/.nvm'
HTTP_PROXY = 'http://100.100.72.89:8899'
API_KEY = ' '
REAL_BASE_URL = 'http://35.220.164.252:3888/v1'
OPENCLAW_HOME = pathlib.Path.home() / '.openclaw'

# Skip if openclaw not set up
pytestmark = pytest.mark.skipif(
    not (OPENCLAW_HOME / 'openclaw.json').exists(),
    reason="OpenClaw not configured (~/.openclaw/openclaw.json missing)",
)


def log(msg):
    print(f"  → {msg}")


def _setup_openclaw_config(proxy_url=None):
    """Write openclaw.json and auth-profiles for testing."""
    base_url = proxy_url or REAL_BASE_URL
    config = {
        'models': {'mode': 'merge', 'providers': {'custom-openai': {
            'baseUrl': base_url,
            'api': 'openai-completions',
            'models': [{'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini',
                        'reasoning': False, 'input': ['text'],
                        'contextWindow': 128000, 'maxTokens': 16384}],
        }}},
        'agents': {'defaults': {'model': {'primary': 'custom-openai/gpt-4o-mini'}}},
    }
    (OPENCLAW_HOME / 'openclaw.json').write_text(json.dumps(config, indent=2))

    key = f'sk-proxy-test' if proxy_url else API_KEY
    auth = {'version': 1, 'profiles': {'default': {
        'type': 'api_key', 'provider': 'custom-openai', 'key': key,
    }}}
    auth_path = OPENCLAW_HOME / 'agents/main/agent/auth-profiles.json'
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(auth))

    # Clear sessions and state
    sd = OPENCLAW_HOME / 'agents/main/sessions'
    if sd.exists():
        shutil.rmtree(sd)
    sd.mkdir(parents=True, exist_ok=True)
    (OPENCLAW_HOME / 'agents/main/agent/auth-state.json').unlink(missing_ok=True)


def _make_agent(**kwargs):
    """Create an OpenClawAdapter with standard test config."""
    defaults = dict(
        thinking='off',
        timeout=120,
        json_output=False,
        nvm_dir=NVM_DIR,
        env_vars={'http_proxy': HTTP_PROXY, 'https_proxy': HTTP_PROXY},
    )
    defaults.update(kwargs)
    return OpenClawAdapter(**defaults)


class TestOpenClawAdapter:

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """Single turn returns correct result."""
        _setup_openclaw_config()
        agent = _make_agent()
        result = await agent("What is 9+3? Just the number.")
        log(f"result: {result.content}")
        assert isinstance(result, AgentMessage)
        assert '12' in result.content

    @pytest.mark.asyncio
    async def test_multiturn(self):
        """Real multi-turn via session-id."""
        _setup_openclaw_config()
        agent = _make_agent()

        r1 = await agent("Remember: fruit=mango. Just say OK.")
        log(f"turn 1: {r1.content}")

        r2 = await agent("What fruit? Just the name.")
        log(f"turn 2: {r2.content}")
        assert 'mango' in r2.content.lower()

    @pytest.mark.asyncio
    async def test_memory_accumulates(self):
        """Memory stores input+output for each turn."""
        _setup_openclaw_config()
        agent = _make_agent()
        await agent("Say 'alpha'.")
        await agent("Say 'beta'.")

        memory = agent.memory.get_memory()
        log(f"memory: {len(memory)} entries")
        assert len(memory) == 4
        assert memory[0].sender == 'user'
        assert memory[1].sender == 'openclaw'

    @pytest.mark.asyncio
    async def test_with_proxy(self):
        """Proxy captures LLM calls from OpenClaw."""
        proxy = LLMProxyRecorder(
            real_api_key=API_KEY,
            real_base_url=REAL_BASE_URL,
            http_proxy=HTTP_PROXY,
        )
        await proxy.start()
        try:
            _setup_openclaw_config(proxy_url=f'{proxy.url}/v1')

            # No http_proxy in env — OpenClaw talks to localhost proxy directly
            agent = _make_agent(env_vars={})
            r1 = await agent("What is 5*6? Just the number.")
            log(f"result: {r1.content}")
            assert '30' in r1.content

            all_records = []
            for recs in proxy._records.values():
                all_records.extend(recs)
            log(f"LLM calls: {len(all_records)}")
            assert len(all_records) >= 1

            # Verify response was parsed
            proxy._records['test'] = all_records
            norm = proxy.get_normalized_records('test')
            for n in norm:
                resp_text = n['response'].get('content', '')
                log(f"  response: '{resp_text[:50]}'")
            assert any(n['response'].get('content', '') for n in norm)
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_proxy_multiturn_training_sample(self):
        """Multi-turn + Proxy produces correct training sample."""
        proxy = LLMProxyRecorder(
            real_api_key=API_KEY,
            real_base_url=REAL_BASE_URL,
            http_proxy=HTTP_PROXY,
        )
        await proxy.start()
        try:
            _setup_openclaw_config(proxy_url=f'{proxy.url}/v1')

            # No http_proxy — OpenClaw talks to localhost proxy directly
            agent = _make_agent(env_vars={})
            r1 = await agent("Remember: planet=Mars. Just say OK.")
            log(f"turn 1: {r1.content}")
            r2 = await agent("What planet? Just the name.")
            log(f"turn 2: {r2.content}")
            assert 'mars' in r2.content.lower()

            all_records = []
            for recs in proxy._records.values():
                all_records.extend(recs)
            log(f"LLM calls: {len(all_records)}")

            proxy._records['test'] = all_records
            chains = proxy.rebuild_chains('test')
            log(f"chains: {len(chains)}")

            samples = proxy.to_training_samples('test')
            log(f"training samples: {len(samples)}")
            assert len(samples) >= 1

            s = samples[0]
            log(f"sample: {len(s['messages'])} msgs, {s['meta']['num_calls']} calls")
            assert s['meta']['num_calls'] >= 1
            assert len(s['messages']) >= 2

            # Messages should have system + user + assistant pattern
            roles = [m['role'] for m in s['messages']]
            log(f"roles: {roles}")
            assert 'assistant' in roles
            assert 'user' in roles
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_state_dict(self):
        """state_dict has memory."""
        _setup_openclaw_config()
        agent = _make_agent()
        await agent("Say hello.")

        state = agent.state_dict()
        log(f"state keys: {list(state.keys())}")
        assert 'memory' in state
        assert len(state['memory']) == 2


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
    for name in sorted(dir(TestOpenClawAdapter)):
        if name.startswith('test_'):
            await _run_test(TestOpenClawAdapter, name)
    print(f"\n{'='*60}\n  Done!\n{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        asyncio.run(_run_test(TestOpenClawAdapter, sys.argv[1]))
    else:
        asyncio.run(run_all())
