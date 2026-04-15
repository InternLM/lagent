"""Tests for InternClawAgent initialization and provider integration."""

import asyncio
import tempfile
from pathlib import Path

from lagent.agents.internclaw_agent import AsyncEnvAgent, AsyncPolicyAgent, InternClawAgent
from lagent.agents.compact_agent import AsyncCompactAgent
from lagent.memory.openclaw_provider import OpenClawMemoryProvider
from lagent.memory.claude_code_provider import ClaudeCodeMemoryProvider


class MockLLM:
    async def chat(self, messages, **kwargs):
        return {"content": "mock response"}


async def test_internclaw_with_compact():
    llm = MockLLM()
    agent = InternClawAgent(
        policy_agent=AsyncPolicyAgent(llm=llm),
        env_agent=AsyncEnvAgent(actions=[]),
        compact_agent=AsyncCompactAgent(llm=llm),
        consolidate_agent=None,
        max_turn=10,
    )
    assert agent.compact_agent is not None
    assert agent.consolidate_agent is None


async def test_internclaw_without_compact():
    """Claude Code style — no compact, no consolidate."""
    llm = MockLLM()
    agent = InternClawAgent(
        policy_agent=AsyncPolicyAgent(llm=llm),
        env_agent=AsyncEnvAgent(actions=[]),
        max_turn=10,
    )
    assert agent.compact_agent is None
    assert agent.consolidate_agent is None


async def test_env_with_openclaw_provider():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        mem_dir = workspace / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("# User likes Python")

        provider = OpenClawMemoryProvider(workspace)
        env = AsyncEnvAgent(actions=[], long_term_memory=provider)
        info = await env.get_env_info()
        assert 'Python' in info['memory']['long_term']


async def test_env_with_claude_code_provider():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "MEMORY.md").write_text("- [Pref](pref.md) — likes Rust\n")
        (d / "pref.md").write_text("User prefers Rust.\n")

        provider = ClaudeCodeMemoryProvider(d)
        env = AsyncEnvAgent(actions=[], long_term_memory=provider)
        info = await env.get_env_info()
        assert info['memory']['available'] is True
        assert 'Rust' in info['memory']['memories'][0]


async def test_env_without_provider():
    env = AsyncEnvAgent(actions=[])
    info = await env.get_env_info()
    assert info['memory'] == ''


async def main():
    tests = [
        test_internclaw_with_compact,
        test_internclaw_without_compact,
        test_env_with_openclaw_provider,
        test_env_with_claude_code_provider,
        test_env_without_provider,
    ]
    for test in tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
