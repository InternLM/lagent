"""Tests for lagent.memory.claude_code_provider"""

import asyncio
import tempfile
from pathlib import Path

from lagent.memory.claude_code_provider import ClaudeCodeMemoryProvider


async def test_get_info_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = ClaudeCodeMemoryProvider(Path(tmpdir))
        info = await provider.get_info()
        assert info == {}


async def test_get_info_with_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "user_role.md").write_text(
            "---\nname: user role\ntype: user\n---\nSenior engineer.\n"
        )
        (d / "feedback.md").write_text(
            "---\nname: feedback\ntype: feedback\n---\nNo mocks.\n"
        )
        (d / "MEMORY.md").write_text(
            "- [Role](user_role.md) — engineer\n"
            "- [Feedback](feedback.md) — no mocks\n"
        )

        provider = ClaudeCodeMemoryProvider(d)
        info = await provider.get_info()
        assert info['available'] is True
        assert len(info['memories']) == 2
        assert 'Senior engineer' in info['memories'][0]
        assert 'No mocks' in info['memories'][1]


async def test_broken_link_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "exists.md").write_text("Content here.\n")
        (d / "MEMORY.md").write_text(
            "- [Exists](exists.md) — ok\n"
            "- [Missing](gone.md) — not found\n"
        )

        provider = ClaudeCodeMemoryProvider(d)
        info = await provider.get_info()
        assert len(info['memories']) == 1


async def test_no_actions():
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = ClaudeCodeMemoryProvider(Path(tmpdir))
        assert provider.actions == []


async def test_parse_various_link_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "a.md").write_text("A")
        (d / "b.md").write_text("B")
        (d / "MEMORY.md").write_text(
            "- [Title with spaces](a.md) — description\n"
            "  - [Indented](b.md) — also works\n"
            "some text without links\n"
        )

        provider = ClaudeCodeMemoryProvider(d)
        info = await provider.get_info()
        assert len(info['memories']) == 2


async def main():
    tests = [
        test_get_info_empty,
        test_get_info_with_files,
        test_broken_link_skipped,
        test_no_actions,
        test_parse_various_link_formats,
    ]
    for test in tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
