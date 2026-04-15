"""Tests for lagent.memory.openclaw_provider + lagent.actions.save_memory"""

import asyncio
import tempfile
from pathlib import Path

from lagent.memory.openclaw_provider import OpenClawMemoryProvider
from lagent.actions.save_memory import SaveMemoryAction


async def test_get_info_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = OpenClawMemoryProvider(Path(tmpdir))
        info = await provider.get_info()
        assert info == {}


async def test_save_then_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        provider = OpenClawMemoryProvider(workspace)
        action = SaveMemoryAction(workspace)

        await action.save_memory(
            history_entry="[2026-04-08 10:00] User discussed refactoring",
            memory_update="# Facts\n- Refactoring lagent memory",
        )

        info = await provider.get_info()
        assert info['available'] is True
        assert 'Refactoring' in info['long_term']


async def test_history_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        action = SaveMemoryAction(workspace)

        await action.save_memory(history_entry="[2026-04-08] First entry")
        await action.save_memory(history_entry="[2026-04-08] Second entry")

        history = (workspace / "memory" / "HISTORY.md").read_text()
        assert "First entry" in history
        assert "Second entry" in history


async def test_memory_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        provider = OpenClawMemoryProvider(workspace)
        action = SaveMemoryAction(workspace)

        await action.save_memory(memory_update="version 1")
        await action.save_memory(memory_update="version 2")

        info = await provider.get_info()
        assert 'version 2' in info['long_term']
        assert 'version 1' not in info['long_term']


async def test_provider_and_action_independent():
    """Provider and action don't reference each other."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        provider = OpenClawMemoryProvider(workspace)
        action = SaveMemoryAction(workspace)

        # No shared state
        assert not hasattr(provider, '_action')
        assert not hasattr(action, '_provider')

        # But they operate on the same storage
        await action.save_memory(memory_update="shared data")
        info = await provider.get_info()
        assert 'shared data' in info['long_term']


async def main():
    tests = [
        test_get_info_empty,
        test_save_then_read,
        test_history_append,
        test_memory_overwrite,
        test_provider_and_action_independent,
    ]
    for test in tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
