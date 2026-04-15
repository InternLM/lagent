"""Claude Code memory provider.

Claude Code uses a **directory of markdown files** managed by the
model itself through standard file read/write tools:

  memory/
  ├── MEMORY.md           ← index file (pointers to memory files)
  ├── user_role.md        ← individual memory with frontmatter
  ├── feedback_testing.md
  └── project_context.md

Each memory file has YAML frontmatter::

    ---
    name: user role
    description: user is a senior engineer working on lagent
    type: user
    ---
    (memory content)

``MEMORY.md`` is an index — each entry is one line::

    - [User Role](user_role.md) — senior engineer on lagent project

``get_info()`` reads the index + all referenced files, returning
the assembled content for env_info injection.

``actions`` is empty — Claude Code relies on the model using
standard file read/write tools (guided by prompt instructions)
to manage memory. No special memory actions needed.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List


class ClaudeCodeMemoryProvider:
    """Claude Code style memory: index + individual markdown files.

    The model manages memory files autonomously via read/write tools.
    This provider only handles the **read side** — loading memory
    content into env_info each turn.

    Usage::

        provider = ClaudeCodeMemoryProvider(memory_dir)
        env = AsyncEnvAgent(
            actions=actions,  # no special memory actions needed
            long_term_memory=provider,
        )

    Parameters
    ----------
    memory_dir : Path
        Directory containing MEMORY.md index and memory files.
    """

    def __init__(self, memory_dir: Path):
        self._dir = Path(memory_dir)
        self._index_file = self._dir / "MEMORY.md"

    async def get_info(self) -> dict:
        """Load index + all referenced memory files."""
        index_content = await self._read_index()
        if not index_content:
            return {}

        # Parse referenced files from index
        memory_files = self._parse_index_links(index_content)

        # Load each memory file
        memories = []
        for filename in memory_files:
            filepath = self._dir / filename
            content = await self._read_file(filepath)
            if content:
                memories.append(content)

        return {
            "available": True,
            "index": index_content,
            "memories": memories,
        }

    @property
    def actions(self) -> list:
        """Claude Code doesn't need special memory actions.

        The model uses standard file read/write tools, guided by
        prompt instructions that describe the memory directory format.
        """
        return []

    # ── Internal helpers ──

    async def _read_index(self) -> str:
        if self._index_file.exists():
            return await asyncio.to_thread(
                self._index_file.read_text, encoding="utf-8"
            )
        return ""

    async def _read_file(self, path: Path) -> str:
        if path.exists():
            return await asyncio.to_thread(
                path.read_text, encoding="utf-8"
            )
        return ""

    def _parse_index_links(self, index_content: str) -> List[str]:
        """Extract markdown link targets from index lines.

        Parses lines like:
            - [Title](filename.md) — description
        Returns list of filenames.
        """
        pattern = re.compile(r'\[.*?\]\((.+?\.md)\)')
        filenames = []
        for line in index_content.splitlines():
            match = pattern.search(line)
            if match:
                filenames.append(match.group(1))
        return filenames
