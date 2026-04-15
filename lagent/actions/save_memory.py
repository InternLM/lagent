"""SaveMemoryAction — OpenClaw memory write action.

Writes to MEMORY.md + HISTORY.md in the workspace memory directory.
Operates independently from ``OpenClawMemoryProvider`` (which reads).
Both are initialized with the same workspace path by the caller.

Usage::

    provider = OpenClawMemoryProvider(workspace_path)
    save_action = SaveMemoryAction(workspace_path)

    env = AsyncEnvAgent(
        actions=[save_action, ...],
        long_term_memory=provider,
    )
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class SaveMemoryAction(BaseAction):
    """Write consolidated memory to OpenClaw's MEMORY.md + HISTORY.md.

    Parameters
    ----------
    workspace : Path
        Workspace directory containing the ``memory/`` subdirectory.
    """

    def __init__(
        self,
        workspace: Path,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        memory_dir = Path(workspace) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = memory_dir / "MEMORY.md"
        self._history_file = memory_dir / "HISTORY.md"

    async def _save(self, history_entry: str, memory_update: str) -> None:
        if memory_update:
            await asyncio.to_thread(
                self._memory_file.write_text, memory_update, encoding="utf-8"
            )
        if history_entry:
            def _append():
                with open(self._history_file, "a", encoding="utf-8") as f:
                    f.write(history_entry.rstrip() + "\n\n")
            await asyncio.to_thread(_append)

    @tool_api
    def run(
        self,
        history_entry: str = '',
        memory_update: str = '',
    ) -> ActionReturn:
        """Save memory consolidation result to persistent storage.

        Args:
            history_entry: A paragraph (2-5 sentences) summarizing key
                events/decisions/topics. Start with [YYYY-MM-DD HH:MM].
                Include detail useful for grep search.
            memory_update: Full updated long-term memory as markdown.
                Include all existing facts plus new ones. Return
                unchanged if nothing new.

        Returns:
            ActionReturn with confirmation message.
        """
        try:
            asyncio.run(self._save(history_entry, memory_update))
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content="Memory saved successfully.")],
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to save memory: {exc}",
                state=ActionStatusCode.API_ERROR,
            )


class AsyncSaveMemoryAction(AsyncActionMixin, SaveMemoryAction):
    """Async version of SaveMemoryAction."""

    @tool_api
    async def run(
        self,
        history_entry: str = '',
        memory_update: str = '',
    ) -> ActionReturn:
        """Save memory consolidation result to persistent storage.

        Args:
            history_entry: A paragraph (2-5 sentences) summarizing key
                events/decisions/topics. Start with [YYYY-MM-DD HH:MM].
                Include detail useful for grep search.
            memory_update: Full updated long-term memory as markdown.
                Include all existing facts plus new ones. Return
                unchanged if nothing new.

        Returns:
            ActionReturn with confirmation message.
        """
        try:
            await self._save(history_entry, memory_update)
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content="Memory saved successfully.")],
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to save memory: {exc}",
                state=ActionStatusCode.API_ERROR,
            )
