"""OpenClaw memory provider.

Reads memory context from OpenClaw's MEMORY.md for env_info injection.
Write operations are handled by ``SaveMemoryAction`` (separate action).
Both operate on the same workspace directory independently.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class OpenClawMemoryProvider:
    """OpenClaw memory reader: MEMORY.md on filesystem.

    Usage::

        provider = OpenClawMemoryProvider(workspace_path)
        save_action = SaveMemoryAction(workspace_path)

        env = AsyncEnvAgent(
            actions=[save_action, ...],
            long_term_memory=provider,
        )
    """

    def __init__(self, workspace: Path):
        memory_dir = Path(workspace) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = memory_dir / "MEMORY.md"

    async def get_info(self) -> dict:
        content = await self._read()
        if not content:
            return {}
        return {"available": True, "long_term": content}

    async def _read(self) -> str:
        if self._memory_file.exists():
            return await asyncio.to_thread(
                self._memory_file.read_text, encoding="utf-8"
            )
        return ""


class SandboxOpenClawMemoryProvider:
    """OpenClaw memory reader on a remote sandbox."""

    def __init__(self, action: Any, *, workspace_root: str = "."):
        self._shell = action
        root = workspace_root.rstrip("/") or "."
        self._memory_file = f"{root}/memory/MEMORY.md"

    async def get_info(self) -> dict:
        content = await self._read()
        if not content:
            return {}
        return {"available": True, "long_term": content}

    async def _read(self) -> str:
        from lagent.schema import ActionStatusCode
        result = await self._shell.run(
            command=f"cat {self._memory_file!r} 2>/dev/null || true"
        )
        if result.state != ActionStatusCode.SUCCESS:
            return ""
        try:
            if isinstance(result.result, list) and result.result:
                content_str = result.result[0].get("content", "")
                content_dict = json.loads(content_str)
                if content_dict.get("exit_code") == 0:
                    return content_dict.get("stdout", "").strip()
        except Exception:
            pass
        return ""
