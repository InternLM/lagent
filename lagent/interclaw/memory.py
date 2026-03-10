"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger



_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]

import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path



class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        session,
        provider,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.memory
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.memory))
        else:
            keep_count = memory_window // 2
            if len(session.memory) <= keep_count:
                return True
            if len(session.memory) - session.recent_n <= 0:
                return True
            old_messages = session.memory[session.recent_n:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.content:
                continue
            tools = f" [tools: {', '.join(m.tool_calls)}]" if m.tool_calls else ""
            lines.append(f"[{m.timestamp[:16]}] {m.role.upper()}{tools}: {m.content}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
            )

            if 'tool_calls' not in response or not response['tool_calls']:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response['tool_calls'][0]['function']['arguments']
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.recent_n = 0 if archive_all else len(session.memory) - keep_count
            logger.info("Memory consolidation done: {} messages, recent_n={}", len(session.memory), session.recent_n)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False


if __name__ == "__main__":
    from lagent.interclaw.model import AsyncAPIClient, ModelConfig, SampleParameters
    from lagent.memory.manager import Memory
    from lagent.schema import AgentMessage
    model_name = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1_mini_official/interns1_1_mini_sft_based_cpt_bs512_epoch1_maxlr3e-5_minlr1e-6_max16k-hf/20260207101512/hf-4374"
    # model_name = "gpt-4o-2024-08-06"
    api_base = "http://10.102.218.26:23333/v1/"
    # api_base = f"http://35.220.164.252:3888/v1beta/models/{model_name}:generateContent"
    api_key = "sk-blAvnaExZFrQfHVuyIF5VEB3I0GrQ7FNhdAobU3pKpfLvxLb"
    extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    model = AsyncAPIClient(
            model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key),
            sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
            timeout=600,
            max_retry=5,
            sleep_interval=5,
            extra_body=extra_body,
    )
    # Quick test
    store = MemoryStore(Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace"))
    session = Memory(recent_n=0)
    session.add(
        [
            AgentMessage(sender="user", content="What is the weather today?", role="user"),
            AgentMessage(sender="agent", content="The weather is sunny.", role="assistant"),
            AgentMessage(sender="user", content="What about tomorrow?", role="user"),
            AgentMessage(sender="agent", content="Tomorrow will be cloudy.", role="assistant"),
            AgentMessage(sender="user", content="Any plans for the weekend?", role="user"),
            AgentMessage(sender="agent", content="I am planning to go hiking.", role="assistant"),
        ]
    )

    import asyncio
    asyncio.run(store.consolidate(session, model, archive_all=False, memory_window=5))
    
