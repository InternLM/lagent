"""自定义 LongTermMemory 示例。

Memory 在 Lagent 中有清晰的分层:

1. **Agent.memory (Memory)** — 短期记忆，存储当前对话的消息列表。
   这是 Agent 基类自带的，通常不需要自定义。

2. **LongTermMemory** — 长期记忆存储服务（纯数据，不依赖 LLM）。
   由 EnvAgent 持有用于读取（get_info → env_info['memory']），
   由 orchestrator 编排写入（通过 SaveMemoryAction 或直接调用）。

3. **CompactAgent** — 上下文压缩 Agent，token 超阈值时压缩对话。

4. **ConsolidateAgent** — 就是一个配了 consolidation prompt +
   SaveMemoryAction 的 InternClawAgent 实例，不需要单独的类。

内置实现:
  - FilesystemLongTermMemory: 本地 MEMORY.md + HISTORY.md
  - SandboxLongTermMemory:    远程沙箱存储
  - (可扩展) MemoryOS / OpenViking 等外部库包装
"""

from typing import Any

from lagent.memory.long_term import LongTermMemory


class ProjectLongTermMemory(LongTermMemory):
    """示例：带项目上下文的 LongTermMemory。

    在标准的持久化记忆之外，额外注入一个 "项目知识" 摘要，
    让 Agent 在长对话中始终记得项目的关键信息。
    """

    def __init__(
        self,
        backend_ltm: LongTermMemory,
        project_context: str = "",
    ):
        self._inner = backend_ltm
        self._project_context = project_context

    async def get_info(self) -> dict[str, Any]:
        """返回注入 system prompt 的记忆信息。"""
        info = await self._inner.get_info()
        if self._project_context:
            info["project_context"] = self._project_context
        return info

    async def read(self) -> str:
        return await self._inner.read()

    async def write(self, content: str) -> None:
        await self._inner.write(content)

    async def append_history(self, entry: str) -> None:
        await self._inner.append_history(entry)
