"""Agent Project 入口 — build() 工厂函数。

当 Agent 有运行时依赖（AgentService、CompactAgent 等）时，
纯声明式的 config.py 无法完整表达。此时用 build() 来处理组装逻辑。

AgentLoader 发现 __init__.py 中有 build() 时，会将其存入
AgentSpec.build，由 default_agent_factory 在 spawn 时调用。

签名:
    async def build(spec: AgentSpec, task: str) -> Agent
"""

from lagent.utils import create_object
from lagent.actions.subagent import AgentAction

# 从同目录的 config.py 导入基础配置
from .config import (
    agent_config,
    name,
    description,
    max_turns,
    background,
)


async def build(spec, task):
    """组装完整的 InternClaw Agent，处理运行时依赖注入。

    调用链:
        AgentService.spawn("internclaw-standard", task)
          → default_agent_factory(spec, task)
            → spec.build(spec, task)  ← 就是这个函数

    Parameters
    ----------
    spec : AgentSpec
        由 AgentLoader 从 config.py 解析出的 spec。
        spec.agent_config 就是 config.py 中的 agent_config dict。
    task : str
        用户任务描述。

    Returns
    -------
    InternClawAgent
        完全组装好的 Agent 实例，可以直接 await agent(task)。
    """
    import copy

    # 1. 深拷贝 config，避免污染原始模板
    cfg = copy.deepcopy(spec.agent_config or agent_config)

    # 2. 先创建核心 Agent（不含运行时依赖的 actions）
    agent = create_object(cfg)

    # 3. 注入运行时依赖
    #    这些组件需要 agent 实例或 agent_service 才能创建
    #
    #    示例: 注入 CompactAgent + LongTermMemory
    #
    #    from lagent.agents.compact_agent import AsyncCompactAgent
    #    from lagent.memory.long_term import FilesystemLongTermMemory
    #    from lagent.actions.save_memory import SaveMemoryAction
    #
    #    ltm = FilesystemLongTermMemory(workspace_path)
    #    compact = AsyncCompactAgent(llm=model)
    #    agent.compact_agent = compact
    #    agent.long_term_memory = ltm
    #    agent.env_agent.long_term_memory = ltm
    #
    #    # For consolidation, create a standard InternClawAgent
    #    # with SaveMemoryAction + consolidation prompt
    #    save_action = SaveMemoryAction(ltm)
    #    # ... configure consolidate_agent as InternClawAgent instance

    return agent
