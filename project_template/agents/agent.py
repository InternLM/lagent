"""自定义 Agent 子类示例。

大多数情况下不需要自定义 Agent 子类 — 直接用 InternClawAgent 配合
不同的 config 就够了。只有在需要修改核心循环逻辑时才继承。

典型场景:
  - 自定义 finish_condition
  - 在 Policy-Env 循环中插入额外逻辑（如 reward 计算）
  - 修改 forward() 的消息路由方式
"""

from typing import Optional

from lagent.agents.internclaw_agent import InternClawAgent
from lagent.schema import AgentMessage


class MyAgent(InternClawAgent):
    """示例：带自定义终止条件的 InternClaw Agent。"""

    def __init__(
        self,
        *args,
        stop_words: Optional[list] = None,
        **kwargs,
    ):
        # 覆盖 finish_condition
        kwargs.setdefault("finish_condition", self._should_stop)
        super().__init__(*args, **kwargs)
        self.stop_words = stop_words or ["TASK_COMPLETE"]

    def _should_stop(self, policy_msg, env_msg) -> bool:
        """当 Policy 不再调用工具，或输出包含 stop word 时终止。"""
        if policy_msg is None:
            return False
        # 无工具调用 → 结束
        if not policy_msg.tool_calls:
            return True
        # 包含 stop word → 结束
        content = policy_msg.content or ""
        return any(w in content for w in self.stop_words)
