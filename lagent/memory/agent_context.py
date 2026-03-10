import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field

# 假设这里引入了 Lagent 的 AgentMessage
from lagent.schema import AgentMessage


# -------------------------------------------------------------------
# 1. 结构化 Payload (给大模型吃的标准格式数据)
# -------------------------------------------------------------------
class ContextPayload(BaseModel):
    """
    负责彻底去字符串化（不再用 string template）。
    这就是最终送交 policy 的 Structured Prompt Data。
    """
    system_instruction: str = ""
    core_memory: str = ""    # 比如 Memory.md 的内容
    history: List[AgentMessage] = Field(default_factory=list) # 当前 Session 的消息流
    # 这里未来还能加上 tools_schema, skills 等等


# -------------------------------------------------------------------
# 2. SessionMemory (RAM / State，带脱水和注水能力)
# -------------------------------------------------------------------
class SessionMemory:
    """
    代表 Agent 的短期记忆和当前状态。
    它负责记录对话流水，以及维护一个极其关键的水位线：last_consolidated。
    """
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.messages: List[AgentMessage] = []
        self.last_consolidated_idx: int = 0  # 关键元数据：游标水位线

    def add(self, message: AgentMessage):
        self.messages.append(message)

    def get_unconsolidated(self) -> List[AgentMessage]:
        return self.messages[self.last_consolidated_idx:]

    def mark_consolidated(self, to_index: int):
        """移动游标，表示这些信息已被 MemoryOS 消费（无论有没有被压缩）"""
        self.last_consolidated_idx = to_index

    # --- 解决脱水（Dehydration）与 注水（Hydration） ---
    
    def state_dict(self) -> dict:
        """从内存中提取纯状态（脱水）"""
        return {
            "session_id": self.session_id,
            "last_consolidated_idx": self.last_consolidated_idx,
            "messages": [m.model_dump() for m in self.messages]
        }
        
    def load_state_dict(self, state: dict):
        """进程重启时恢复状态（注水）"""
        self.session_id = state.get("session_id", self.session_id)
        self.last_consolidated_idx = state.get("last_consolidated_idx", 0)
        self.messages = [AgentMessage(**m) for m in state.get("messages", [])]


# -------------------------------------------------------------------
# 3. MemoryOS (长期记忆门面 / Environment)
# -------------------------------------------------------------------
class MemoryOS:
    """
    完全作为外部环境（Environment）运作。
    主要职责：
    1. 管理核心知识 (Core Memory / Memory.md) —— 这个会每次被塞进 Context
    2. 管理归档历史 (Archival Memory / History.md) —— 这个只能被 Tool 查询
    """
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.core_memory_file = os.path.join(workspace_dir, "MEMORY.md")
        # 可以有更多比如 HISTORY.jsonl, SKILLS.toml
        
        # 确保目录存在
        os.makedirs(workspace_dir, exist_ok=True)
        if not os.path.exists(self.core_memory_file):
            with open(self.core_memory_file, 'w') as f:
                f.write("# Core Memory\n")

    def get_core_memory(self) -> str:
        """供 Context 提取，拼装进 System Prompt"""
        with open(self.core_memory_file, 'r') as f:
            return f.read()

    def update_core_memory(self, new_summary: str):
        """这是工具（Internal Action）可以调用的入口，用于 Agent 改变自身长期属性"""
        with open(self.core_memory_file, 'a') as f:
            f.write(f"\n{new_summary}")


# -------------------------------------------------------------------
# 4. AgentContext (总线与同步屏障 / Facade)
# -------------------------------------------------------------------
class AgentContext:
    """
    这才是真正在 Agent.__init__ 和 Agent.forward 中流转的超集包裹。
    它是 Session (State) 和 MemoryOS (Env) 的结合体。
    """
    def __init__(self, session_id: str, memory_os_dir: str):
        # 内部状态
        self.session = SessionMemory(session_id)
        # 外部环境
        self.memory_os = MemoryOS(memory_os_dir)
        
        # 这里进行第三个边界的“冷启动”处理
        self._hydrate_session()

    def _get_session_backup_path(self):
        # 临时用本地文件模拟 redis 状态存储
        return os.path.join(self.memory_os.workspace_dir, f"session_{self.session.session_id}.json")

    def _hydrate_session(self):
        """进程启动时，通过会话 ID 寻找之前的 State 并复活"""
        path = self._get_session_backup_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
                self.session.load_state_dict(state)

    def _dehydrate_session(self):
        """
        每次执行完一轮，保存当前纯状态（这应该被加入到 after_agent Hook 中） 
        """
        path = self._get_session_backup_path()
        with open(path, 'w') as f:
            json.dump(self.session.state_dict(), f, ensure_ascii=False, indent=2)

    def build_payload(self) -> ContextPayload:
        """
        Aggregator 的职责：不丢信息，完美组装
        这层可以做截断报警：如果 session 长度过大，截断时必须要校验 last_consolidated 游标！
        """
        core_mem = self.memory_os.get_core_memory()
        
        # 这里你可以放置针对 session.messages 的滑动窗口或截断逻辑
        # 比如：只截取最近 10 条
        # 但截取前可以检测：是否抛弃了尚未 consolidate 的信息？
        
        return ContextPayload(
            core_memory=core_mem,
            history=self.session.messages, # TODO: 截断逻辑
        )

    # ---------------- 业务中转站 ----------------
    def push_message(self, msg: AgentMessage):
        self.session.add(msg)
        self._dehydrate_session() # 每次落盘/或者通过 Hook 处理

