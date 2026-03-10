import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("lagent.interclaw.session")

class SessionManager:
    """
    真实落地的会话管理器，提供物理隔离与硬盘持久化。
    参考 nanobot 的 session 实现，使用 JSON 保存。
    """
    def __init__(self, data_dir: str = ".interclaw_data/sessions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_file_path(self, session_id: str) -> Path:
        # 清理可能引起路径风险的符号
        safe_name = session_id.replace(":", "_").replace("/", "_")
        return self.data_dir / f"{safe_name}.json"

    def load_state(self, session_id: str) -> Dict[str, Any]:
        """加载时优先查缓存，再读硬盘"""
        if session_id in self._cache:
            return self._cache[session_id]

        file_path = self._get_file_path(session_id)
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self._cache[session_id] = state
                    return state
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode session file {file_path}: {e}")
                return {}
        return {}

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """更新缓存并持久化到硬盘"""
        self._cache[session_id] = state
        file_path = self._get_file_path(session_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Session [{session_id}] persisted to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
