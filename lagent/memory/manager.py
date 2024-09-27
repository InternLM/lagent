from typing import Dict

from ..utils import create_object
from .base_memory import Memory


class MemoryManager:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.memory_map: Dict[str, Memory] = {}

    def create_instance(self, session_id):
        self.memory_map[session_id] = create_object(self.cfg)

    def get_memory(self, session_id=0, **kwargs) -> list:
        return self.memory_map[session_id].get_memory(**kwargs)

    def add(self, memory, session_id=0, **kwargs) -> None:
        if session_id not in self.memory_map:
            self.create_instance(session_id)
        self.memory_map[session_id].add(memory, **kwargs)

    def get(self, session_id=0) -> Memory:
        return self.memory_map.get(session_id, None)

    def reset(self, session_id=0) -> None:
        if session_id in self.memory_map:
            del self.memory_map[session_id]
