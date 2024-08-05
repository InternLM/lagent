from typing import Dict

from lagent.registry import MEMORY_REGISTRY, ObjectFactory
from .base_memory import Memory


class MemoryManager:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.memory_map: Dict[str, Memory] = {
        }  # Maps client IDs to Memory instances

    def create_instance(self, session_id):
        self.memory_map[session_id] = ObjectFactory.create(
            self.cfg, MEMORY_REGISTRY)

    def get_memory(self, session_id=0, **kwargs) -> list:
        return self.memory_map[session_id].get_memory(**kwargs)

    def add(self, session_id=0, **kwargs) -> None:
        self.memory_map.get(
            session_id,
            self.create_instance(session_id=session_id)).add(**kwargs)

    def get(self, session_id=0) -> Memory:
        return self.memory_map.get(session_id, None)
