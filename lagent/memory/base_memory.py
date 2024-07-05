from typing import Callable, Dict, List, Optional, Union

from lagent.registry import MEMORY_REGISTRY, AutoRegister, RegistryMeta
from lagent.schema import AgentMessage


class Memory(metaclass=AutoRegister(MEMORY_REGISTRY, RegistryMeta)):

    def __init__(self, recent_n=None) -> None:
        self.memory = []
        self.recent_n = recent_n

    def get_memory(
        self,
        recent_n: Optional[int] = None,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        recent_n = recent_n or self.recent_n
        if recent_n is not None:
            memory = self.memory[-recent_n:]
        else:
            memory = self.memory
        if filter_func is not None:
            memory = [m for i, m in enumerate(memory) if filter_func(i, m)]
        return memory

    def add(self, memories: Union[List[Dict], Dict, None]) -> None:
        if isinstance(memories, AgentMessage):
            self.memory.append(memories)
        elif isinstance(memories, list):
            self.memory.extend(memories)

    def delete(self, index: Union[List, int]) -> None:
        if isinstance(index, int):
            del self.memory[index]
        else:
            for i in index:
                del self.memory[i]

    def load(
        self,
        memories: Union[str, Dict, List],
        overwrite: bool = False,
    ) -> None:
        if overwrite:
            self.memory = []
        if isinstance(memories, dict):
            self.memory.append(memories)
        elif isinstance(memories, list):
            self.memory.extend(memories)
