from typing import Callable, Dict, List, Optional, Union

from lagent.schema import AgentMessage


class Memory:

    def __init__(self, recent_n=None) -> None:
        self.memory: List[AgentMessage] = []
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
        for memory in memories if isinstance(memories,
                                             (list, tuple)) else [memories]:
            if isinstance(memory, str):
                memory = AgentMessage(sender='user', content=memory)
            if isinstance(memory, AgentMessage):
                self.memory.append(memory)

    def delete(self, index: Union[List, int]) -> None:
        if isinstance(index, int):
            del self.memory[index]
        else:
            for i in index:
                del self.memory[i]

    def load(
        self,
        memories: Union[str, Dict, List],
        overwrite: bool = True,
    ) -> None:
        if overwrite:
            self.memory = []
        if isinstance(memories, dict):
            self.memory.append(AgentMessage(**memories))
        elif isinstance(memories, list):
            for m in memories:
                self.memory.append(AgentMessage(**m))
        else:
            raise TypeError(f'{type(memories)} is not supported')

    def save(self) -> List[dict]:
        memory = []
        for m in self.memory:
            memory.append(m.model_dump())
        return memory
