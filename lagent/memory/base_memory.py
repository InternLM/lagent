from typing import Callable, Dict, List, Optional, Union

from lagent.schema import AgentMessage


class Memory:
    """Session message buffer.  The only memory primitive in lagent.

    A simple append-only list of AgentMessage.  No windowing, no
    boundary tracking — those concerns belong to ContextBuilder
    (which reads compact state from env_info).
    """

    _item_cls = AgentMessage

    def __init__(self) -> None:
        self.memory: List[AgentMessage] = []

    def reset(self) -> None:
        """Clear all messages."""
        self.memory = []

    def get_memory(
        self,
        filter_func: Optional[Callable[[int, AgentMessage], bool]] = None,
    ) -> list:
        memory = self.memory
        if filter_func is not None:
            memory = [m for i, m in enumerate(memory) if filter_func(i, m)]
        return memory

    # Alias for backward compatibility
    get = get_memory

    def add(self, memories: Union[List[Dict], Dict, None]) -> None:
        for memory in memories if isinstance(memories, (list, tuple)) else [memories]:
            if isinstance(memory, str):
                memory = self._item_cls(sender='user', content=memory)
            if isinstance(memory, AgentMessage):
                if not isinstance(memory, self._item_cls):
                    memory = self._item_cls.model_validate(memory, from_attributes=True)
                self.memory.append(memory)

    def delete(self, index: Union[List, int]) -> None:
        if isinstance(index, int):
            del self.memory[index]
        else:
            for i in sorted(index, reverse=True):
                del self.memory[i]

    def load(
        self,
        memories: Union[str, Dict, List],
        overwrite: bool = True,
    ) -> None:
        if overwrite:
            self.memory = []
        if isinstance(memories, dict):
            self.memory.append(self._item_cls.model_validate(memories))
        elif isinstance(memories, list):
            for m in memories:
                self.memory.append(self._item_cls.model_validate(m))
        else:
            raise TypeError(f'{type(memories)} is not supported')

    def save(self) -> List[dict]:
        return [m.model_dump() for m in self.memory]
