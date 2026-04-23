from typing import Callable, Dict, List, Optional, Union

from lagent.schema import AgentMessage
from lagent.utils import load_class_from_string


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
                memory = AgentMessage(sender='user', content=memory)
            if isinstance(memory, AgentMessage):
                self.memory.append(memory)

    def delete(self, index: Union[List[int], int]) -> None:
        if isinstance(index, int):
            del self.memory[index]
        else:
            for i in sorted(index, reverse=True):
                del self.memory[i]

    def load(self, memories: Union[dict, List], overwrite: bool = True) -> None:
        if overwrite:
            self.memory = []
        if isinstance(memories, dict):
            memories = memories.copy()
            _cls = (
                load_class_from_string(memories.pop('__model_spec__'))
                if '__model_spec__' in memories
                else AgentMessage
            )
            self.memory.append(_cls.model_validate(memories))
        elif isinstance(memories, list):
            for m in memories:
                m = m.copy()
                _cls = load_class_from_string(m.pop('__model_spec__')) if '__model_spec__' in m else AgentMessage
                self.memory.append(_cls.model_validate(m))
        else:
            raise TypeError(f'{type(memories)} is not supported')

    def save(self) -> List[dict]:
        memory = []
        for m in self.memory:
            m_dumped = m.model_dump()
            m_dumped['__model_spec__'] = f'{m.__module__}.{m.__class__.__name__}'
            memory.append(m_dumped)
        return memory
