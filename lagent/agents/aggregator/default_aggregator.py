from typing import Dict, List

from lagent.memory import Memory
from lagent.prompts import StrParser


class DefaultAggregator:

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  parser: StrParser = None,
                  system_instruction: str = None) -> List[Dict[str, str]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.extend(
                self.aggregate_system_intruction(system_instruction))
        for message in messages:
            if message.sender == name:
                _message.append(
                    dict(role='assistant', content=str(message.content)))
            else:
                user_message = message.content
                if len(_message) > 0 and _message[-1]['role'] == 'user':
                    _message[-1]['content'] += user_message
                else:
                    _message.append(dict(role='user', content=user_message))
        return _message

    @staticmethod
    def aggregate_system_intruction(system_intruction) -> List[dict]:
        if isinstance(system_intruction, str):
            system_intruction = dict(role='system', content=system_intruction)
        if isinstance(system_intruction, dict):
            system_intruction = [system_intruction]
        if isinstance(system_intruction, list):
            for msg in system_intruction:
                if not isinstance(msg, dict):
                    raise TypeError(f'Unsupported message type: {type(msg)}')
                if not ('role' in msg and 'content' in msg):
                    raise KeyError(
                        f"Missing required key 'role' or 'content': {msg}")
        return system_intruction
