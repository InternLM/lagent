from typing import List

from lagent.memory import Memory
from lagent.prompts import StrParser
from lagent.schema import ActionReturn


class DefaultAggregator:

    def aggregate(self, messages: Memory, name: str, parser: StrParser = None, system_instruction=None) -> List[dict]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.extend(self.aggregate_system_intruction(system_instruction))
        for message in messages:
            if message.sender == name:
                _message.append(message.to_model_request())
            else:
                user_message, extra_info = message.content, message.extra_info
                if isinstance(user_message, list):
                    for m in user_message:
                        if isinstance(m, dict):
                            m = ActionReturn(**m)
                        assert isinstance(m, ActionReturn), f"Expected m to be ActionReturn, but got {type(m)}"
                        _message.append(
                            dict(
                                role='tool',
                                tool_call_id=m.tool_call_id,
                                content=m.format_result(),
                                name=m.type,
                                extra_info=extra_info,
                            )
                        )
                else:
                    if len(_message) > 0 and _message[-1]['role'] == 'user':
                        _message[-1]['content'] += user_message
                        _message[-1]['extra_info'] = extra_info
                    else:
                        _message.append(dict(role='user', content=user_message, extra_info=extra_info))
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
                    raise KeyError(f"Missing required key 'role' or 'content': {msg}")
        return system_intruction
