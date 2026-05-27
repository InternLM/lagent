from typing import Dict, List, Optional, Tuple

from lagent.memory import Memory
from lagent.prompts import StrParser
from lagent.schema import ActionReturn


class DefaultAggregator:

    def aggregate(
        self,
        messages: Memory,
        name: str,
        parser: StrParser = None,
        system_instruction: str = None,
        tools: List[Dict] = None,
    ) -> Tuple[List[Dict[str, str]], Optional[List[Dict]]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.extend(self.aggregate_system_intruction(system_instruction))
        for message in messages:
            if message.sender == name:
                _message.append(message.to_model_request('assistant'))
            elif isinstance(message.content, list):
                _message.extend(message.to_model_request('tool'))
            elif (
                len(_message) > 0
                and _message[-1]['role'] == 'user'
                and isinstance(_message[-1]['content'], str)
                and isinstance(message.content, str)
            ):
                _message[-1]['content'] += message.content
                _message[-1]['extra_info'] = message.extra_info
            else:
                _message.append(message.to_model_request('user'))

        latest_env_info = None
        for message in messages:
            if getattr(message, 'env_info', None) is not None:
                latest_env_info = message.env_info

        tools_to_use = tools
        if latest_env_info and latest_env_info.get("tools"):
            tools_to_use = latest_env_info.get("tools")

        return _message, tools_to_use

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
