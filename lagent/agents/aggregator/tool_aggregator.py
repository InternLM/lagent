from typing import Dict, List, Optional, Union

from lagent.agents.aggregator.default_aggregator import DefaultAggregator
from lagent.memory.base_memory import Memory
from lagent.prompts.parsers.tool_parser import MixedToolParser, ToolParser, ToolStatusCode


class InternLMToolAggregator(DefaultAggregator):

    def __init__(self,
                 environment_role='environment',
                 environment_begin='',
                 environment_end='',
                 user_names: Optional[List[str]] = None,
                 few_shot: Optional[List[List[dict]]] = None):
        self.environment_role = environment_role
        self.environment_begin = environment_begin
        self.environment_end = environment_end
        self.user_names = user_names or ['user']
        self.few_shot = few_shot or []

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  parser: Union[ToolParser, MixedToolParser],
                  system_instruction: str = None) -> List[Dict[str, str]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.extend(
                self.aggregate_system_intruction(system_instruction))
        tool_instruction = parser.format_instruction()
        if tool_instruction:
            if isinstance(tool_instruction, str):
                tool_instruction = dict(
                    role='system', content=tool_instruction)
                if parser.tool_type:
                    tool_instruction['name'] = parser.tool_type
            if isinstance(tool_instruction, dict):
                tool_instruction = [tool_instruction]
            _message.extend(tool_instruction)

        for shot in self.few_shot:
            i = 0
            while i < len(shot):
                msg = shot[i]
                if msg['role'] in ['assistant', 'user', 'system']:
                    _message.append(msg)
                elif msg['role'] == self.environment_role:
                    if not msg['content'].startswith(self.environment_begin):
                        msg['content'] = self.environment_begin + msg['content']
                    if not msg['content'].endswith(self.environment_end):
                        msg['content'] += self.environment_end
                    _message.append(msg)
                elif msg['role'] in ['thought', 'language']:
                    if i < len(shot) - 1 and shot[i + 1]['role'] == 'tool':
                        _message.append(
                            dict(
                                role='assistant',
                                content=parser.format_response(
                                    dict(
                                        tool_type=shot[i + 1]['name'],
                                        thought=msg['content'],
                                        action=shot[i + 1]['content'],
                                        status=None))))
                        i += 1
                    else:
                        _message.append(
                            dict(
                                role='assistant',
                                content=parser.format_response(
                                    dict(
                                        tool_type=None,
                                        thought=msg['content'],
                                        action=None,
                                        status=None))))
                else:
                    raise KeyError(f'Unkown role: {msg["role"]}')
                i += 1

        tool_type = None
        for message in messages:
            if message.sender == name:
                if isinstance(message.formatted, dict):
                    parsed = message.formatted
                    if parsed['status'] == ToolStatusCode.PARSING_ERROR:
                        continue
                    _message.append(
                        dict(
                            role='assistant',
                            content=parser.format_response(parsed)))
                    tool_type = parsed['tool_type']
                else:
                    _message.append(
                        dict(role='assistant', content=str(message.content)))
            elif message.sender in self.user_names:
                _message.append(dict(role='user', content=message.content))
            else:
                msg = dict(
                    role=self.environment_role,
                    content=self.environment_begin + str(message.content) +
                    self.environment_end)
                if tool_type:
                    msg['name'] = tool_type
                _message.append(msg)
        return _message
