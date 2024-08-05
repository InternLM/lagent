from typing import Dict, List, Optional

from lagent.agents.aggregator.default_aggregator import DefaultAggregator
from lagent.memory.base_memory import Memory
from lagent.prompts.protocols import InternLMToolProtocol
from lagent.schema import AgentStatusCode


class InternLMToolAggregator(DefaultAggregator):

    def __init__(self,
                 meta_prompt: str = None,
                 interpreter_prompt: str = None,
                 plugin_prompt: str = None,
                 few_shot: Optional[List] = None,
                 protocol=InternLMToolProtocol(),
                 remove_message_name: bool = False,
                 **kwargs):
        self.meta_prompt = meta_prompt
        self.interpreter_prompt = interpreter_prompt
        self.plugin_prompt = plugin_prompt
        self.few_shot = few_shot or []
        self.protocol = protocol
        self.remove_message_name = remove_message_name
        super().__init__(**kwargs)

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  system_instruction: str = None) -> List[Dict[str, str]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.append(
                dict(role='system', content=str(system_instruction)))
        if self.meta_prompt:
            _message.append(dict(role='system', content=self.meta_prompt))
        if self.interpreter_prompt:
            _message.append(
                dict(
                    role='system',
                    content=self.interpreter_prompt,
                    name='interpreter'))
        if self.plugin_prompt:
            _message.append(
                dict(role='system', content=self.plugin_prompt, name='plugin'))
        for few_shot in self.few_shot:
            _message += self.protocol.format_sub_role(few_shot)

        inner_steps = []
        for message in messages:
            if message.sender == name:
                if isinstance(message.formatted, dict):
                    formatted = message.formatted
                    if formatted[
                            'status'] == AgentStatusCode.SESSION_INVALID_ARG:
                        continue
                    inner_steps.append(
                        dict(role='language', content=formatted['thought']))
                    if formatted['tool_type']:
                        inner_steps.append(
                            dict(
                                role='tool',
                                content=formatted['action'],
                                name=formatted['tool_type']))
                else:
                    inner_steps.append(
                        dict(role='assistant', content=str(message.content)))
            elif message.sender == 'user':
                inner_steps.append(dict(role='user', content=message.content))
            else:
                execute_cfg = self.protocol.execute
                assert isinstance(execute_cfg, dict)
                content = execute_cfg['begin'] + message.content + execute_cfg[
                    'end']
                execute_role = execute_cfg.get(
                    'fallback_role',
                    execute_cfg.get('role', execute_cfg['role']))
                inner_steps.append(
                    dict(
                        role=execute_role,
                        content=content,
                        name=inner_steps[-1].get('name')
                        if inner_steps else None))
        _message += self.protocol.format_sub_role(inner_steps)
        if self.remove_message_name:
            for msg in _message:
                msg.pop('name', None)
        return _message
