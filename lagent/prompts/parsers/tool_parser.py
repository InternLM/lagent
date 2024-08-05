import json
import re
from typing import Optional

from lagent.prompts.protocols import InternLMToolProtocol
from lagent.registry import PARSER_REGISTRY, AutoRegister
from lagent.schema import AgentStatusCode


class InternLMToolParser(metaclass=AutoRegister(PARSER_REGISTRY)):

    def __init__(self,
                 finish_pattern: Optional[str] = None,
                 protocol=InternLMToolProtocol()):
        self.protocol = protocol
        self.finish_pattern = finish_pattern and re.compile(
            finish_pattern, re.DOTALL)

    def parse(self, data: str):
        tool_type, thought, action = self.protocol.parse(data)
        status = AgentStatusCode.STREAM_ING
        if tool_type == 'plugin':
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                status = AgentStatusCode.SESSION_INVALID_ARG
        if self.finish_pattern and self.finish_pattern.search(
                thought) or not self.finish_pattern and not tool_type:
            status = AgentStatusCode.END
        return dict(
            tool_type=tool_type, thought=thought, action=action, status=status)
