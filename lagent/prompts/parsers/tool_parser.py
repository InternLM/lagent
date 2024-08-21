import json
import re
from typing import Any, Callable, List

from lagent.prompts.parsers import StrParser
from lagent.schema import AgentStatusCode
from lagent.utils import create_object, load_class_from_string


class ToolParser(StrParser):

    def __init__(self,
                 tool_type: str,
                 template: str = '',
                 begin: str = '<tool>\n',
                 end: str = '</tool>\n',
                 validate: Callable[[str], Any] = None,
                 **kwargs):
        super().__init__(template, begin=begin, end=end, **kwargs)
        self.template = template
        self.tool_type = tool_type
        self.pattern = re.compile(
            '(.*?){}(.*)({})?'.format(re.escape(begin), re.escape(end)),
            re.DOTALL)
        self.validate = load_class_from_string(validate) if isinstance(
            validate, str) else validate

    def parse_response(self, data: str) -> dict:
        match = self.pattern.search(data)
        if not match:
            return dict(
                tool_type=None,
                thought=data.strip(),
                action=None,
                status=AgentStatusCode.END)
        thought, action = match.group(1).strip(), match.group(2).strip()
        status = AgentStatusCode.STREAM_ING
        if self.validate:
            try:
                action = self.validate(action)
            except Exception:
                status = AgentStatusCode.SESSION_INVALID_ARG
        return dict(
            tool_type=self.tool_type,
            thought=thought,
            action=action,
            status=status)

    def format_response(self, parsed: dict) -> str:
        if parsed['action'] is None:
            return parsed['thought']
        assert parsed['tool_type'] == self.tool_type
        if isinstance(parsed['action'], dict):
            action = json.dumps(parsed['action'], ensure_ascii=False)
        else:
            action = str(parsed['action'])
        return parsed['thought'] + self.format_field[
            'begin'] + action + self.format_field['end']


class InterpreterParser(ToolParser):

    def __init__(self,
                 tool_type: str = 'interpreter',
                 template: str = '',
                 begin: str = '<|action_start|><|interpreter|>\n',
                 end: str = '<|action_end|>\n',
                 validate: Callable[[str], Any] = None,
                 **kwargs):
        super().__init__(tool_type, template, begin, end, validate, **kwargs)


class PluginParser(ToolParser):

    def __init__(self,
                 tool_type: str = 'plugin',
                 template: str = '',
                 begin: str = '<|action_start|><|plugin|>\n',
                 end: str = '<|action_end|>\n',
                 validate: Callable[[str], Any] = json.loads,
                 **kwargs):
        super().__init__(tool_type, template, begin, end, validate, **kwargs)


class MixedToolParser(StrParser):

    def __init__(self,
                 template='',
                 parsers: List[ToolParser] = None,
                 **format_field):
        self.parsers = {}
        for parser in parsers or []:
            parser = create_object(parser)
            self.parsers[parser.tool_type] = parser
        super().__init__(template, **format_field)

    def format_instruction(self) -> List[dict]:
        inst = []
        content = super().format_instruction().strip()
        if content:
            inst.append(dict(role='system', content=content))
        for name, parser in self.parsers.items():
            content = parser.format_instruction().strip()
            if content:
                inst.append(dict(role='system', content=content, name=name))
        return inst

    def parse_response(self, data: str) -> dict:
        res = dict(
            tool_type=None,
            thought=data.strip(),
            action=None,
            status=AgentStatusCode.END)
        for name, parser in self.parsers.items():
            res = parser.parse_response(data)
            if res['tool_type'] == name:
                break
        return res

    def format_response(self, parsed: dict) -> str:
        if parsed['action'] is None:
            return parsed['thought']
        assert parsed['tool_type'] in self.parsers
        return self.parsers[parsed['tool_type']].format_response(parsed)
