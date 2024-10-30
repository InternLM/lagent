import json
from enum import IntEnum

# import re
from typing import Any, Callable, List, Optional

from lagent.prompts.parsers import StrParser
from lagent.utils import create_object, load_class_from_string


def default_plugin_validate(plugin: str):
    plugin = plugin.strip()
    if not (plugin.startswith('{') and plugin.endswith("}")):
        raise json.decoder.JSONDecodeError
    return json.loads(plugin)


class ToolStatusCode(IntEnum):
    NO_TOOL = 0
    VALID_TOOL = 1
    PARSING_ERROR = -1


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
        # self.pattern = re.compile(
        #     '(.*?){}(.*)({})?'.format(re.escape(begin), re.escape(end)),
        #     re.DOTALL)
        self.validate = load_class_from_string(validate) if isinstance(
            validate, str) else validate

    def parse_response(self, data: str) -> dict:
        if self.format_field['begin'] not in data:
            return dict(
                tool_type=None,
                thought=data,
                action=None,
                status=ToolStatusCode.NO_TOOL)
        thought, action, *_ = data.split(self.format_field["begin"])
        action = action.split(self.format_field['end'])[0]
        status = ToolStatusCode.VALID_TOOL
        if self.validate:
            try:
                action = self.validate(action)
            except Exception:
                status = ToolStatusCode.PARSING_ERROR
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
                 validate: Callable[[str], Any] = default_plugin_validate,
                 **kwargs):
        super().__init__(tool_type, template, begin, end, validate, **kwargs)


class MixedToolParser(StrParser):

    def __init__(self,
                 tool_type: Optional[str] = None,
                 template='',
                 parsers: List[ToolParser] = None,
                 **format_field):
        self.parsers = {}
        self.tool_type = tool_type
        for parser in parsers or []:
            parser = create_object(parser)
            self.parsers[parser.tool_type] = parser
        super().__init__(template, **format_field)

    def format_instruction(self) -> List[dict]:
        inst = []
        content = super().format_instruction()
        if content.strip():
            msg = dict(role='system', content=content)
            if self.tool_type:
                msg['name'] = self.tool_type
            inst.append(msg)
        for name, parser in self.parsers.items():
            content = parser.format_instruction()
            if content.strip():
                inst.append(dict(role='system', content=content, name=name))
        return inst

    def parse_response(self, data: str) -> dict:
        res = dict(
            tool_type=None,
            thought=data,
            action=None,
            status=ToolStatusCode.NO_TOOL)
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
