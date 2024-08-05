import json
from copy import deepcopy
from typing import Dict, List

from lagent.registry import TOOL_REGISTRY, ObjectFactory

API_PREFIX = (
    "This is the subfunction for tool '{tool_name}', you can use this tool. "
    'The description of this function is: \n{description}')

META_CN = ('当开启工具以及代码时，根据需求选择合适的工具进行调用')

INTERPRETER_CN = ('你现在已经能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。'
                  '当你向 python 发送含有 Python 代码的消息时，它将在该环境中执行。'
                  '这个工具适用于多种场景，如数据分析或处理（包括数据操作、统计分析、图表绘制），'
                  '复杂的计算问题（解决数学和物理难题），编程示例（理解编程概念或特性），'
                  '文本处理和分析（比如文本解析和自然语言处理），'
                  '机器学习和数据科学（用于展示模型训练和数据可视化），'
                  '以及文件操作和数据导入（处理CSV、JSON等格式的文件）。')

PLUGIN_CN = ('你可以使用如下工具：'
             '\n{prompt}\n'
             '如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! '
             '同时注意你可以使用的工具，不要随意捏造！')


def get_plugin_prompt(actions,
                      template=PLUGIN_CN,
                      api_desc_template=API_PREFIX):
    plugin_descriptions = []
    for action in actions if isinstance(actions, list) else [actions]:
        action = ObjectFactory.create(action, TOOL_REGISTRY)
        action_desc = deepcopy(action.description)
        if action.is_toolkit:
            for api in action_desc['api_list']:
                api['name'] = f"{action.name}.{api['name']}"
                api['description'] = api_desc_template.format(
                    tool_name=action.name, description=api['description'])
                api['parameters'] = [
                    param for param in api['parameters']
                    if param['name'] in api['required']
                ]
                plugin_descriptions.append(api)
        else:
            action_desc['description'] = api_desc_template.format(
                tool_name=action.name, description=api['description'])
            action_desc['parameters'] = [
                param for param in action_desc['parameters']
                if param['name'] in action_desc['required']
            ]
            plugin_descriptions.append(action_desc)
    return template.format(
        prompt=json.dumps(plugin_descriptions, ensure_ascii=False, indent=4))


class InternLMToolProtocol:

    def __init__(
        self,
        language: Dict = dict(
            begin='',
            end='',
            belong='assistant',
        ),
        tool: Dict = dict(
            begin='{start_token}{name}\n',
            start_token='<|action_start|>',
            name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),
            belong='assistant',
            end='<|action_end|>\n',
        ),
        execute: Dict = dict(
            role='execute', begin='', end='', fallback_role='environment'),
    ):
        self.roles_cfg = dict(tool=tool, language=language)
        self.language = language
        self.execute = execute
        self.tool = tool

    def format_sub_role(self, messages: List[Dict]) -> List[Dict]:

        def format_interpreter(message):
            if isinstance(message['content'], dict):
                return dict(
                    role=message['role'],
                    name=message['name'],
                    content=message['content']['parameters']['command'])
            else:
                return message

        def format_plugin(message):
            if isinstance(message['content'], dict):
                return dict(
                    role=message['role'],
                    name=message['name'],
                    content=json.dumps(message['content']))
            else:
                return message

        new_message = list()
        for message in messages:
            if message['role'] in [
                    'assistant', 'user', 'system', 'environment'
            ]:
                new_message.append(message)
                continue
            role_cfg = self.roles_cfg[message['role']]
            begin = role_cfg['begin']
            if message['role'] == 'tool':
                if message['name'] == 'interpreter':
                    message = format_interpreter(message)
                elif message['name'] == 'plugin':
                    message = format_plugin(message)
                else:
                    raise NotImplementedError
                begin = role_cfg['begin'].format(
                    start_token=role_cfg.get('start_token', ''),
                    name=role_cfg.get('name_map', {}).get(message['name'], ''))
            new_content = begin + message['content'] + role_cfg['end']
            if role_cfg.get('fallback_role'):
                new_message.append(
                    dict(role=role_cfg['fallback_role'], content=new_content))
            elif role_cfg.get('belong'):
                if new_message[-1]['role'] != role_cfg.get('belong'):
                    new_message.append(
                        dict(role=role_cfg.get('belong'), content=new_content))
                else:
                    new_message[-1]['content'] += new_content
            else:
                new_message.append(
                    dict(role=message['role'], content=new_content))

        return new_message

    def parse(self, message: str):
        if self.language['begin']:
            message = message.split(self.language['begin'])[-1]
        if self.tool['name_map']['plugin'] in message:
            try:
                message, action, *_ = message.split(
                    f"{self.tool['start_token']}"
                    f"{self.tool['name_map']['plugin']}")
            except ValueError:
                message, action, *_ = message.split(
                    f"{self.tool['name_map']['plugin']}")
                tool_start_idx = message.rfind(self.tool['start_token'])
                if tool_start_idx != -1:
                    message = message[:tool_start_idx]
                message = message.strip()
            action = action.split(self.tool['end'].strip())[0].strip()
            return 'plugin', message, action
        if self.tool['name_map']['interpreter'] in message:
            try:
                message, code, *_ = message.split(
                    f"{self.tool['start_token']}"
                    f"{self.tool['name_map']['interpreter']}")
            except ValueError:
                message, code, *_ = message.split(
                    self.tool['name_map']['interpreter'])
                tool_start_idx = message.rfind(self.tool['start_token'])
                if tool_start_idx != -1:
                    message = message[:tool_start_idx]
                message = message.strip()
            code = code.split(self.tool['end'].strip())[0].strip()
            return 'interpreter', message, code
        start_idx = message.rfind(self.tool['start_token'])
        if start_idx != -1:
            message = message[:start_idx]
        message = message.strip()
        return None, message, None
