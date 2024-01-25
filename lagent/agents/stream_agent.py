import json
import logging
from copy import deepcopy
from typing import Dict, List, Union

from ilagent.schema import AgentReturn, AgentStatusCode

from lagent import BaseAgent
from lagent.actions import ActionExecutor
from lagent.llms import BaseAPIModel, BaseModel
from lagent.schema import ActionReturn, ActionStatusCode

API_PREFIX = (
    "This is the subfunction for tool '{tool_name}', you can use this tool. "
    'The description of this function is: \n{description}')

META_INS = ('You are InternLM, a large language model trained by PJLab. '
            'Answer as concisely as possible. '
            '当开启工具以及代码时，根据需求选择合适的工具进行调用')

INTERPRETER_CN = ('你现在可以通过如下格式向 Jupyter Notebook 发送并执行代码:'
                  '\n<|action_start|><|interpreter|>```python\n\n代码\n\n```\n'
                  '\n当遇到以下问题时，请使用上述格式调用 Jupyter Notebook 去解决，并根据执行结果做出友好的回复：\n'
                  '1. 文件操作和数据导入，比如处理CSV、JSON等格式文件\n'
                  '2. 数据分析或处理，比如数据操作或图像绘制如折线图、柱状图等\n'
                  '3. 数学相关的问题。当遇到数学问题时，你需要分析题目，并给出代码去解决这个题目')

PLUGIN_CN = (
    '你可以使用如下工具：'
    '\n{prompt}\n'
    '当你需要使用工具时，你可以使用如下格式：\n'
    '<|action_start|><|plugin|>{{"name": "工具名称", "parameters": {{参数}}}}\n'
    '如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! '
    '同时注意你可以使用的工具，不要随意捏造！')


class StreamProtocol:

    def __init__(
        self,
        meta_prompt=META_INS,
        interpreter_prompt=INTERPRETER_CN,
        plugin_prompt=PLUGIN_CN,
        few_shot=None,
        language=dict(
            begin='',
            end='',
            belong='assistant',
        ),
        tool=dict(
            begin='{start_token}{name}\n',
            start_token='<|action_start|>',
            name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),
            belong='assistant',
            end='<|action_end|>\n',
        ),
        execute: dict = dict(
            role='execute', begin='', end='', fallback_role='environment'),
    ) -> None:
        self.meta_prompt = meta_prompt
        self.interpreter_prompt = interpreter_prompt
        self.plugin_prompt = plugin_prompt
        self.roles_cfg = dict(tool=tool, language=language)
        self.language = language
        self.execute = execute
        self.tool = tool
        self.few_shot = few_shot

    def format_sub_role(self, messages: List[Dict]) -> List[Dict]:

        def format_interpreter(message):
            if isinstance(message['content'], dict):
                # assert message['content']['name'] == 'IPythonInterpreter'
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

    def format(self,
               inner_step: List[Dict],
               plugin_executor: ActionExecutor = None,
               interpreter_executor: ActionExecutor = None,
               **kwargs) -> list:
        formatted = []
        if self.meta_prompt:
            formatted.append(dict(role='system', content=self.meta_prompt))
        if interpreter_executor and self.interpreter_prompt:
            interpreter_info = interpreter_executor.get_actions_info()[0]
            interpreter_prompt = self.interpreter_prompt.format(
                code_prompt=interpreter_info['description'])
            formatted.append(
                dict(
                    role='system',
                    content=interpreter_prompt,
                    name='interpreter'))
        if plugin_executor and plugin_executor.actions and self.plugin_prompt:
            plugin_descriptions = []
            for api_info in plugin_executor.get_actions_info():
                plugin = deepcopy(api_info)
                if isinstance(api_info, dict):
                    tool_name = api_info['name'].split('.')[0]
                    plugin['description'] = API_PREFIX.format(
                        tool_name=tool_name, description=plugin['description'])
                plugin_descriptions.append(plugin)
            plugin_prompt = self.plugin_prompt.format(
                prompt=json.dumps(
                    plugin_descriptions, ensure_ascii=False, indent=4))
            formatted.append(
                dict(role='system', content=plugin_prompt, name='plugin'))
        if self.few_shot:
            for few_shot in self.few_shot:
                formatted += self.format_sub_role(few_shot)
        formatted += self.format_sub_role(inner_step)
        return formatted

    def parse(self, message, plugin_executor: ActionExecutor,
              interpreter_executor: ActionExecutor):
        if self.language['begin']:
            message = message.split(self.language['begin'])[-1]
        if self.tool['name_map']['plugin'] in message:
            message, action = message.split(
                f"{self.tool['start_token']}{self.tool['name_map']['plugin']}")
            action = action.split(self.tool['end'].strip())[0]
            action = json.loads(action)
            return 'plugin', message, action
        if self.tool['name_map']['interpreter'] in message:
            message, code = message.split(
                f"{self.tool['start_token']}"
                f"{self.tool['name_map']['interpreter']}")
            code = code.split(self.tool['end'].strip())[0].strip()
            return 'interpreter', message, dict(
                name=interpreter_executor.action_names()[0],
                parameters=dict(command=code))
        return None, message, None

    def format_response(self, action_return, name) -> str:
        if action_return.state == ActionStatusCode.SUCCESS:
            if isinstance(action_return.result, list):
                response = []
                for item in action_return.result:
                    if item['type'] == 'text':
                        response.append(item['content'])
                    else:
                        response.append(f"[{item['type']}]({item['content']})")
                response = '\n'.join(response)
            elif isinstance(action_return.result, dict):
                response = action_return.result['text']
                if 'image' in action_return.result:
                    response += '\n'.join([
                        f'[image]({im})'
                        for im in action_return.result['image']
                    ])
                if 'audio' in action_return.result:
                    response += '\n'.join([
                        f'[audio]({im})'
                        for im in action_return.result['audio']
                    ])
            elif isinstance(action_return.result, str):
                response = action_return.result
            else:
                raise NotImplementedError
        else:
            response = action_return.errmsg
        content = self.execute['begin'] + response + self.execute['end']
        if self.execute.get('fallback_role'):
            return dict(
                role=self.execute['fallback_role'], content=content, name=name)
        elif self.execute.get('belong'):
            return dict(
                role=self.execute['belong'], content=content, name=name)
        else:
            return dict(role=self.execute['role'], content=response, name=name)


class StreamAgent(BaseAgent):

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 plugin_executor: ActionExecutor = None,
                 interpreter_executor: ActionExecutor = None,
                 protocol=StreamProtocol(),
                 max_turn: int = 3) -> None:
        self.max_turn = max_turn
        self._interpreter_executor = interpreter_executor
        super().__init__(
            llm=llm, action_executor=plugin_executor, protocol=protocol)

    def chat(self, message: Union[str, Dict], **kwargs) -> AgentReturn:
        if isinstance(message, str):
            message = dict(role='user', content=message)
        if isinstance(message, dict):
            message = [message]
        inner_history = message[:]
        offset = len(inner_history)
        agent_return = AgentReturn()
        for _ in range(self.max_turn):
            # list of dict
            prompt = self._protocol.format(
                inner_step=inner_history,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            response = self._llm.chat(prompt, **kwargs)
            name, language, action = self._protocol.parse(
                message=response,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            if name:
                if name == 'plugin':
                    if self._action_executor:
                        executor = self._action_executor
                    else:
                        logging.info(msg='No plugin is instantiated!')
                        continue
                elif name == 'interpreter':
                    if self._interpreter_executor:
                        executor = self._interpreter_executor
                    else:
                        logging.info(msg='No interpreter is instantiated!')
                        continue
                else:
                    logging.info(
                        msg=(f"Invalid name '{name}'. Currently only 'plugin' "
                             "and 'interpreter' are supported."))
                    continue
                action_return: ActionReturn = executor(action['name'],
                                                       action['parameters'])
                action_return.thought = language
                agent_return.actions.append(action_return)
            inner_history.append(dict(role='language', content=language))
            if not name or action_return.type == executor.finish_action.name:
                agent_return.response = language
                agent_return.state = AgentStatusCode.END
                break
            else:
                inner_history.append(
                    dict(role='tool', content=action, name=name))
                inner_history.append(
                    self._protocol.format_response(action_return, name=name))
            yield agent_return
        agent_return.inner_steps = inner_history[offset:]
        yield agent_return
