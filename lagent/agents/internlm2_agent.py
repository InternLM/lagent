import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

from termcolor import colored

from lagent.actions import ActionExecutor
from lagent.agents.base_agent import BaseAgent
from lagent.llms import BaseAPIModel, BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn, AgentStatusCode, ModelStatusCode  # noqa: E501

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


class Internlm2Protocol:

    def __init__(
        self,
        meta_prompt: str = META_CN,
        interpreter_prompt: str = INTERPRETER_CN,
        plugin_prompt: str = PLUGIN_CN,
        few_shot: Optional[List] = None,
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
                    # only keep required parameters
                    required_parameters = [
                        param for param in plugin['parameters']
                        if param['name'] in plugin['required']
                    ]
                    plugin['parameters'] = required_parameters
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

    def parse(self,
              message,
              plugin_executor: ActionExecutor = None,
              interpreter_executor: ActionExecutor = None):
        if self.language['begin']:
            message = message.split(self.language['begin'])[-1]
        if self.tool['name_map']['plugin'] in message:
            message, action = message.split(
                f"{self.tool['start_token']}{self.tool['name_map']['plugin']}")
            action = action.split(self.tool['end'].strip())[0]
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
            return 'interpreter', message, dict(
                name=interpreter_executor.action_names()[0] if isinstance(
                    interpreter_executor, ActionExecutor) else
                'IPythonInterpreter',
                parameters=dict(command=code))
        return None, message.split(self.tool['start_token'])[0], None

    def format_response(self, action_return, name) -> dict:
        if action_return.state == ActionStatusCode.SUCCESS:
            response = action_return.format_result()
        else:
            response = str(action_return.errmsg)
        content = self.execute['begin'] + response + self.execute['end']
        if self.execute.get('fallback_role'):
            return dict(
                role=self.execute['fallback_role'], content=content, name=name)
        elif self.execute.get('belong'):
            return dict(
                role=self.execute['belong'], content=content, name=name)
        return dict(role=self.execute['role'], content=response, name=name)


class Internlm2Agent(BaseAgent):

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 plugin_executor: ActionExecutor = None,
                 interpreter_executor: ActionExecutor = None,
                 protocol=Internlm2Protocol(),
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
                    try:
                        action = json.loads(action)
                    except Exception as e:
                        logging.info(msg=f'Invaild action {e}')
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
        agent_return.inner_steps = inner_history[offset:]
        return agent_return

    def stream_chat(self, message: List[dict], **kwargs) -> AgentReturn:
        if isinstance(message, str):
            message = dict(role='user', content=message)
        if isinstance(message, dict):
            message = [message]
        inner_history = message[:]
        offset = len(inner_history)
        agent_return = AgentReturn()
        agent_return.inner_steps = deepcopy(inner_history)
        last_agent_state = AgentStatusCode.SESSION_READY
        for _ in range(self.max_turn):
            # list of dict
            prompt = self._protocol.format(
                inner_step=inner_history,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            response = ''
            for model_state, res, _ in self._llm.stream_chat(prompt, **kwargs):
                model_state: ModelStatusCode
                response = res
                if model_state.value < 0:
                    agent_return.state = getattr(AgentStatusCode,
                                                 model_state.name)
                    yield deepcopy(agent_return)
                    return
                else:
                    name, language, action = self._protocol.parse(
                        message=response,
                        plugin_executor=self._action_executor,
                        interpreter_executor=self._interpreter_executor,
                    )
                    if name:
                        if model_state == ModelStatusCode.END:
                            agent_state = last_agent_state + 1
                            if name == 'plugin':
                                if self._action_executor:
                                    executor = self._action_executor
                                else:
                                    logging.info(
                                        msg='No plugin is instantiated!')
                                    continue
                                try:
                                    action = json.loads(action)
                                except Exception as e:
                                    logging.info(msg=f'Invaild action {e}')
                                    continue
                            elif name == 'interpreter':
                                if self._interpreter_executor:
                                    executor = self._interpreter_executor
                                else:
                                    logging.info(
                                        msg='No interpreter is instantiated!')
                                    continue
                            agent_return.state = agent_state
                            agent_return.response = action
                        else:
                            agent_state = (
                                AgentStatusCode.PLUGIN_START if name
                                == 'plugin' else AgentStatusCode.CODING)
                            if agent_state != last_agent_state:
                                # agent_return.state = agent_state
                                agent_return.response = language
                                yield deepcopy(agent_return)
                            agent_return.state = agent_state
                            agent_return.response = action
                    else:
                        agent_state = AgentStatusCode.STREAM_ING
                        agent_return.state = agent_state
                        agent_return.response = language
                    last_agent_state = agent_state
                    yield deepcopy(agent_return)
            print(colored(response, 'red'))
            if name:
                action_return: ActionReturn = executor(action['name'],
                                                       action['parameters'])
                action_return.type = action['name']
                action_return.thought = language
                agent_return.actions.append(action_return)
                print(colored(action_return.result, 'magenta'))
            inner_history.append(dict(role='language', content=language))
            if not name:
                agent_return.response = language
                break
            elif action_return.type == executor.finish_action.name:
                try:
                    response = action_return.args['text']['response']
                except Exception:
                    logging.info(msg='Unable to parse FinishAction.')
                    response = ''
                agent_return.response = response
                break
            else:
                inner_history.append(
                    dict(role='tool', content=action, name=name))
                inner_history.append(
                    self._protocol.format_response(action_return, name=name))
                agent_state += 1
                agent_return.state = agent_state
                agent_return.inner_steps = deepcopy(inner_history[offset:])
                yield agent_return
        agent_return.inner_steps = deepcopy(inner_history[offset:])
        agent_return.state = AgentStatusCode.END
        yield agent_return

    def batch_chat(self, batch_messages: List[Union[List[dict], dict, str]],
                   **kwargs) -> List[AgentReturn]:
        assert isinstance(batch_messages, list)
        agent_returns = [AgentReturn() for _ in range(len(batch_messages))]
        inner_histories = []
        for message in batch_messages:
            if isinstance(message, str):
                message = dict(role='user', content=message)
            if isinstance(message, dict):
                message = [message]
            inner_histories.append(deepcopy(message))
        offsets = [len(inner) for inner in inner_histories]
        finish_flags = [False for _ in range(len(batch_messages))]
        for _ in range(self.max_turn):
            unfinished = [
                index for index, is_finish in enumerate(finish_flags)
                if not is_finish
            ]
            if not unfinished:
                break
            batch_prompt = []
            for index in unfinished:
                batch_prompt.append(
                    self._protocol.format(
                        inner_step=inner_histories[index],
                        plugin_executor=self._action_executor,
                        interpreter_executor=self._interpreter_executor,
                    ))
            batch_response = self._llm.chat(batch_prompt, **kwargs)
            executor2action_args = defaultdict(lambda: defaultdict(list))
            for response, index in zip(batch_response, unfinished):
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
                        try:
                            action = json.loads(action)
                        except Exception as e:
                            logging.info(msg=f'Invaild action {e}')
                            continue
                    elif name == 'interpreter':
                        if self._interpreter_executor:
                            executor = self._interpreter_executor
                        else:
                            logging.info(msg='No interpreter is instantiated!')
                            continue
                    else:
                        logging.info(
                            msg=  # noqa
                            (f"Invalid name '{name}'. Currently only 'plugin' "
                             "and 'interpreter' are supported."))
                        continue
                    executor2action_args[executor][action['name']].append(
                        (index, name, action, language))
                else:
                    inner_histories[index].append(
                        dict(role='language', content=language))
                    agent_returns[index].response = language
                    agent_returns[index].state = AgentStatusCode.END
                    finish_flags[index] = True

            for executor, action_args in executor2action_args.items():
                for action_name, args in action_args.items():
                    indexes, _, actions, _ = list(zip(*args))
                    action_returns = executor.actions[action_name]([
                        action['parameters']['command'] for action in actions
                    ], list(indexes))
                    for (index, name, action,
                         language), action_return in zip(args, action_returns):
                        action_return.thought = language
                        inner_histories[index].append(
                            dict(role='language', content=language))
                        if action_return.type == executor.finish_action.name:
                            agent_returns[index].response = language
                            agent_returns[index].state = AgentStatusCode.END
                            finish_flags[index] = True
                        else:
                            inner_histories[index].append(
                                dict(role='tool', content=action, name=name))
                            inner_histories[index].append(
                                self._protocol.format_response(
                                    action_return, name=name))
        for agent_return, offset, inner_history in zip(agent_returns, offsets,
                                                       inner_histories):
            agent_return.inner_steps = inner_history[offset:]
        return agent_returns
