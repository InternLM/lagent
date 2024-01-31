from typing import Dict, List, Tuple, Union

from lagent.actions import ActionExecutor
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn
from .base_agent import BaseAgent

# The Chinese prompts for ReAct

CALL_PROTOCOL_CN = """你是一个可以调用外部工具的助手，可以使用的工具包括：
{tool_description}
如果使用工具请遵循以下格式回复：
```
{thought}思考你当前步骤需要解决什么问题，是否需要使用工具
{action}工具名称，你的工具必须从 [{action_names}] 选择
{action_input}工具输入参数
```
工具返回按照以下格式回复：
```
{response}调用工具后的结果
```
如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
```
{thought}给出最终答案的思考过程
{finish}最终答案
```
开始!"""

FORCE_STOP_PROMPT_CN = '你需要基于历史消息返回一个最终结果'

# The English prompts for ReAct

CALL_PROTOCOL_EN = """You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please use the following format:
```
{thought}Think what you need to solve, do you need to use tools?
{action}the tool name, should be one of [{action_names}]
{action_input}the input to the action
```
The response after utilizing tools should using the following format:
```
{response}the results after call the tool.
```
If you already know the answer, or you do not need to use tools,
please using the following format to reply:
```
{thought}the thought process to get the final answer
{finish}final answer
```
Begin!"""

FORCE_STOP_PROMPT_EN = """You should directly give results
 based on history information."""


class ReActProtocol:
    """A wrapper of ReAct prompt which manages the response from LLM and
    generate desired prompts in a ReAct format.

    Args:
        thought (dict): the information of thought pattern
        action (dict): the information of action pattern
        action_input (dict): the information of action_input pattern
        response (dict): the information of response pattern
        finish (dict): the information of finish pattern
        call_protocol (str): the format of ReAct
        force_stop (str): the prompt to force LLM to generate response
    """

    def __init__(self,
                 thought: dict = dict(
                     role='THOUGHT',
                     begin='Thought:',
                     end='\n',
                     belong='assistant'),
                 action: dict = dict(role='ACTION', begin='Action:', end='\n'),
                 action_input: dict = dict(
                     role='ARGS', begin='Action Input:', end='\n'),
                 response: dict = dict(
                     role='RESPONSE', begin='Response:', end='\n'),
                 finish: dict = dict(
                     role='FINISH', begin='Final Answer:', end='\n'),
                 call_protocol: str = CALL_PROTOCOL_EN,
                 force_stop: str = FORCE_STOP_PROMPT_EN) -> None:
        self.call_protocol = call_protocol
        self.force_stop = force_stop
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.response = response
        self.finish = finish

    def format(self,
               chat_history: List[Dict],
               inner_step: List[Dict],
               action_executor: ActionExecutor,
               force_stop: bool = False) -> list:
        """Generate the ReAct format prompt.

        Args:
            chat_history (List[Dict]): The history log in previous runs.
            inner_step (List[Dict]): The log in the current run.
            action_executor (ActionExecutor): the action manager to
                execute actions.
            force_stop (boolean): whether force the agent to give responses
                under pre-defined turns.

        Returns:
            List[Dict]: ReAct format prompt.
        """

        call_protocol = self.call_protocol.format(
            tool_description=action_executor.get_actions_info(),
            action_names=action_executor.action_names(),
            thought=self.thought['begin'],
            action=self.action['begin'],
            action_input=self.action_input['begin'],
            response=self.response['begin'],
            finish=self.finish['begin'],
        )
        formatted = []
        formatted.append(dict(role='system', content=call_protocol))
        formatted += chat_history
        formatted += inner_step
        if force_stop:
            formatted.append(dict(role='system', content=self.force_stop))
        return formatted

    def parse(
        self,
        message: str,
        action_executor: ActionExecutor,
    ) -> Tuple[str, str, str]:
        """Parse the action returns in a ReAct format.

        Args:
            message (str): The response from LLM with ReAct format.
            action_executor (ActionExecutor): Action executor to
                provide no_action/finish_action name.

        Returns:
            tuple: the return value is a tuple contains:
                - thought (str): contain LLM thought of the current step.
                - action (str): contain action scheduled by LLM.
                - action_input (str): contain the required action input
                    for current action.
        """

        import re
        thought = message.split(self.action['begin'])[0]
        thought = thought.split(self.thought['begin'])[-1]
        thought = thought.split(self.finish['begin'])[0]
        if self.finish['begin'] in message:
            final_answer = message.split(self.finish['begin'])[-1]
            return thought, action_executor.finish_action.name, final_answer

        action_regex = f"{self.action['begin']}(.*?)\n"
        args_regex = f"{self.action_input['begin']}(.*)"
        action_match = re.findall(action_regex, message)
        if not action_match:
            return thought, action_executor.no_action.name, ''
        action = action_match[-1]
        arg_match = re.findall(args_regex, message, re.DOTALL)

        if not arg_match:
            return thought, action_executor.no_action.name, ''
        action_input = arg_match[-1]
        return thought, action.strip(), action_input.strip().strip('"')

    def format_response(self, action_return: ActionReturn) -> dict:
        """Format the final response at current step.

        Args:
            action_return (ActionReturn): return value of the current action.

        Returns:
            dict: the final response at current step.
        """
        if action_return.state == ActionStatusCode.SUCCESS:
            response = action_return.format_result()
        else:
            response = action_return.errmsg
        return dict(
            role='system',
            content=self.response['begin'] + response + self.response['end'])


class ReAct(BaseAgent):
    """An implementation of ReAct (https://arxiv.org/abs/2210.03629)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as backend.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReActProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReAct protocol.
            Defaults to 4.
    """

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 action_executor: ActionExecutor,
                 protocol: ReActProtocol = ReActProtocol(),
                 max_turn: int = 4) -> None:
        self.max_turn = max_turn
        super().__init__(
            llm=llm, action_executor=action_executor, protocol=protocol)

    def chat(self, message: Union[str, dict, List[dict]],
             **kwargs) -> AgentReturn:
        if isinstance(message, str):
            inner_history = [dict(role='user', content=message)]
        elif isinstance(message, dict):
            inner_history = [message]
        elif isinstance(message, list):
            inner_history = message[:]
        else:
            raise TypeError(f'unsupported type: {type(message)}')
        offset = len(inner_history)
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        for turn in range(self.max_turn):
            prompt = self._protocol.format(
                chat_history=[],
                inner_step=inner_history,
                action_executor=self._action_executor,
                force_stop=(turn == self.max_turn - 1))
            response = self._llm.chat(prompt, **kwargs)
            inner_history.append(dict(role='assistant', content=response))
            thought, action, action_input = self._protocol.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            action_return.thought = thought
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.format_result()
                break
            inner_history.append(self._protocol.format_response(action_return))
        else:
            agent_return.response = default_response
        agent_return.inner_steps = inner_history[offset:]
        return agent_return
