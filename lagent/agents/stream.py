import json
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Union

from lagent.actions import ActionExecutor, AsyncActionExecutor, AsyncIPythonInterpreter, IPythonInteractive
from lagent.agents.agent import Agent, AsyncAgent
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.hooks import InternLMActionProcessor
from lagent.llms import BaseLLM
from lagent.memory import Memory
from lagent.prompts.parsers import InterpreterParser, MixedToolParser, PluginParser, ToolStatusCode
from lagent.schema import AgentMessage
from lagent.utils import create_object

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


def get_plugin_prompt(actions, api_desc_template=API_PREFIX):
    plugin_descriptions = []
    for action in actions if isinstance(actions, list) else [actions]:
        action = create_object(action)
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
                tool_name=action.name, description=action_desc['description'])
            action_desc['parameters'] = [
                param for param in action_desc['parameters']
                if param['name'] in action_desc['required']
            ]
            plugin_descriptions.append(action_desc)
    return json.dumps(plugin_descriptions, ensure_ascii=False, indent=4)


class AgentForInternLM(Agent):

    _INTERNAL_AGENT_CLS = Agent

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        plugins: Union[dict, List[dict]] = None,
        interpreter: dict = None,
        template: Union[str, dict, List[dict]] = None,
        memory: Dict = dict(type=Memory),
        output_format: Dict = dict(
            type=MixedToolParser,
            template=META_CN,
            parsers=[
                dict(type=PluginParser, template=PLUGIN_CN),
                dict(type=InterpreterParser, template=INTERPRETER_CN),
            ]),
        aggregator: Dict = dict(type=InternLMToolAggregator),
        action_hooks: List = [dict(type=InternLMActionProcessor)],
        finish_condition: Callable[
            [AgentMessage],
            bool] = lambda m: m.formatted['status'] == ToolStatusCode.NO_TOOL,
        max_turn: int = 4,
        **kwargs,
    ):
        agent = dict(
            type=self._INTERNAL_AGENT_CLS,
            llm=llm,
            template=template,
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=kwargs.pop('hooks', None),
        )
        self.agent = create_object(agent)
        self.plugin_executor = plugins and ActionExecutor(
            plugins, hooks=action_hooks)
        self.interpreter_executor = interpreter and ActionExecutor(
            interpreter, hooks=action_hooks)
        if not (self.plugin_executor or self.interpreter_executor):
            warnings.warn(
                'Neither plugin nor interpreter executor is initialized. '
                'An exception will be thrown when the agent call a tool.')
        self.finish_condition = finish_condition
        self.max_turn = max_turn
        super().__init__(**kwargs)

    def forward(self, message: AgentMessage, session_id=0, **kwargs):
        if isinstance(message, str):
            message = AgentMessage(sender='user', content=message)
        for _ in range(self.max_turn):
            message = self.agent(message, session_id=session_id, **kwargs)
            assert isinstance(message.formatted, dict)
            if self.finish_condition(message):
                return message
            if message.formatted['tool_type']:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f'{tool_type}_executor', None)
                if not executor:
                    raise RuntimeError(f'No available {tool_type} executor')
                message = executor(message, session_id=session_id)
        return message

    def get_steps(self, session_id=0):
        steps, tool_type = [], None
        for msg in self.agent.memory.get_memory(session_id):
            if msg.sender == self.agent.name:
                steps.append(
                    dict(role='language', content=msg.formatted['thought']))
                if msg.formatted['tool_type']:
                    tool_type = msg.formatted['tool_type']
                    steps.append(
                        dict(
                            role='tool',
                            content=msg.formatted['action'],
                            name=tool_type))
            elif msg.sender != 'user':
                feedback = dict(role='environment', content=msg.content)
                if tool_type:
                    feedback['name'] = tool_type
                steps.append(feedback)
        return steps


class MathCoder(AgentForInternLM):

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        interpreter: dict = dict(
            type=IPythonInteractive, timeout=20, max_out_len=8192),
        template: Union[str, dict, List[dict]] = None,
        memory: Dict = dict(type=Memory),
        output_format: Dict = dict(
            type=InterpreterParser,
            template=
            ('Integrate step-by-step reasoning and Python code to solve math problems '
             'using the following guidelines:\n'
             '- Analyze the question and write jupyter code to solve the problem;\n'
             r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
             'units. \n')),
        aggregator: Dict = dict(type=InternLMToolAggregator),
        action_hooks: List = [dict(type=InternLMActionProcessor)],
        finish_condition: Callable[
            [AgentMessage],
            bool] = lambda m: m.formatted['status'] == ToolStatusCode.NO_TOOL,
        max_turn: int = 6,
        **kwargs,
    ):
        kwargs.pop('plugins', None)
        super().__init__(
            llm=llm,
            interpreter=interpreter,
            template=template,
            memory=memory,
            output_format=output_format,
            aggregator=aggregator,
            action_hooks=action_hooks,
            finish_condition=finish_condition,
            max_turn=max_turn,
            **kwargs)


class AsyncAgentForInternLM(AsyncAgent):

    _INTERNAL_AGENT_CLS = AsyncAgent

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        plugins: Union[dict, List[dict]] = None,
        interpreter: dict = None,
        template: Union[str, dict, List[dict]] = None,
        memory: Dict = dict(type=Memory),
        output_format: Dict = dict(
            type=MixedToolParser,
            template=META_CN,
            parsers=[
                dict(type=PluginParser, template=PLUGIN_CN),
                dict(type=InterpreterParser, template=INTERPRETER_CN),
            ]),
        aggregator: Dict = dict(type=InternLMToolAggregator),
        action_hooks: List = [dict(type=InternLMActionProcessor)],
        finish_condition: Callable[
            [AgentMessage],
            bool] = lambda m: m.formatted['status'] == ToolStatusCode.NO_TOOL,
        max_turn: int = 4,
        **kwargs,
    ):
        agent = dict(
            type=self._INTERNAL_AGENT_CLS,
            llm=llm,
            template=template,
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=kwargs.pop('hooks', None),
        )
        self.agent = create_object(agent)
        self.plugin_executor = plugins and AsyncActionExecutor(
            plugins, hooks=action_hooks)
        self.interpreter_executor = interpreter and AsyncActionExecutor(
            interpreter, hooks=action_hooks)
        if not (self.plugin_executor or self.interpreter_executor):
            warnings.warn(
                'Neither plugin nor interpreter executor is initialized. '
                'An exception will be thrown when the agent call a tool.')
        self.finish_condition = finish_condition
        self.max_turn = max_turn
        super().__init__(**kwargs)

    async def forward(self, message: AgentMessage, session_id=0, **kwargs):
        if isinstance(message, str):
            message = AgentMessage(sender='user', content=message)
        for _ in range(self.max_turn):
            message = await self.agent(
                message, session_id=session_id, **kwargs)
            assert isinstance(message.formatted, dict)
            if self.finish_condition(message):
                return message
            if message.formatted['tool_type']:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f'{tool_type}_executor', None)
                if not executor:
                    raise RuntimeError(f'No available {tool_type} executor')
                message = await executor(message, session_id=session_id)
        return message

    def get_steps(self, session_id=0):
        steps, tool_type = [], None
        for msg in self.agent.memory.get_memory(session_id):
            if msg.sender == self.agent.name:
                steps.append(
                    dict(role='language', content=msg.formatted['thought']))
                if msg.formatted['tool_type']:
                    tool_type = msg.formatted['tool_type']
                    steps.append(
                        dict(
                            role='tool',
                            content=msg.formatted['action'],
                            name=tool_type))
            elif msg.sender != 'user':
                feedback = dict(role='environment', content=msg.content)
                if tool_type:
                    feedback['name'] = tool_type
                steps.append(feedback)
        return steps


class AsyncMathCoder(AsyncAgentForInternLM):

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        interpreter: dict = dict(type=AsyncIPythonInterpreter),
        template: Union[str, dict, List[dict]] = None,
        memory: Dict = dict(type=Memory),
        output_format: Dict = dict(
            type=InterpreterParser,
            template=
            ('Integrate step-by-step reasoning and Python code to solve math problems '
             'using the following guidelines:\n'
             '- Analyze the question and write jupyter code to solve the problem;\n'
             r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
             'units. \n')),
        aggregator: Dict = dict(type=InternLMToolAggregator),
        action_hooks: List = [dict(type=InternLMActionProcessor)],
        finish_condition: Callable[
            [AgentMessage],
            bool] = lambda m: m.formatted['status'] == ToolStatusCode.NO_TOOL,
        max_turn: int = 6,
        **kwargs,
    ):
        kwargs.pop('plugins', None)
        super().__init__(
            llm=llm,
            interpreter=interpreter,
            template=template,
            memory=memory,
            output_format=output_format,
            aggregator=aggregator,
            action_hooks=action_hooks,
            finish_condition=finish_condition,
            max_turn=max_turn,
            **kwargs)

    async def forward(self, message: AgentMessage, session_id=0, **kwargs):
        try:
            return await super().forward(message, session_id, **kwargs)
        finally:
            interpreter = next(
                iter(self.interpreter_executor.actions.values()))
            if interpreter.name == 'AsyncIPythonInterpreter':
                await interpreter.close_session(session_id)


if __name__ == '__main__':
    from lagent.llms import GPTAPI

    interpreter_prompt = (
        'Below is a math problem. Please solve it step by step with the assistance of Python programming. Consider using Sympy or Numpy library '
        'to facilitate your derivation, calculation and equation solving. Utilize the "pi" symbol and "Rational" from Sympy '
        'for $$\pi$$ and fractions, and simplify all fractions and square roots without converting them to decimal values. '
        'Please encapsulate each generated Jupyter Python code block with tags "{begin}" and "{end}". Conclude the '
        r'final answer when observations are sufficient and encapsulate the numerical result with LaTeX syntax "\boxed{{}}" '
        'without any unit, and end your conclusion with the special token "[END]" to denote the completion of your response. '
        'Keep the following points in mind:\n'
        '- You must alternately use human and programming languages in the chain of thought;\n'
        '- The number of your reasoning steps should not exceed **three**, which means you may merge some intermediate steps when the original answer is tedious.'
    )

    llm = dict(
        type=GPTAPI,
        model_type='gpt-4o-2024-05-13',
        retry=50,
        key=None,
        max_new_tokens=2048,
        stop_words=['</python'],
    )
    agent = MathCoder(
        llm=llm,
        output_format=InterpreterParser(
            template=interpreter_prompt, begin='<python>', end='</python>'),
        aggregator=InternLMToolAggregator(
            environment_begin='<output>\n', environment_end='\n</output>'),
        finish_condition=lambda m: '[END]' in m.content)
