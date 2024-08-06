import warnings
from typing import Dict, List, Union

from lagent.actions import ActionExecutor, AsyncActionExecutor
from lagent.agents.agent import Agent, AsyncAgent
from lagent.llms import BaseLLM
from lagent.registry import AGENT_REGISTRY, ObjectFactory
from lagent.schema import AgentMessage, AgentStatusCode


class AgentForInternLM(Agent):

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        plugins: Union[dict, List[dict]] = None,
        interpreter: dict = None,
        memory: Dict = dict(type='Memory'),
        output_format: Dict = dict(type='InternLMToolParser'),
        aggregator: Dict = dict(type='InternLMToolAggregator'),
        action_hooks: List = [dict(type='InternLMActionProcessor')],
        max_turn: int = 4,
        **kwargs,
    ):
        agent = dict(
            type=Agent,
            llm=llm,
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=kwargs.pop('hooks', None),
        )
        self.agent = ObjectFactory.create(agent, AGENT_REGISTRY)
        self.plugin_executor = plugins and ActionExecutor(
            plugins, hooks=action_hooks)
        self.interpreter_executor = interpreter and ActionExecutor(
            interpreter, hooks=action_hooks)
        if not (self.plugin_executor or self.interpreter_executor):
            warnings.warn(
                'Neither plugin nor interpreter executor is initialized. '
                'An exception will be thrown when the agent call a tool.')
        self.max_turn = max_turn
        super().__init__(**kwargs)

    def forward(self, message: AgentMessage, session_id=0, **kwargs):
        if isinstance(message, str):
            message = AgentMessage(sender='user', content=message)
        for _ in range(self.max_turn):
            message = self.agent(message, session_id=session_id, **kwargs)
            assert isinstance(message.formatted, dict)
            if message.formatted['status'] == AgentStatusCode.END:
                return message
            if message.formatted['tool_type']:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f'{tool_type}_executor', None)
                if not executor:
                    raise RuntimeError(f'No available {tool_type} executor')
                message = executor(message)
        return message

    def get_steps(self, session_id):
        steps, tool_type = [], None
        for msg in self.agent.memory.get_memory(session_id):
            if msg.formatted:
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
            type='IPythonInteractive', timeout=20, max_out_len=8192),
        memory: Dict = dict(type='Memory'),
        output_format: Dict = dict(type='InternLMToolParser'),
        aggregator: Dict = dict(
            type='InternLMToolAggregator',
            interpreter_prompt=
            ('Integrate step-by-step reasoning and Python code to solve math problems '
             'using the following guidelines:\n'
             '- Analyze the question and write jupyter code to solve the problem;\n'
             r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
             'units. \n')),
        action_hooks: List = [dict(type='InternLMActionProcessor')],
        max_turn: int = 6,
        **kwargs,
    ):
        kwargs.pop('plugins', None)
        super().__init__(
            llm=llm,
            interpreter=interpreter,
            memory=memory,
            output_format=output_format,
            aggregator=aggregator,
            action_hooks=action_hooks,
            max_turn=max_turn,
            **kwargs)


class AsyncAgentForInternLM(AsyncAgent):

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        plugins: Union[dict, List[dict]] = None,
        interpreter: dict = None,
        memory: Dict = dict(type='Memory'),
        output_format: Dict = dict(type='InternLMToolParser'),
        aggregator: Dict = dict(type='InternLMToolAggregator'),
        action_hooks: List = [dict(type='InternLMActionProcessor')],
        max_turn: int = 4,
        **kwargs,
    ):
        agent = dict(
            type=AsyncAgent,
            llm=llm,
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=kwargs.pop('hooks', None),
        )
        self.agent = ObjectFactory.create(agent, AGENT_REGISTRY)
        self.plugin_executor = plugins and AsyncActionExecutor(
            plugins, hooks=action_hooks)
        self.interpreter_executor = interpreter and AsyncActionExecutor(
            interpreter, hooks=action_hooks)
        if not (self.plugin_executor or self.interpreter_executor):
            warnings.warn(
                'Neither plugin nor interpreter executor is initialized. '
                'An exception will be thrown when the agent call a tool.')
        self.max_turn = max_turn
        super().__init__(**kwargs)

    async def forward(self, message: AgentMessage, session_id=0, **kwargs):
        if isinstance(message, str):
            message = AgentMessage(sender='user', content=message)
        for _ in range(self.max_turn):
            message = await self.agent(
                message, session_id=session_id, **kwargs)
            assert isinstance(message.formatted, dict)
            if message.formatted['status'] == AgentStatusCode.END:
                return message
            if message.formatted['tool_type']:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f'{tool_type}_executor', None)
                if not executor:
                    raise RuntimeError(f'No available {tool_type} executor')
                message = await executor(message)
        return message

    def get_steps(self, session_id):
        steps, tool_type = [], None
        for msg in self.agent.memory.get_memory(session_id):
            if msg.formatted:
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
        interpreter: dict = dict(
            type='AsyncIPythonInteractive', timeout=20, max_out_len=8192),
        memory: Dict = dict(type='Memory'),
        output_format: Dict = dict(type='InternLMToolParser'),
        aggregator: Dict = dict(
            type='InternLMToolAggregator',
            interpreter_prompt=
            ('Integrate step-by-step reasoning and Python code to solve math problems '
             'using the following guidelines:\n'
             '- Analyze the question and write jupyter code to solve the problem;\n'
             r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
             'units. \n')),
        action_hooks: List = [dict(type='InternLMActionProcessor')],
        max_turn: int = 6,
        **kwargs,
    ):
        kwargs.pop('plugins', None)
        super().__init__(
            llm=llm,
            interpreter=interpreter,
            memory=memory,
            output_format=output_format,
            aggregator=aggregator,
            action_hooks=action_hooks,
            max_turn=max_turn,
            **kwargs)


if __name__ == '__main__':
    from lagent.agents.aggregator import InternLMToolAggregator
    from lagent.llms import GPTAPI
    from lagent.prompts.parsers.tool_parser import InternLMToolParser
    from lagent.prompts.protocols.tool_protocol import InternLMToolProtocol

    interpreter_prompt = (
        'Below is a math problem. Please solve it step by step with the assistance of Python programming. Consider using Sympy or Numpy library '
        'to facilitate your derivation, calculation and equation solving. Utilize the "pi" symbol and "Rational" from Sympy '
        'for $$\pi$$ and fractions, and simplify all fractions and square roots without converting them to decimal values. '
        'Please encapsulate each generated Jupyter Python code block with tags "<python>" and "</python>". Conclude the '
        r'final answer when observations are sufficient and encapsulate the numerical result with LaTeX syntax "\boxed{{}}" '
        'without any unit, and end your conclusion with the special token "[END]" to denote the completion of your response. '
        'Keep the following points in mind:\n'
        '- You must alternately use human and programming languages in the chain of thought;\n'
        '- The number of your reasoning steps should not exceed **three**, which means you may merge some intermediate steps when the original answer is tedious.'
    )

    protocol = InternLMToolProtocol(
        tool=dict(
            begin='{start_token}{name}\n',
            start_token='',
            name_map=dict(plugin='<|plugin|>', interpreter='<python>'),
            belong='assistant',
            end='</python>',
        ),
        execute=dict(
            role='execute',
            begin='<output>\n',
            end='\n</output>',
            fallback_role='system'))
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
        output_format=InternLMToolParser(
            finish_pattern=r'\[END\]',
            protocol=protocol,
        ),
        aggregator=InternLMToolAggregator(
            interpreter_prompt=interpreter_prompt,
            protocol=protocol,
            remove_message_name=True,
        ),
    )
