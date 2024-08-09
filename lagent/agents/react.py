import json
from typing import Dict, List, Union

from pydantic import BaseModel, Field

from lagent.actions import ActionExecutor, AsyncActionExecutor, BaseAction
from lagent.agents.agent import Agent, AsyncAgent
from lagent.llms import BaseLLM
from lagent.prompts.parsers.json_parser import JSONParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.registry import AGENT_REGISTRY, TOOL_REGISTRY, ObjectFactory
from lagent.schema import AgentMessage

select_action_template = """你是一个可以调用外部工具的助手，可以使用的工具包括：
{action_info}
{output_format}
开始!"""

output_format_template = """如果使用工具请遵循以下格式回复：
{function_format}

如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
{finish_format}"""


class ReAct(Agent):

    def __init__(self,
                 llm: Union[BaseLLM, Dict],
                 actions: Union[BaseAction, List[BaseAction]],
                 template: Union[PromptTemplate, str] = None,
                 memory: Dict = dict(type='Memory'),
                 output_format: Dict = dict(type='JsonParser'),
                 aggregator: Dict = dict(type='DefaultAggregator'),
                 hooks: List = [dict(type='ActionPreprocessor')],
                 max_turn: int = 5,
                 **kwargs):
        self.max_turn = max_turn

        actions = dict(
            type=ActionExecutor,
            actions=actions,
            hooks=hooks,
        )

        self.actions: ActionExecutor = ObjectFactory.create(
            actions, TOOL_REGISTRY)
        select_agent = dict(
            type=Agent,
            llm=llm,
            template=template.format(
                action_info=json.dumps(self.actions.description()),
                output_format=output_format.format()),
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=hooks,
        )
        self.select_agent = ObjectFactory.create(select_agent, AGENT_REGISTRY)
        super().__init__(**kwargs)

    def forward(self, message: AgentMessage, **kwargs) -> AgentMessage:
        for _ in range(self.max_turn):
            message = self.select_agent(message)
            if 'conclusion' in message.content or 'conclusion' in message.formatted:
                return message
            message = self.actions(message)

        return message


class AsyncReAct(AsyncAgent):

    def __init__(self,
                 llm: Union[BaseLLM, Dict],
                 actions: Union[BaseAction, List[BaseAction]],
                 template: Union[PromptTemplate, str] = None,
                 memory: Dict = dict(type='Memory'),
                 output_format: Dict = dict(type='JsonParser'),
                 aggregator: Dict = dict(type='DefaultAggregator'),
                 hooks: List = [dict(type='ActionPreprocessor')],
                 max_turn: int = 5,
                 **kwargs):
        self.max_turn = max_turn

        actions = dict(
            type=AsyncActionExecutor,
            actions=actions,
            hooks=hooks,
        )

        self.actions: AsyncActionExecutor = ObjectFactory.create(
            actions, TOOL_REGISTRY)
        select_agent = dict(
            type=AsyncAgent,
            llm=llm,
            template=template.format(
                action_info=json.dumps(self.actions.description()),
                output_format=output_format.format()),
            output_format=output_format,
            memory=memory,
            aggregator=aggregator,
            hooks=hooks,
        )
        self.select_agent = ObjectFactory.create(select_agent, AGENT_REGISTRY)
        super().__init__(**kwargs)

    async def forward(self, message: AgentMessage, **kwargs) -> AgentMessage:
        for _ in range(self.max_turn):
            message = await self.select_agent(message)
            if 'conclusion' in message.content or 'conclusion' in message.formatted:
                return message
            message = await self.actions(message)

        return message


if __name__ == '__main__':
    from lagent.llms import GPTAPI

    class ActionCall(BaseModel):
        name: str = Field(description='调用的函数名称')
        parameters: Dict = Field(description='调用函数的参数')

    class ActionFormat(BaseModel):
        thought_process: str = Field(
            description='描述当前所处的状态和已知信息。这有助于明确目前所掌握的信息和接下来的搜索方向。')
        action: ActionCall = Field(description='当前步骤需要执行的操作，包括函数名称和参数。')

    class FinishFormat(BaseModel):
        thought_process: str = Field(
            description='描述当前所处的状态和已知信息。这有助于明确目前所掌握的信息和接下来的搜索方向。')
        conclusion: str = Field(description='总结当前的搜索结果，回答问题。')

    prompt_template = PromptTemplate(select_action_template)
    output_format = JSONParser(
        output_format_template,
        function_format=ActionFormat,
        finish_format=FinishFormat)

    llm = dict(
        type=GPTAPI,
        model_type='gpt-4o-2024-05-13',
        key=None,
        max_new_tokens=4096,
        proxies=dict(),
        retry=1000)

    agent = ReAct(
        llm=llm,
        template=prompt_template,
        output_format=output_format,
        aggregator=dict(type='DefaultAggregator'),
        actions=[dict(type='PythonInterpreter')],
    )
    response = agent(
        AgentMessage(sender='user', content='用 Python 计算一下 3 ** 5'))
    print(response)
    response = agent(AgentMessage(sender='user', content=' 2 ** 5 呢'))
    print(response)
