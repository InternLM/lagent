from typing import Dict, List, Union

from pydantic import BaseModel, Field

from lagent.actions import ActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.agents.agent import Agent
from lagent.prompts.parsers.json_parser import JSONParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.registry import AGENT_REGISTRY, ObjectFactory
from lagent.schema import AgentMessage

select_action_template = """你是一个可以调用外部工具的助手，可以使用的工具包括：
{action_info}
{output_format}
开始!"""

output_format_template = """如果使用工具请遵循以下格式回复：
{function_format}

如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
{finish_format}"""

summary_template = """你已经完成了所有的搜索，总结你的搜索结果。"""


class ReAct(Agent):

    def __init__(self,
                 action_select_agent: Union[Dict, Agent],
                 summary_agent: Union[Dict, Agent],
                 actions: Union[BaseAction, List[BaseAction]],
                 max_turn: int = 5,
                 **kwargs):
        self.max_turn = max_turn
        self.action_select_agent = ObjectFactory.create(
            action_select_agent, AGENT_REGISTRY)
        self.summary_agent = ObjectFactory.create(summary_agent,
                                                  AGENT_REGISTRY)
        self.action_manager = ActionExecutor(actions, hooks=[ParameterHook()])
        super().__init__(**kwargs)

    def forward(self, message: AgentMessage):
        for _ in range(self.max_turn):
            message = self.action_select_agent(message)
            if isinstance(message.content, FinishFormat):
                return message
            message = self.action_manager(message)

        return self.summary_agent(
            AgentMessage(
                sender=self.name,
                content=self.action_select_agent.memory.get_memory()))


if __name__ == '__main__':
    from lagent.actions import PythonInterpreter
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

    class ParameterHook:

        def pre_forward(self, message):
            content = message.content
            content = content.action.model_dump()
            content['command'] = content.pop('parameters')
            message.content = content
            return message

        def post_forward(self, message):
            content = message.content
            content = str(content.result or content.errmsg)
            message.content = content
            return message

    llm = dict(
        type=GPTAPI,
        model_type='gpt-4o-2024-05-13',
        query_per_second=50,
        max_new_tokens=4096,
        retry=1000)
    action = PythonInterpreter()
    variables = {
        'action_info': 'Searcher 助手是一个强大的搜索引擎，可以帮助标注员快速获取信息。',
    }
    prompt = prompt_template.format(
        output_format=output_format.format(), action_info=action.description)
    action_select_agent = dict(
        type=Agent,
        llm=llm,
        template=prompt,
        output_format=output_format,
        aggregator=dict(type='DefaultAggregator'),
    )

    summary_agent = dict(
        type=Agent,
        name='SelectNextActionAgent',
        llm=llm,
        template=prompt,
        aggregator=dict(type='DefaultAggregator'),
    )
    agent = ReAct(
        action_select_agent,
        summary_agent,
        actions=[action],
    )
    response = agent(
        AgentMessage(sender='user', content='用 Python 计算一下 3 ** 5'))
    print(response)
    response = agent(AgentMessage(sender='user', content=' 2 ** 5 呢'))
    print(response)
