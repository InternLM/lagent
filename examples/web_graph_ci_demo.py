from datetime import datetime

from lagent.actions import ActionExecutor
from lagent.actions import BingBrowser
from lagent.agents.mindsearch_agent import MindSearchAgent, MindSearchProtocol
from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_CN, searcher_input_template_cn, searcher_system_prompt_cn
from lagent.llms import INTERNLM2_META, GPTAPI

url = 'https://puyu.staging.openxlab.org.cn/puyu/api/v1/chat/completions'
llm = GPTAPI(
    model_type='internlm2.5-api-7b-0627',
    meta_template=[
        dict(role='system', api_role='system'),
        dict(role='user', api_role='user'),
        dict(role='assistant', api_role='assistant'),
        dict(role='environment', api_role='environment'),
    ],
    openai_api_base=url,
    key='YOUR API KEY',
    stop_words=['<|im_end|>'],
)

agent = MindSearchAgent(
    llm=llm,
    searcher_cfg=dict(
        llm=llm,
        plugin_executor=ActionExecutor(BingBrowser('YOUR API KEY')),
        protocol=MindSearchProtocol(
            meta_prompt=datetime.now().strftime(
                'The current date is %Y-%m-%d.'),
            plugin_prompt=searcher_system_prompt_cn,
        ),
        template=searcher_input_template_cn),
    protocol=MindSearchProtocol(
        meta_prompt=datetime.now().strftime('The current date is %Y-%m-%d.'),
        interpreter_prompt=GRAPH_PROMPT_CN,
        response_prompt='请根据上文内容对问题给出详细的回复'),
    max_turn=10)

for agent_return in agent.stream_chat('上海今天天气怎么样？'):
    print(agent_return)
