from lagent.actions.action_executor import ActionExecutor
from lagent.actions.llm_qa import LLMQA
from lagent.actions.serper_search import SerperSearch
from lagent.agents.rewoo import ReWOO
from lagent.llms.openai import GPTAPI

model = GPTAPI(model_type='gpt-3.5-turbo')
# please set the serper search API key
search_tool = SerperSearch(api_key=None)
llmqa_tool = LLMQA(model)

chatbot = ReWOO(
    llm=model,
    action_executor=ActionExecutor(actions=[llmqa_tool, search_tool]),
)

prompt = 'What profession does Nicholas Ray and Elia Kazan have in common'

agent_return = chatbot.chat(prompt)
print(agent_return.response)
