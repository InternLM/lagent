from lagent.actions import LLMQA, ActionExecutor, GoogleSearch
from lagent.agents import ReWOO
from lagent.llms.openai import GPTAPI

# set OPEN_API_KEY in your environment or directly pass it with key=''
model = GPTAPI(model_type='gpt-3.5-turbo')
# please set the serper search API key
search_tool = GoogleSearch(api_key='SERPER_API_KEY')
llmqa_tool = LLMQA(model)

chatbot = ReWOO(
    llm=model,
    action_executor=ActionExecutor(actions=[llmqa_tool, search_tool]),
)

prompt = 'What profession does Nicholas Ray and Elia Kazan have in common'

agent_return = chatbot.chat(prompt)
print(agent_return.response)
