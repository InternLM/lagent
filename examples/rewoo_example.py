from etangent.actions.action_executor import ActionExecutor
from etangent.actions.llm_qa import LLMQA
from etangent.actions.serper_search import SerperSearch
from etangent.agents.rewoo import ReWOO
from etangent.llms.openai import GPTAPI

model = GPTAPI(model_type='gpt-3.5-turbo')

chatbot = ReWOO(
    llm=model,
    action_executor=ActionExecutor(
        actions=[LLMQA(model), SerperSearch(api_key=None)]),
)


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


prompt = 'What profession does Nicholas Ray and Elia Kazan have in common'

agent_return = chatbot.chat(prompt)
print(agent_return.response)
