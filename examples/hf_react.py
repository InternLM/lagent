from etangent.actions.action_executor import ActionExecutor
from etangent.actions.python import PythonExecutor
from etangent.agents.react import ReACT
from etangent.llms.huggingface import HuggingFace

model = HuggingFace(
    path='internlm/internlm-chat-7b',
    meta_template=[
        dict(role='system', begin='<|System|>:', end='<TOKENS_UNUSED_2>\n'),
        dict(role='user', begin='<|User|>:', end='<eoh>\n'),
        dict(role='assistant', begin='<|Bot|>:', end='<eoa>\n', generate=True)
    ],
)

chatbot = ReACT(
    llm=model,
    action_executor=ActionExecutor(actions=[PythonExecutor()]),
)


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


while True:
    try:
        prompt = input_prompt()
    except UnicodeDecodeError:
        print('UnicodeDecodeError')
        continue
    if prompt == 'exit':
        exit(0)

    agent_return = chatbot.chat(prompt)
    print(agent_return.response)
