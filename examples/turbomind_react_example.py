import argparse

from lagent.actions.action_executor import ActionExecutor
from lagent.actions.python_interpreter import PythonInterpreter
from lagent.agents.react import ReAct
from lagent.llms.lmdeploy import TurboMind

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="The path to the model")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = TurboMind(
        path=args.path,
        meta_template=[
            dict(role='system', begin='<|System|>:', end='<TOKENS_UNUSED_2>\n'),
            dict(role='user', begin='<|User|>:', end='<eoh>\n'),
            dict(role='assistant', begin='<|Bot|>:', end='<eoa>\n', generate=True)
        ],
    )

    chatbot = ReAct(
        llm=model,
        action_executor=ActionExecutor(actions=[PythonInterpreter()]),
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
