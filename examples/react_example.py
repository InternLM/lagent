from etangent.actions.action_executor import ActionExecutor
from etangent.actions.python_interpreter import PythonInterpreter
from etangent.agents.react import ReACT
from etangent.llms.openai import GPTAPI


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main():
    model = GPTAPI(model_type='gpt-3.5-turbo', )

    chatbot = ReACT(
        llm=model,
        action_executor=ActionExecutor(actions=[PythonInterpreter()]),
    )

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


if __name__ == '__main__':
    main()
