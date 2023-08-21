from lagent.actions.action_executor import ActionExecutor
from lagent.actions.builtin_actions import FinishAction
from lagent.actions.python_interpreter import PythonInterpreter
from lagent.agents.autogpt import AutoGPT
from lagent.llms.openai import GPTAPI


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main():
    # set OPEN_API_KEY in your environment or directly pass it with key=''
    model = GPTAPI(model_type='gpt-3.5-turbo')

    chatbot = AutoGPT(
        llm=model,
        action_executor=ActionExecutor(
            actions=[
                PythonInterpreter(),
            ],
            finish_action=FinishAction(
                description=(
                    'Goals are accomplished and there is nothing left '
                    'to do. Parameter: {"response: "final response '
                    'for the goal"}')),
            finish_in_action=True),
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
