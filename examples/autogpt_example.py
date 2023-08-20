from etangent.actions.action_executor import ActionExecutor
from etangent.actions.finish_action import FinishAction
from etangent.actions.python import PythonExecutor
from etangent.agents.autogpt import AutoGPT
from etangent.llms.openai import GPTAPI


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main():
    model = GPTAPI(model_type='gpt-3.5-turbo')

    chatbot = AutoGPT(
        llm=model,
        action_executor=ActionExecutor(
            actions=[
                PythonExecutor(),
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
