from argparse import ArgumentParser

from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META


def parse_args():
    parser = ArgumentParser(description='chatbot')
    parser.add_argument(
        '--path',
        type=str,
        default='internlm/internlm2-chat-20b',
        help='The path to the model')
    parser.add_argument(
        '--mode',
        type=str,
        default='chat',
        help='Completion through chat or generate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Initialize the HFTransformer-based Language Model (llm)
    model = HFTransformer(
        path=args.path,
        meta_template=META,
        max_new_tokens=1024,
        top_p=0.8,
        top_k=None,
        temperature=0.1,
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])

    def input_prompt():
        print('\ndouble enter to end input >>> ', end='', flush=True)
        sentinel = ''  # ends when this string is seen
        return '\n'.join(iter(input, sentinel))

    history = []
    while True:
        try:
            prompt = input_prompt()
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if prompt == 'exit':
            exit(0)
        history.append(dict(role='user', content=prompt))
        if args.mode == 'generate':
            history = [dict(role='user', content=prompt)]
        print('\nInternLm2：', end='')
        current_length = 0
        for status, response, _ in model.stream_chat(history):
            print(response[current_length:], end='', flush=True)
            current_length = len(response)
        history.append(dict(role='assistant', content=response))
        print('')


if __name__ == '__main__':
    main()
