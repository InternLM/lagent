from argparse import ArgumentParser

from lagent.llms.openai import GPTAPI


def parse_args():
    parser = ArgumentParser(description='chatbot')
    parser.add_argument('--mode', default='chat')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # set OPEN_API_KEY in your environment or directly pass it with key=''
    model = GPTAPI(model_type='gpt-3.5-turbo')
    history = []
    while True:
        try:
            prompt = input('>>> ')
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if prompt == 'exit':
            exit(0)
        if args.mode == 'chat':
            history.append(dict(role='user', content=prompt))
            response = model.generate_from_template(history, max_out_len=512)
            history.append(dict(role='assistant', content=response))
        elif args.mode == 'generate':
            response = model.generate(prompt, max_out_len=512)
        print('Assistant:', response)


if __name__ == '__main__':
    main()
