import tiktoken


class SimpleTokenizer:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        pass

    def get_token_num(self, content: str):
        # TODO
        enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(content)

        return len(tokens)
