import tiktoken


class SimpleTokenizer:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        pass

    def get_token_num(self, content: str):
        # TODO
        # 使用tiktoken加载编码器
        enc = tiktoken.get_encoding("cl100k_base")

        # 将文本分词为 token
        tokens = enc.encode(content)

        # 返回token数量
        return len(tokens)
