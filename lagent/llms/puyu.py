from typing import Dict, List, Union
from lagent.llms.base_api import BaseAPILLM
import requests

class puyuAPI(BaseAPILLM):
    """自定义的 API LLM 类，用于调用外部 API 进行文本生成。"""

    def __init__(self, model_type, meta_template=None, **gen_params):
        super().__init__(model_type, meta_template=meta_template, **gen_params)

    def call_api(self, messages):
        """调用外部 API 并返回响应结果。"""
        url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            "Authorization": "Bearer " + YOUR_TOKEN_HERE  
        }
        data = {
            "model": self.model_type,
            "messages": messages,
            "n": 1,
            "temperature": self.gen_params['temperature'],
            "top_p": self.gen_params['top_p'],
            "stream": False,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API 调用失败，状态码: {response.status_code}")

    def generate(self, inputs: Union[str, List[str]], **gen_params) -> Union[str, List[str]]:
        """调用外部 API。"""
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            inputs = [{"role": "user", "content": text} for text in inputs]

        # 调用 call_api 并返回响应
        response = self.call_api(inputs)
        content = response["choices"][0]["message"]["content"]

        if len(inputs) == 1:
            return content
        else:
            return [content]
