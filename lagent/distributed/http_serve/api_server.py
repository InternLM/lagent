# server.py
import json
import os
import subprocess
import sys
import time

import aiohttp
import requests

from lagent.llms import INTERNLM2_META
from lagent.schema import AgentMessage


class HTTPAgentServer:

    def __init__(self, gpu_id, config, host='127.0.0.1', port=8090):
        self.gpu_id = gpu_id
        self.config = config
        self.host = host
        self.port = port
        self.start_server()

    def start_server(self):
        # set CUDA_VISIBLE_DEVICES in subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        cmds = [
            sys.executable, 'lagent/distributed/http_serve/app.py', '--host',
            self.host, '--port',
            str(self.port), '--config',
            json.dumps(self.config)
        ]
        self.process = subprocess.Popen(
            cmds,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)

        while True:
            output = self.process.stdout.readline()
            if not output:  # 如果读到 EOF，跳出循环
                break
            sys.stdout.write(output)  # 打印到标准输出
            sys.stdout.flush()
            if 'Uvicorn running on' in output:  # 根据实际输出调整
                break
            time.sleep(0.1)

    def __call__(self, message: AgentMessage, session_id: int = 0):
        response = requests.post(
            f'http://{self.host}:{self.port}/chat_completion',
            json={
                'message': message.model_dump(),
                'session_id': session_id
            },
            headers={'Content-Type': 'application/json'})
        return response.json()

    def state_dict(self, session_id: int = 0):
        resp = requests.get(
            f'http://{self.host}:{self.port}/memory/{session_id}')
        return resp.json()

    def shutdown(self):
        self.process.terminate()
        self.process.wait()


class AsyncHTTPAgentServer(HTTPAgentServer):

    async def __call__(self, message: AgentMessage, session_id: int = 0):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f'http://{self.host}:{self.port}/chat_completion',
                    json={
                        'message': message.model_dump(),
                        'session_id': session_id
                    },
                    headers={'Content-Type': 'application/json'},
            ) as response:
                response = await response.json()
                return response


if __name__ == '__main__':
    import asyncio

    server = HTTPAgentServer(
        '1',
        {
            'type': 'lagent.agents.AsyncAgent',
            'llm': {
                'type': 'AsyncLMDeployPipeline',
                'model_name': 'internlm2',
                'path': '/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2',
                'meta_template': INTERNLM2_META,
            }
        },
        port=8090,
    )
    message = AgentMessage(sender='user', content='hello')
    result = server(message)
    print(result)
    server.shutdown()

    # math coder
    server = AsyncHTTPAgentServer(
        '1',
        {
            'type': 'lagent.agents.AsyncMathCoder',
            'llm': {
                'type': 'AsyncLMDeployPipeline',
                'model_name': 'internlm2',
                'path': '/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2',
                'meta_template': INTERNLM2_META,
                'tp': 1,
                'top_k': 1,
                'temperature': 1.0,
                'stop_words': ['<|im_end|>', '<|action_end|>'],
                'max_new_tokens': 1024,
            },
            'interpreter': {
                'type': 'AsyncIPythonInterpreter',
                'max_kernels': 100
            },
        },
        port=8091,
    )
    message = AgentMessage(
        sender='user',
        content=
        ('Marie is thinking of a multiple of 63, while Jay is thinking of a factor '
         'of 63. They happen to be thinking of the same number. There are two '
         'possibilities for the number that each of them is thinking of, one '
         'positive and one negative. Find the product of these two numbers.'))
    result = server(message)
    print(asyncio.run(result))
    print(server.state_dict())
    server.shutdown()
