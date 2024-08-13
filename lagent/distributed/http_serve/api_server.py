# server.py
import json
import os
import subprocess
import sys
import time

import aiohttp

from lagent.llms import INTERNLM2_META
from lagent.schema import AgentMessage


class AsyncHTTPAgentServer:

    def __init__(self, gpu_id, config, host='0.0.0.0', port=8090):
        self.gpu_id = gpu_id
        self.config = config
        self.host = host
        self.port = port
        self.start_server()

    def start_server(self):
        # set CUDA_VISIBLE_DEVICES in subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.process = subprocess.Popen([
            sys.executable, 'lagent/distributed/http_serve/app.py', '--host',
            self.host, '--port',
            str(self.port), '--config',
            json.dumps(self.config)
        ],
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

    async def __call__(self, message: AgentMessage, session_id: str = 0):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f'http://localhost:{self.port}/chat_completion',
                    data=message.model_dump_json(),
                    headers={'Content-Type': 'application/json'},
            ) as response:
                response = await response.json()
                return response

    def shutdown(self):
        self.process.terminate()
        self.process.wait()


# class HTTPAgentServer:

#     def __init__(self, gpu_id, config, host='0.0.0.0', port=8090):

if __name__ == '__main__':

    server = AsyncHTTPAgentServer(
        '1', {
            'type': 'lagent.agents.AsyncAgent',
            'llm': {
                'type': 'AsyncLMDeployPipeline',
                'model_name': 'internlm2',
                'path':
                '/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2',
                'meta_template': INTERNLM2_META,
            }
        })
    message = AgentMessage(sender='user', content='hello')
    result = server(message)
    import asyncio
    print(asyncio.run(result))
