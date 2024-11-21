import json
import os
import subprocess
import sys
import time
import threading

import aiohttp
import requests

from lagent.schema import AgentMessage


class HTTPAgentClient:

    def __init__(self, host='127.0.0.1', port=8090, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    @property
    def is_alive(self):
        try:
            resp = requests.get(
                f'http://{self.host}:{self.port}/health_check',
                timeout=self.timeout)
            return resp.status_code == 200
        except:
            return False

    def __call__(self, *message, session_id: int = 0, **kwargs):
        response = requests.post(
            f'http://{self.host}:{self.port}/chat_completion',
            json={
                'message': [
                    m if isinstance(m, str) else m.model_dump()
                    for m in message
                ],
                'session_id': session_id,
                **kwargs,
            },
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout)
        resp = response.json()
        if response.status_code != 200:
            return resp
        return AgentMessage.model_validate(resp)

    def state_dict(self, session_id: int = 0):
        resp = requests.get(
            f'http://{self.host}:{self.port}/memory/{session_id}',
            timeout=self.timeout)
        return resp.json()


class HTTPAgentServer(HTTPAgentClient):

    def __init__(self, gpu_id, config, host='127.0.0.1', port=8090):
        super().__init__(host, port)
        self.gpu_id = gpu_id
        self.config = config
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

        self.service_started = False

        def log_output(stream):
            if stream is not None:
                for line in iter(stream.readline, ''):
                    print(line, end='')
                    if 'Uvicorn running on' in line:
                        self.service_started = True

        # Start log output thread
        threading.Thread(target=log_output, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=log_output, args=(self.process.stderr,), daemon=True).start()

        # Waiting for the service to start
        while not self.service_started:
            time.sleep(0.1)

    def shutdown(self):
        self.process.terminate()
        self.process.wait()


class AsyncHTTPAgentMixin:

    async def __call__(self, *message, session_id: int = 0, **kwargs):
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(
                    f'http://{self.host}:{self.port}/chat_completion',
                    json={
                        'message': [
                            m if isinstance(m, str) else m.model_dump()
                            for m in message
                        ],
                        'session_id': session_id,
                        **kwargs,
                    },
                    headers={'Content-Type': 'application/json'},
            ) as response:
                resp = await response.json()
                if response.status != 200:
                    return resp
                return AgentMessage.model_validate(resp)


class AsyncHTTPAgentClient(AsyncHTTPAgentMixin, HTTPAgentClient):
    pass


class AsyncHTTPAgentServer(AsyncHTTPAgentMixin, HTTPAgentServer):
    pass
