import json
import os
import subprocess
import sys
import time
import threading
from typing import Dict, Optional

import aiohttp
import requests

from lagent.schema import AgentMessage


class HTTPAgentClient:
    """Stateless HTTP client for the Agent API.

    The client manages an optional local ``_state_dict`` that is automatically
    sent with each ``__call__`` and updated from the response.  This makes the
    *server* stateless while keeping a familiar stateful feel on the client
    side.
    """

    def __init__(self, host='127.0.0.1', port=8090, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._state_dict: Optional[Dict] = None

    @property
    def base_url(self):
        return f'http://{self.host}:{self.port}'

    @property
    def is_alive(self):
        try:
            resp = requests.get(
                f'{self.base_url}/health_check',
                timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core call – send state, receive new state
    # ------------------------------------------------------------------
    def __call__(self, *message, session_state: Optional[Dict] = None, **kwargs):
        """Run the remote agent.

        Parameters
        ----------
        *message : str | AgentMessage
            Messages to send.
        session_state : dict, optional
            Explicit state to use for this call.  If *None* the client's
            internal ``_state_dict`` is used (which may also be *None* for a
            fresh session).
        **kwargs
            Extra keyword arguments forwarded to the agent.

        Returns
        -------
        AgentMessage
            The agent's response.
        """
        state = session_state if session_state is not None else self._state_dict
        payload = {
            'message': [
                m if isinstance(m, str) else m.model_dump()
                for m in message
            ],
            **kwargs,
        }
        if state is not None:
            payload['state_dict'] = state

        response = requests.post(
            f'{self.base_url}/chat_completion',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout)
        resp = response.json()
        if response.status_code != 200:
            return resp

        # Update local state from server response
        self._state_dict = resp.get('state_dict', self._state_dict)
        return AgentMessage.model_validate(resp['response'])

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        """Return the current local state dict."""
        if self._state_dict is not None:
            return self._state_dict
        # Fetch a fresh empty state from the server
        resp = requests.post(
            f'{self.base_url}/state_dict',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout)
        return resp.json().get('state_dict', {})

    def load_state_dict(self, state_dict: Dict):
        """Load a state dict into the client (validated by the server)."""
        resp = requests.post(
            f'{self.base_url}/load_state_dict',
            json={'state_dict': state_dict},
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout)
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f'Failed to load state_dict: {data}')
        self._state_dict = data.get('state_dict', state_dict)

    def reset(self):
        """Reset the client to a fresh state."""
        resp = requests.post(
            f'{self.base_url}/reset',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout)
        data = resp.json()
        self._state_dict = data.get('state_dict', None)


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

    async def __call__(self, *message, session_state: Optional[Dict] = None, **kwargs):
        """Async version of the stateless agent call."""
        state = session_state if session_state is not None else self._state_dict
        payload = {
            'message': [
                m if isinstance(m, str) else m.model_dump()
                for m in message
            ],
            **kwargs,
        }
        if state is not None:
            payload['state_dict'] = state

        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(
                    f'{self.base_url}/chat_completion',
                    json=payload,
                    headers={'Content-Type': 'application/json'},
            ) as response:
                resp = await response.json()
                if response.status != 200:
                    return resp
                self._state_dict = resp.get('state_dict', self._state_dict)
                return AgentMessage.model_validate(resp['response'])

    async def async_state_dict(self) -> Dict:
        """Async version of state_dict."""
        if self._state_dict is not None:
            return self._state_dict
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(
                    f'{self.base_url}/state_dict',
                    json={},
                    headers={'Content-Type': 'application/json'},
            ) as response:
                data = await response.json()
                return data.get('state_dict', {})

    async def async_load_state_dict(self, state_dict: Dict):
        """Async version of load_state_dict."""
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(
                    f'{self.base_url}/load_state_dict',
                    json={'state_dict': state_dict},
                    headers={'Content-Type': 'application/json'},
            ) as response:
                data = await response.json()
                if response.status != 200:
                    raise RuntimeError(f'Failed to load state_dict: {data}')
                self._state_dict = data.get('state_dict', state_dict)

    async def async_reset(self):
        """Async version of reset."""
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(
                    f'{self.base_url}/reset',
                    json={},
                    headers={'Content-Type': 'application/json'},
            ) as response:
                data = await response.json()
                self._state_dict = data.get('state_dict', None)


class AsyncHTTPAgentClient(AsyncHTTPAgentMixin, HTTPAgentClient):
    pass


class AsyncHTTPAgentServer(AsyncHTTPAgentMixin, HTTPAgentServer):
    pass
