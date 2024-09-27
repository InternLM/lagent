import argparse
import importlib
import json
import logging
import sys
import time

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from lagent.schema import AgentMessage
from lagent.utils import load_class_from_string


class AgentAPIServer:

    def __init__(self,
                 config: dict,
                 host: str = '127.0.0.1',
                 port: int = 8090):
        self.app = FastAPI(docs_url='/')
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name
        self.agent = cls_name(**config)
        self.setup_routes()
        self.run(host, port)

    def setup_routes(self):

        @self.app.get('/health_check')
        def heartbeat():
            return {'status': 'success', 'timestamp': time.time()}

        @self.app.post('/chat_completion')
        async def process_message(message: AgentMessage,
                                  session_id: int = Body(0)):
            try:
                result = await self.agent(message, session_id=session_id)
                return result
            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

        @self.app.get('/memory/{session_id}')
        def get_memory(session_id: int = 0):
            try:
                result = self.agent.state_dict(session_id)
                return result
            except KeyError:
                raise HTTPException(
                    status_code=404, detail="Session ID not found")
            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

    def run(self, host='127.0.0.1', port=8090):
        logging.info(f'Starting server at {host}:{port}')
        uvicorn.run(self.app, host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='Async Agent API Server')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument(
        '--config',
        type=json.loads,
        required=True,
        help='JSON configuration for the agent')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    AgentAPIServer(args.config, host=args.host, port=args.port)
