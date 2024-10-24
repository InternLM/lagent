import argparse
import json
import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

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

        def heartbeat():
            return {'status': 'success', 'timestamp': time.time()}

        async def process_message(request: Request):
            try:
                body = await request.json()
                message = [
                    m if isinstance(m, str) else AgentMessage.model_validate(m)
                    for m in body.pop('message')
                ]
                result = await self.agent(*message, **body)
                return result
            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

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

        self.app.add_api_route('/health_check', heartbeat, methods=['GET'])
        self.app.add_api_route(
            '/chat_completion', process_message, methods=['POST'])
        self.app.add_api_route(
            '/memory/{session_id}', get_memory, methods=['GET'])

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
