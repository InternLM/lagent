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
    """Stateless Agent API Server.

    Each request can optionally carry a ``state_dict`` so the server restores
    agent state before execution and returns the updated state afterwards.
    The server itself holds only a *template* agent used for creating fresh
    instances via ``load_state_dict``.

    API routes
    ----------
    GET  /health_check          – liveness probe
    POST /chat_completion       – run agent with optional state round-trip
    POST /state_dict            – export current agent state (from given state)
    POST /load_state_dict       – validate & echo back a state_dict
    POST /reset                 – return a fresh (empty) state_dict
    """

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
        # Keep both the class and default config so we can mint new instances
        self._agent_cls = cls_name
        self._agent_config = config
        # Template agent – used as the prototype for every request
        self.agent = cls_name(**config)
        self.setup_routes()
        self.run(host, port)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_agent(self, state_dict=None):
        """Create a fresh agent instance with shared heavy resources
        (llm, actions, skills) but a new empty memory, then optionally
        restore state from *state_dict*."""
        agent = self.agent.new_instance()
        if state_dict:
            agent.load_state_dict(state_dict)
        return agent

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def setup_routes(self):

        def heartbeat():
            return {'status': 'success', 'timestamp': time.time()}

        async def process_message(request: Request):
            """Run the agent.

            Request body
            ------------
            - ``message``: list of str / AgentMessage dicts  (required)
            - ``state_dict``: dict – agent state to restore before running
            - any other keys are forwarded as ``**kwargs`` to the agent

            Response
            --------
            - ``response``: the AgentMessage returned by the agent
            - ``state_dict``: updated agent state after execution
            """
            try:
                body = await request.json()
                message = [
                    m if isinstance(m, str) else AgentMessage.model_validate(m)
                    for m in body.pop('message')
                ]
                incoming_state = body.pop('state_dict', None)
                agent = self._prepare_agent(incoming_state)
                result = await agent(*message, **body)
                return {
                    'response': result,
                    'state_dict': agent.state_dict(),
                }
            except Exception as e:
                logging.error(f'Error processing message: {str(e)}',
                              exc_info=True)
                raise HTTPException(
                    status_code=500, detail=str(e))

        async def get_state_dict(request: Request):
            """Export agent state.

            If a ``state_dict`` is provided in the body, load it first then
            re-export (useful for normalisation).  Otherwise return a fresh
            empty state.
            """
            try:
                body = await request.json() if request.headers.get(
                    'content-length', '0') != '0' else {}
                incoming_state = body.get('state_dict', None)
                agent = self._prepare_agent(incoming_state)
                return {'state_dict': agent.state_dict()}
            except Exception as e:
                logging.error(f'Error in state_dict: {str(e)}', exc_info=True)
                raise HTTPException(
                    status_code=500, detail=str(e))

        async def load_state_dict(request: Request):
            """Validate a state_dict by loading it and returning the result."""
            try:
                body = await request.json()
                state_dict = body.get('state_dict')
                if state_dict is None:
                    raise HTTPException(
                        status_code=400,
                        detail='state_dict is required')
                agent = self._prepare_agent(state_dict)
                return {
                    'status': 'success',
                    'state_dict': agent.state_dict(),
                }
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f'Error in load_state_dict: {str(e)}',
                              exc_info=True)
                raise HTTPException(
                    status_code=500, detail=str(e))

        async def reset_state():
            """Return a fresh empty state_dict."""
            try:
                agent = self._prepare_agent()
                return {'state_dict': agent.state_dict()}
            except Exception as e:
                logging.error(f'Error in reset: {str(e)}', exc_info=True)
                raise HTTPException(
                    status_code=500, detail=str(e))

        self.app.add_api_route('/health_check', heartbeat, methods=['GET'])
        self.app.add_api_route(
            '/chat_completion', process_message, methods=['POST'])
        self.app.add_api_route(
            '/state_dict', get_state_dict, methods=['POST'])
        self.app.add_api_route(
            '/load_state_dict', load_state_dict, methods=['POST'])
        self.app.add_api_route(
            '/reset', reset_state, methods=['POST'])

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
