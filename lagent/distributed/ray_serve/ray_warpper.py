import importlib
import sys
from typing import Dict

import ray

from lagent.schema import AgentMessage


def load_class_from_string(class_path: str, path=None):
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)  # Temporarily add the path to sys.path

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(
                path)  # Ensure to clean up by removing the path from sys.path


class AsyncAgentRayActor:

    def __init__(
        self,
        config: Dict,
        num_gpus: int,
    ):
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name
        AsyncAgentActor = ray.remote(num_gpus=num_gpus)(cls_name)
        self.agent_actor = AsyncAgentActor.options(max_concurrency=100).remote(
            **config)

    async def __call__(self, *message: AgentMessage, session_id=0, **kwargs):
        response = await self.agent_actor.__call__.remote(
            *message, session_id=session_id, **kwargs)
        return response


class AgentRayActor:

    def __init__(
        self,
        config: Dict,
        num_gpus: int,
    ):
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name
        AgentActor = ray.remote(num_gpus=num_gpus)(cls_name)
        self.agent_actor = AgentActor.remote(**config)

    def __call__(self, *message: AgentMessage, session_id=0, **kwargs):
        response = self.agent_actor.__call__.remote(
            *message, session_id=session_id, **kwargs)
        return ray.get(response)
