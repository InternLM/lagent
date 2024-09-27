import importlib
import sys
from typing import Dict

import ray

from lagent.schema import AgentMessage
from lagent.utils import load_class_from_string


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
        self.agent_actor = AsyncAgentActor.remote(**config)

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
