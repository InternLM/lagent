import copy
import importlib
import sys
from typing import Dict, Optional

import ray

from lagent.schema import AgentMessage
from lagent.utils import load_class_from_string


class _StatelessAgentWrapper:
    """Wrapper that runs inside a Ray actor.  Holds a *template* agent and
    creates a fresh instance (via ``new_instance``) per call so that
    concurrent ``__call__`` invocations are isolated from each other while
    sharing heavy resources (llm, actions, connections)."""

    def __init__(self, agent):
        self.agent = agent

    async def stateless_call(self, messages, state_dict=None, **kwargs):
        """Create a new agent, optionally load state, run, return result + new state."""
        agent = self.agent.new_instance()
        if state_dict:
            agent.load_state_dict(state_dict)
        response = await agent(*messages, **kwargs)
        return {'response': response, 'state_dict': agent.state_dict()}

    def sync_stateless_call(self, messages, state_dict=None, **kwargs):
        agent = self.agent.new_instance()
        if state_dict:
            agent.load_state_dict(state_dict)
        response = agent(*messages, **kwargs)
        return {'response': response, 'state_dict': agent.state_dict()}

    def get_state_dict(self):
        """Return a fresh empty state_dict from the template agent."""
        agent = self.agent.new_instance()
        return agent.state_dict()


class AsyncAgentRayActor:
    """Stateless async Ray actor wrapper for agents.

    Each call can carry a ``state_dict`` to restore the agent's state before
    execution.  The response includes the updated ``state_dict``.
    Heavy resources (llm, actions, connections) are shared across calls
    inside the actor while memory is isolated per request.
    """

    def __init__(
        self,
        config: Dict,
        num_gpus: int,
    ):
        config = copy.deepcopy(config)
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name

        # The wrapper holds the template agent; Ray actor wraps the wrapper
        WrappedActor = ray.remote(num_gpus=num_gpus)(_StatelessAgentWrapper)
        template_agent = cls_name(**config)
        self.actor = WrappedActor.remote(template_agent)

    async def __call__(
        self,
        *message: AgentMessage,
        state_dict: Optional[Dict] = None,
        **kwargs,
    ):
        """Run the remote agent with optional state round-trip.

        Returns
        -------
        dict  ``{'response': AgentMessage, 'state_dict': dict}``
        """
        return await self.actor.stateless_call.remote(
            list(message), state_dict=state_dict, **kwargs)

    async def state_dict(self) -> Dict:
        return await self.actor.get_state_dict.remote()

    async def reset(self):
        """No-op for stateless wrapper – each call already starts fresh."""
        pass


class AgentRayActor:
    """Stateless sync Ray actor wrapper for agents."""

    def __init__(
        self,
        config: Dict,
        num_gpus: int,
    ):
        config = copy.deepcopy(config)
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name

        WrappedActor = ray.remote(num_gpus=num_gpus)(_StatelessAgentWrapper)
        template_agent = cls_name(**config)
        self.actor = WrappedActor.remote(template_agent)

    def __call__(
        self,
        *message: AgentMessage,
        state_dict: Optional[Dict] = None,
        **kwargs,
    ):
        """Run the remote agent with optional state round-trip."""
        return ray.get(self.actor.sync_stateless_call.remote(
            list(message), state_dict=state_dict, **kwargs))

    def state_dict(self) -> Dict:
        return ray.get(self.actor.get_state_dict.remote())

    def reset(self):
        """No-op for stateless wrapper."""
        pass
