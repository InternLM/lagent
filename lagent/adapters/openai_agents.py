"""OpenAI Agents SDK adapter — wraps openai-agents as a lagent Agent.

Demonstrates the SDKAgentAdapter pattern with a real framework.
Supports real multi-turn via RunResult chaining, and Proxy for
LLM trace capture.

Usage::

    from lagent.adapters.openai_agents import OpenAIAgentsAdapter

    agent = OpenAIAgentsAdapter(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        max_turns=5,
    )
    r1 = await agent("What is 2+2?")
    r2 = await agent("Now multiply that by 3")  # real multi-turn
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from .sdk_adapter import SDKAgentAdapter


class OpenAIAgentsAdapter(SDKAgentAdapter):
    """Wraps OpenAI Agents SDK as a lagent Agent.

    Real multi-turn is achieved by passing the previous ``RunResult``
    as input to the next ``Runner.run()`` call, so the agent sees
    the full conversation history.

    Args:
        model: Model name (e.g. "gpt-4o-mini"). Required.
        instructions: System instructions for the agent.
        max_turns: Max turns per call. Default: 10.
        agent_name: Name for the OpenAI Agent. Default: "assistant".
        **kwargs: Passed to SDKAgentAdapter.
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        instructions: Optional[str] = None,
        max_turns: int = 10,
        tools: Optional[list] = None,
        mcp_servers: Optional[list] = None,
        agent_name: str = 'assistant',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_proxy: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'openai-agents')
        kwargs.setdefault('description', f'OpenAI Agents SDK ({model})')
        kwargs.setdefault('sdk_module', 'agents')
        kwargs.setdefault(
            'sdk_config',
            {
                'model': model,
                'instructions': instructions or 'You are a helpful assistant.',
                'max_turns': max_turns,
                'tools': tools or [],
                'mcp_servers': mcp_servers or [],
                'agent_name': agent_name,
                'api_key': api_key or os.environ.get('OPENAI_API_KEY', ''),
                'base_url': base_url or os.environ.get('OPENAI_BASE_URL', ''),
                'http_proxy': http_proxy,
            },
        )
        super().__init__(**kwargs)
        self._last_result: Any = None

    def create_sdk_agent(self, config: Dict[str, Any]) -> Any:
        from agents import Agent
        from agents.model_settings import ModelSettings
        from openai.types import Reasoning

        return Agent(
            name=config.get('agent_name', 'assistant'),
            model=config['model'],
            model_settings=ModelSettings(reasoning=Reasoning(summary="auto")),
            instructions=config.get('instructions', 'You are a helpful assistant.'),
            tools=config.get('tools'),
            mcp_servers=config.get('mcp_servers'),
        )

    def invoke_sdk_agent(self, agent: Any, task: str, **kwargs) -> str:
        # Sync fallback — use async path instead
        raise NotImplementedError("Use invoke_sdk_agent_async")

    async def invoke_sdk_agent_async(self, agent: Any, task: str, **kwargs) -> str:
        import httpx
        from agents import RunConfig, Runner
        from agents.models.openai_provider import OpenAIProvider
        from openai import AsyncOpenAI

        config = self.sdk_config

        # Build OpenAI client with optional proxy and base_url
        client_kwargs = {}
        if config.get('api_key'):
            client_kwargs['api_key'] = config['api_key']
        if config.get('base_url'):
            client_kwargs['base_url'] = config['base_url']
        if config.get('http_proxy'):
            client_kwargs['http_client'] = httpx.AsyncClient(proxy=config['http_proxy'])

        # Override with LLM proxy if present
        if self.proxy:
            session_key = f"sk-proxy-{self.session_id}"
            client_kwargs['api_key'] = session_key
            client_kwargs['base_url'] = self.proxy.openai_base_url
            # No http_proxy needed for local proxy
            client_kwargs.pop('http_client', None)

        client = AsyncOpenAI(**client_kwargs)
        run_config = RunConfig(
            model=config.get('model'),
            model_provider=OpenAIProvider(openai_client=client),
        )

        # Multi-turn: pass previous result as input
        if self._last_result is not None:
            input_data = self._last_result.to_input_list() + [{"role": "user", "content": task}]
        else:
            input_data = task

        result = await Runner.run(
            starting_agent=agent, input=input_data, max_turns=config.get('max_turns', 10), run_config=run_config
        )

        self._last_result = result
        return result.final_output
