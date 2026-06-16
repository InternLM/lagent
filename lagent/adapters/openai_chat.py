"""OpenAI Chat adapter — wraps openai SDK as a lagent Agent.

Uses the raw openai ChatCompletion API (no openai-agents dependency).
Supports real multi-turn by maintaining messages history internally,
and Proxy for LLM trace capture.

Usage::

    from lagent.adapters.openai_chat import OpenAIChatAdapter

    agent = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="sk-...",
        base_url="http://...",
    )
    r1 = await agent("What is 2+2?")
    r2 = await agent("Now multiply that by 3")  # real multi-turn
"""

import os
from typing import Any, Dict, List, Optional

from .sdk_adapter import SDKAgentAdapter


class OpenAIChatAdapter(SDKAgentAdapter):
    """Wraps OpenAI ChatCompletion API as a lagent Agent.

    Real multi-turn is achieved by maintaining the messages list
    internally — each call appends the new user message and assistant
    response, so the full history is sent on every LLM call.

    Args:
        model: Model name. Required.
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: API base URL. Falls back to OPENAI_BASE_URL env var.
        http_proxy: HTTP proxy URL for outbound connections.
        system_prompt: System instructions.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per call.
        **kwargs: Passed to SDKAgentAdapter.
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_proxy: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'openai-chat')
        kwargs.setdefault('description', f'OpenAI Chat ({model})')
        kwargs.setdefault('sdk_module', 'openai')
        kwargs.setdefault('sdk_config', {
            'model': model,
            'api_key': api_key or os.environ.get('OPENAI_API_KEY', ''),
            'base_url': base_url or os.environ.get('OPENAI_BASE_URL', ''),
            'http_proxy': http_proxy,
            'system_prompt': system_prompt or 'You are a helpful assistant.',
            'temperature': temperature,
            'max_tokens': max_tokens,
        })
        super().__init__(**kwargs)
        self._messages: List[dict] = []

    def create_sdk_agent(self, config: Dict[str, Any]) -> Any:
        from openai import AsyncOpenAI
        import httpx

        client_kwargs = {}
        if config.get('api_key'):
            client_kwargs['api_key'] = config['api_key']
        if config.get('base_url'):
            client_kwargs['base_url'] = config['base_url']
        if config.get('http_proxy'):
            client_kwargs['http_client'] = httpx.AsyncClient(
                proxy=config['http_proxy']
            )
        return AsyncOpenAI(**client_kwargs)

    def invoke_sdk_agent(self, agent: Any, task: str, **kwargs) -> str:
        raise NotImplementedError("Use invoke_sdk_agent_async")

    async def invoke_sdk_agent_async(self, agent: Any, task: str, **kwargs) -> str:
        config = self.sdk_config

        # Initialize system prompt on first call
        if not self._messages and config.get('system_prompt'):
            self._messages.append({
                'role': 'system',
                'content': config['system_prompt'],
            })

        # Append user message
        self._messages.append({'role': 'user', 'content': task})

        # If proxy is active, create a separate client pointing to proxy
        if self.proxy:
            from openai import AsyncOpenAI
            session_key = f"sk-proxy-{self.session_id}"
            client = AsyncOpenAI(
                api_key=session_key,
                base_url=self.proxy.openai_base_url,
            )
        else:
            client = agent  # use the original client

        # Call LLM
        call_kwargs = {
            'model': config['model'],
            'messages': list(self._messages),
        }
        if config.get('temperature') is not None:
            call_kwargs['temperature'] = config['temperature']
        if config.get('max_tokens') is not None:
            call_kwargs['max_tokens'] = config['max_tokens']

        response = await client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content or ''

        # Append assistant response for multi-turn
        self._messages.append({'role': 'assistant', 'content': content})

        return content
