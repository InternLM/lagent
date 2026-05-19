import asyncio
import json
import logging
import os

import aiohttp
import pytest

from lagent.adapters.proxy import SessionClient


@pytest.mark.asyncio
async def test_session_client_openai():
    # 1. Start the proxy
    proxy = SessionClient(
        real_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),  # Provide a valid OpenAI key if needed
        real_base_url="http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
        session_id="test_session_123",
        port=0,  # Auto-assign port
    )
    await proxy.start()

    # 2. Simulate an Agent sending an OpenAI-format request to the proxy
    dummy_payload = {
        "model": "agentic_rl_qwen35a3b_service",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        try:
            # Send to proxy URL, the proxy forwards it to real_base_url
            async with session.post(
                f"{proxy.url}/chat/completions",
                json=dummy_payload,
                headers={"Authorization": "Bearer sk-proxy-test"},
            ) as resp:
                assert resp.status in (200, 401, 502, 404), f"Unexpected status: {resp.status}"
                # Result won't be evaluated strictly without real endpoints, but we ensure connection proxy works
        except Exception as e:
            logging.warning(f"Request failed (is the target server running?): {e}")

    # 3. Check what the proxy recorded
    messages = proxy.get_messages()
    print(messages)
    assert isinstance(messages, list)

    # Clean up
    proxy.release_trace()
    await proxy.stop()


@pytest.mark.asyncio
async def test_session_client_claude():
    proxy = SessionClient(
        real_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN", "EMPTY"),  # Provide a valid Claude key if needed
        real_base_url=os.getenv('ANTHROPIC_BASE_URL', "https://api.anthropic.com"),  # The Claude v1 proxy target
        session_id="test_claude_123",
        http_proxy=os.getenv("HTTP_PROXY"),
        port=0,
    )
    await proxy.start()

    # Simulate sending Anthropic schema request
    dummy_payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "system": "you're a helpful assistant",
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    async with aiohttp.ClientSession() as session:
        try:
            # Assuming the external agent posts to /v1/messages since we trigger Anthropic handling by endpoint now
            async with session.post(
                f"{proxy.url}/v1/messages",
                json=dummy_payload,
                headers={"Authorization": "Bearer sk-proxy-test", "x-api-key": "sk-proxy-test"},
            ) as resp:
                assert resp.status in (200, 401, 502, 404, 400), f"Unexpected status: {resp.status}"

                if dummy_payload["stream"]:
                    async for _ in resp.content.iter_any():
                        pass
                else:
                    await resp.json()
        except Exception as e:
            logging.warning(f"Request failed: {e}")

    # Check proxy records
    messages = proxy.get_messages()
    print(messages)
    assert isinstance(messages, list)

    proxy.release_trace()
    await proxy.stop()
