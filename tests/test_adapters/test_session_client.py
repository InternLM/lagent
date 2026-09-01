import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from unittest.mock import Mock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lagent.adapters.proxy import SessionClient, _classify_generation_endpoint


@pytest.mark.parametrize(
    ('method', 'path', 'expected'),
    [
        ('POST', 'v1/messages', 'anthropic_messages'),
        ('POST', '/gateway/v1/messages?beta=true', 'anthropic_messages'),
        ('POST', '/v1/messages/count_tokens', None),
        ('POST', '/v1/chat/completions', 'openai_chat_completions'),
        ('POST', '/prefix/v1/responses', 'openai_responses'),
        ('HEAD', '/api/hello', None),
        ('GET', '/v1/models', None),
        ('GET', '/v1/messages', None),
        ('POST', '/terminate', None),
        ('POST', '/update_weights', None),
        ('POST', '/sleep', None),
        ('POST', '/wakeup', None),
    ],
)
def test_generation_endpoint_classification(method, path, expected):
    assert _classify_generation_endpoint(method, path) == expected


@asynccontextmanager
async def _running_session_client(response_for_path, *, extra_body=None):
    calls = []

    async def upstream_handler(request):
        calls.append(
            {
                'method': request.method,
                'path': request.path,
                'query': request.query_string,
                'headers': dict(request.headers),
                'body': await request.read(),
            }
        )
        status, headers, body = response_for_path(request.path)
        return web.Response(status=status, headers=headers, body=body)

    upstream_app = web.Application()
    upstream_app.router.add_route('*', '/{path:.*}', upstream_handler)
    upstream = TestServer(upstream_app)
    await upstream.start_server()
    proxy = SessionClient(
        real_api_key='upstream-secret',
        real_base_url=str(upstream.make_url('')).rstrip('/'),
        session_id='trace-session',
        extra_body=extra_body,
        port=0,
    )
    await proxy.start()
    try:
        yield proxy, calls
    finally:
        await proxy.stop()
        await upstream.close()


@pytest.mark.asyncio
async def test_count_tokens_is_byte_transparent_and_unrecorded():

    def response_for_path(_path):
        return 200, {'Content-Type': 'application/json'}, b'{"input_tokens":17}'

    raw_request = (
        b'{"model":"test-model","messages":[{"role":"user","content":"hello"}],'
        b'"system":"system","tools":[{"name":"tool","input_schema":{"type":"object"}}]}'
    )
    headers = {
        'Authorization': 'Bearer client-secret',
        'x-api-key': 'client-secret',
        'anthropic-version': '2024-01-01',
        'anthropic-beta': 'unknown-beta',
        'Content-Type': 'application/json',
    }

    async with _running_session_client(response_for_path, extra_body={'return_token_ids': True}) as (proxy, calls):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{proxy.url}/v1/messages/count_tokens?beta=true', data=raw_request, headers=headers
            ) as response:
                assert response.status == 200
                assert await response.read() == b'{"input_tokens":17}'

        assert len(calls) == 1
        assert calls[0]['body'] == raw_request
        assert calls[0]['query'] == 'beta=true'
        assert calls[0]['headers']['Authorization'] == 'Bearer upstream-secret'
        assert calls[0]['headers']['x-api-key'] == 'upstream-secret'
        assert calls[0]['headers']['anthropic-version'] == '2023-06-01'
        assert calls[0]['headers']['anthropic-beta'] == 'unknown-beta'
        assert proxy.get_messages() == []


@pytest.mark.asyncio
async def test_generation_endpoints_are_injected_and_recorded():
    responses = {
        '/v1/messages': b'{"type":"message","role":"assistant","content":[]}',
        '/v1/chat/completions': b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}',
        '/v1/responses': b'{"output":[{"type":"message","role":"assistant","content":[]}]}',
    }

    def response_for_path(path):
        return 200, {'Content-Type': 'application/json'}, responses[path]

    async with _running_session_client(response_for_path, extra_body={'return_token_ids': True}) as (proxy, calls):
        proxy._build_anthropic_record = Mock(return_value=([{'role': 'assistant', 'content': 'a'}], None))
        proxy._build_openai_record = Mock(return_value=([{'role': 'assistant', 'content': 'b'}], None))
        proxy._build_responses_record = Mock(return_value=([{'role': 'assistant', 'content': 'c'}], None))
        requests = [
            ('/v1/messages', {'model': 'm', 'messages': [{'role': 'user', 'content': 'a'}]}),
            ('/v1/chat/completions', {'model': 'm', 'messages': [{'role': 'user', 'content': 'b'}]}),
            ('/v1/responses', {'model': 'm', 'input': 'c'}),
        ]

        async with aiohttp.ClientSession() as session:
            for path, payload in requests:
                async with session.post(f'{proxy.url}{path}', json=payload) as response:
                    assert response.status == 200
                    await response.read()

        assert len(calls) == 3
        forwarded = [json.loads(call['body']) for call in calls]
        assert all(body['session_id'] == 'trace-session' for body in forwarded)
        assert all(body['return_token_ids'] is True for body in forwarded)
        assert forwarded[0]['provider'] == 'anthropic'
        assert 'provider' not in forwarded[1]
        assert 'provider' not in forwarded[2]
        proxy._build_anthropic_record.assert_called_once()
        proxy._build_openai_record.assert_called_once()
        proxy._build_responses_record.assert_called_once()
        assert len(proxy.get_messages()) == 3


@pytest.mark.asyncio
async def test_non_generation_endpoints_are_proxied_without_recording():

    def response_for_path(_path):
        return 200, {'Content-Type': 'application/json'}, b'{"ok":true}'

    async with _running_session_client(response_for_path) as (proxy, calls):
        async with aiohttp.ClientSession() as session:
            async with session.head(f'{proxy.url}/api/hello') as response:
                assert response.status == 200
            for method, path in [
                ('GET', '/v1/models'),
                ('GET', '/v1/messages'),
                ('POST', '/terminate'),
                ('POST', '/update_weights'),
                ('POST', '/sleep'),
                ('POST', '/wakeup'),
                ('POST', '/future/api'),
            ]:
                async with session.request(method, f'{proxy.url}{path}', data=b'super-secret-body') as response:
                    assert response.status == 200
        assert [(call['method'], call['path']) for call in calls] == [
            ('HEAD', '/api/hello'),
            ('GET', '/v1/models'),
            ('GET', '/v1/messages'),
            ('POST', '/terminate'),
            ('POST', '/update_weights'),
            ('POST', '/sleep'),
            ('POST', '/wakeup'),
            ('POST', '/future/api'),
        ]
        assert all(call['body'] == b'super-secret-body' for call in calls[1:])
        assert proxy.get_messages() == []


def test_get_messages_normalizes_openclaw_tool_call_id_underscore_loss():
    proxy = SessionClient(
        real_api_key="EMPTY",
        real_base_url="http://example.test/v1",
        session_id="openclaw_id_replay",
    )
    short = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a45f453d817e4c0cbc5ec29d", "content": "/app"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": None,
    }
    long = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "calla45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "calla45f453d817e4c0cbc5ec29d", "content": "/app"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ],
        "tools": None,
    }
    proxy._records[proxy.session_id] = [short, long]

    assert proxy.get_messages() == [long]


def test_get_messages_keeps_distinct_tool_call_ids_distinct():
    proxy = SessionClient(
        real_api_key="EMPTY",
        real_base_url="http://example.test/v1",
        session_id="distinct_tool_ids",
    )
    first = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
        ],
        "tools": None,
    }
    second = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_b45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "user", "content": "next"},
        ],
        "tools": None,
    }
    proxy._records[proxy.session_id] = [first, second]

    assert proxy.get_messages() == [first, second]


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
