"""Short real HTTP deadlines exercise the same path as long agent generations."""

import asyncio
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lagent.adapters.proxy import SessionClient


@asynccontextmanager
async def proxy_for(handler, **kwargs):
    app = web.Application()
    app.router.add_post('/v1/messages', handler)
    upstream = TestServer(app)
    await upstream.start_server()
    proxy = SessionClient('unused', str(upstream.make_url('')), **kwargs)
    await proxy.start()
    try:
        yield proxy
    finally:
        await proxy.stop()
        await upstream.close()


@pytest.mark.asyncio
async def test_active_stream_outlives_nonstream_total_budget():
    async def handler(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        for _ in range(12):
            await response.write(b': keepalive\n\n')
            await asyncio.sleep(0.03)
        await response.write_eof()
        return response

    async with proxy_for(
        handler,
        forward_timeout=aiohttp.ClientTimeout(total=0.1),
        stream_timeout=aiohttp.ClientTimeout(total=None, connect=1, sock_read=0.2),
    ) as proxy:
        async with aiohttp.ClientSession() as client:
            async with client.post(proxy.url + '/v1/messages', json={'stream': True}) as response:
                assert response.status == 200
                assert (await response.read()).count(b'keepalive') == 12


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_read_stall_is_bounded_even_before_first_byte(stream):
    async def handler(request):
        await asyncio.sleep(0.3)
        return web.json_response({'content': []})

    async with proxy_for(handler, forward_timeout=aiohttp.ClientTimeout(total=None, sock_read=0.05)) as proxy:
        async with aiohttp.ClientSession() as client:
            async with client.post(proxy.url + '/v1/messages', json={'stream': stream}) as response:
                assert response.status == 504
        assert proxy.get_messages() == []


@pytest.mark.asyncio
async def test_nonstream_request_keeps_total_deadline():
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            for _ in range(12):
                await response.write(b' ')
                await asyncio.sleep(0.03)
        except ConnectionError:
            pass
        return response

    async with proxy_for(
        handler,
        forward_timeout=aiohttp.ClientTimeout(total=0.1, sock_read=0.2),
        stream_timeout=aiohttp.ClientTimeout(total=None, sock_read=0.2),
    ) as proxy:
        async with aiohttp.ClientSession() as client:
            async with client.post(proxy.url + '/v1/messages', json={'stream': False}) as response:
                assert response.status == 504


def test_omitted_timeouts_do_not_override_aiohttp_defaults():
    proxy = SessionClient('unused', 'http://localhost')
    assert proxy.forward_timeout is None
    assert proxy.stream_timeout is None


@pytest.mark.asyncio
async def test_connection_budget_applies_before_response(monkeypatch):
    original = aiohttp.TCPConnector._create_connection

    async def delayed_connect(self, request, traces, timeout):
        if request.headers.get('x-api-key') == 'unused':
            await asyncio.sleep(1)
        return await original(self, request, traces, timeout)

    monkeypatch.setattr(aiohttp.TCPConnector, '_create_connection', delayed_connect)

    async def handler(request):
        pytest.fail('connection timeout must happen before a request is sent')

    async with proxy_for(handler, forward_timeout=aiohttp.ClientTimeout(total=2, connect=0.05)) as proxy:
        async with aiohttp.ClientSession() as client:
            async with client.post(proxy.url + '/v1/messages', json={}, headers={'x-api-key': 'client'}) as response:
                assert response.status == 504


@pytest.mark.asyncio
async def test_cancelling_owned_handler_closes_forward_connection(monkeypatch):
    original = SessionClient._handle_request
    handlers = []
    received = asyncio.Event()
    release = asyncio.Event()
    transports = []

    async def tracked(self, request):
        handlers.append(asyncio.current_task())
        return await original(self, request)

    monkeypatch.setattr(SessionClient, '_handle_request', tracked)

    async def handler(request):
        transports.append(request.transport)
        received.set()
        await release.wait()
        return web.json_response({'content': []})

    async with proxy_for(handler, forward_timeout=aiohttp.ClientTimeout(total=5)) as proxy:
        async with aiohttp.ClientSession() as client:
            request = asyncio.create_task(client.post(proxy.url + '/v1/messages', json={}))
            try:
                await asyncio.wait_for(received.wait(), 2)
                handlers[0].cancel()
                await asyncio.gather(handlers[0], return_exceptions=True)
                for _ in range(100):
                    if transports[0].is_closing():
                        break
                    await asyncio.sleep(0.01)
                assert transports[0].is_closing()
                assert handlers[0].cancelled()
                assert proxy.get_messages() == []
            finally:
                release.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
