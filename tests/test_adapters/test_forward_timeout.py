"""Short real HTTP deadlines exercise the same path as long agent generations."""

import asyncio
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lagent.adapters.proxy import SessionClient


@asynccontextmanager
async def proxy_for(handler):
    app = web.Application()
    app.router.add_post('/v1/messages', handler)
    upstream = TestServer(app)
    await upstream.start_server()
    proxy = SessionClient('unused', str(upstream.make_url('')))
    await proxy.start()
    try:
        yield proxy
    finally:
        await proxy.stop()
        await upstream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_forwarding_outlives_aiohttp_default_total_timeout(monkeypatch, stream):
    # Shorten the library default so a regression fails without a five-minute wait.
    monkeypatch.setattr(aiohttp.client, 'DEFAULT_TIMEOUT', aiohttp.ClientTimeout(total=0.05))

    async def handler(request):
        # Include waiting for the first byte as well as reading the response.
        await asyncio.sleep(0.1)
        if not stream:
            return web.json_response({'content': []})
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        for _ in range(4):
            await response.write(b': keepalive\n\n')
            await asyncio.sleep(0.03)
        await response.write_eof()
        return response

    async with proxy_for(handler) as proxy:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.post(proxy.url + '/v1/messages', json={'stream': stream}) as response:
                assert response.status == 200
                if stream:
                    assert (await response.read()).count(b'keepalive') == 4
                else:
                    assert await response.json() == {'content': []}


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

    async with proxy_for(handler) as proxy:
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
