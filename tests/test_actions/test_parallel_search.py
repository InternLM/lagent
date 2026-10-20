import asyncio
import json
from uuid import UUID

import httpx
import pytest
from tenacity import wait_none

from lagent.actions import AsyncParallelSearch, AsyncWebBrowser
from lagent.actions.web_browser import DuckDuckGoSearch
from lagent.agents.agent import AsyncAgent
from lagent.agents.fc_agent import EnvAgent, FunctionCallAgent, get_tool_prompt
from lagent.schema import ActionStatusCode, AgentMessage
from lagent.version import __version__

SOURCE_URL = 'https://docs.python.org/3/library/asyncio.html'


@pytest.fixture
def service(monkeypatch):
    """Exercise the real MCP SDK with a deterministic HTTP boundary."""
    requests = []
    state = dict(error=False, text_only=False, delay=0)
    original = httpx.AsyncClient

    async def handle(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if body['method'] == 'notifications/initialized':
            return httpx.Response(202)
        if body['method'] == 'initialize':
            result = dict(
                protocolVersion='2025-03-26',
                capabilities={},
                serverInfo=dict(name='parallel-test', version='1'),
            )
        elif body['method'] == 'tools/list':
            result = dict(
                tools=[dict(name=name, inputSchema=dict(type='object')) for name in ['web_search', 'web_fetch']]
            )
        else:
            assert body['method'] == 'tools/call'
            await asyncio.sleep(state['delay'])
            tool = body['params']['name']
            args = body['params']['arguments']
            payload = dict(
                results=[dict(url=SOURCE_URL, title='asyncio', excerpts=['Official Python asyncio reference'])],
                session_id=args['session_id'],
            )
            if tool == 'web_fetch':
                payload['results'][0]['content'] = 'The asyncio library supports concurrent code.'
            result = dict(content=[dict(type='text', text=json.dumps(payload))], isError=state['error'])
            if state['error']:
                result['content'][0]['text'] = 'Free anonymous quota exceeded'
            elif not state['text_only']:
                result['structuredContent'] = payload
        return httpx.Response(200, json=dict(jsonrpc='2.0', id=body['id'], result=result))

    class TestClient(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, 'AsyncClient', TestClient)
    monkeypatch.setenv('PARALLEL_API_KEY', 'unused-ambient-key')
    return requests, state


def environment(action=None):
    env = EnvAgent(
        actions=[action or dict(type='lagent.actions.AsyncParallelSearch')],
        stateful_tools=['AsyncParallelSearch.search', 'AsyncParallelSearch.fetch'],
    )
    env._retry_mechanism.wait = wait_none()
    return env


def search_call(objective='Find Python documentation'):
    return dict(
        name='AsyncParallelSearch.search',
        arguments=dict(objective=objective, search_queries=['Python asyncio documentation']),
    )


def test_agent_loop_consumes_search_and_fetch(service):
    requests, state = service
    env = environment()

    class ScriptedSelector(AsyncAgent):
        """Scripted tool selections, with no model inference."""

        async def forward(self, message, session_id, **kwargs):
            if message.sender == 'user':
                call = search_call()
            else:
                response = message.content[0]
                assert response['state'] == ActionStatusCode.SUCCESS
                result = json.loads(response['result'][0]['content'])
                assert result['results'][0]['url'] == SOURCE_URL
                if 'content' in result['results'][0]:
                    return AgentMessage(sender=self.name, content='Found the asyncio reference')
                call = dict(
                    name='AsyncParallelSearch.fetch',
                    arguments=dict(urls=[result['results'][0]['url']], objective='Read the reference'),
                )
            return AgentMessage(sender=self.name, content='', tool_calls=[call], tool_calls_ids=['call-1'])

    async def run():
        agent = FunctionCallAgent(select_agent=ScriptedSelector(), env_agent=env, max_turn=3)
        result = await agent(AgentMessage(sender='user', content='Find asyncio docs'), session_id='conversation-a')
        assert result.content == 'Finished'
        state['text_only'] = True
        follow_up = await env.execute_tool(search_call(), session_id='conversation-a')
        assert SOURCE_URL in follow_up.format_result()

    asyncio.run(run())
    calls = [body for _, body in requests if body['method'] == 'tools/call']
    assert [body['params']['name'] for body in calls] == ['web_search', 'web_fetch', 'web_search']
    ids = [body['params']['arguments']['session_id'] for body in calls]
    assert len(set(ids)) == 1
    assert str(UUID(ids[0])) == ids[0]
    assert ids[0] != 'conversation-a'
    for request, _ in requests:
        assert str(request.url) == 'https://search.parallel.ai/mcp'
        assert request.headers['user-agent'] == f'lagent/{__version__}'
        assert 'authorization' not in request.headers


def test_concurrent_conversations_and_reset(service):
    requests, state = service
    state['delay'] = 0.01
    action = AsyncParallelSearch()
    env = environment(action)

    async def run():
        async def conversation(host_id):
            for _ in range(2):
                result = await env.execute_tool(search_call(host_id), host_id)
                assert result.state == ActionStatusCode.SUCCESS

        await asyncio.gather(conversation('a'), conversation('b'))
        action.reset('a')
        await env.execute_tool(search_call('a'), 'a')
        await env.execute_tool(search_call('b'), 'b')

    asyncio.run(run())
    by_host = {'a': [], 'b': []}
    for _, body in requests:
        if body['method'] == 'tools/call':
            args = body['params']['arguments']
            by_host[args['objective']].append(args['session_id'])
    assert by_host['a'][0] == by_host['a'][1] != by_host['a'][2]
    assert len(set(by_host['b'])) == 1
    assert set(by_host['a']).isdisjoint(by_host['b'])


def test_error_retry_keeps_conversation_and_does_not_fallback(service):
    requests, state = service
    state['error'] = True
    env = environment()

    async def run():
        result = await env(
            AgentMessage(content='', tool_calls=[search_call()], tool_calls_ids=['error']), session_id='retry'
        )
        assert result.content[0]['state'] == ActionStatusCode.API_ERROR
        assert 'quota' in result.content[0]['errmsg']
        state['error'] = False
        result = await env.execute_tool(search_call(), 'retry')
        assert result.state == ActionStatusCode.SUCCESS

    asyncio.run(run())
    calls = [body for _, body in requests if body['method'] == 'tools/call']
    assert len(calls) == 4  # three EnvAgent attempts, then a follow-up
    assert len({body['params']['arguments']['session_id'] for body in calls}) == 1


def test_timeout_and_cancellation(service):
    requests, state = service
    state['delay'] = 10
    action = AsyncParallelSearch(timeout=0.05)
    env = environment(action)

    async def run():
        result = await env.execute_tool(search_call(), 'timeout')
        assert result.state == ActionStatusCode.API_ERROR
        assert 'timed out' in result.errmsg
        task = asyncio.create_task(env.execute_tool(search_call(), 'cancel'))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_choices_permissions_and_arguments(service):
    requests, _ = service
    assert isinstance(AsyncWebBrowser().searcher, DuckDuckGoSearch)
    assert isinstance(AsyncWebBrowser(searcher_type='DuckDuckGoSearch').searcher, DuckDuckGoSearch)
    action = AsyncParallelSearch(headers={'Authorization': 'Bearer explicit-key'})
    assert action.headers['Authorization'] == 'Bearer explicit-key'
    prompt = get_tool_prompt([action])
    assert 'session_id' not in prompt
    assert 'AsyncParallelSearch.search' in prompt and 'AsyncParallelSearch.fetch' in prompt

    async def run():
        disabled = EnvAgent(actions=[])
        assert (await disabled.execute_tool(search_call(), 'disabled')).errmsg.endswith('Not Found')
        env = environment(action)
        invalid = search_call()
        invalid['arguments']['search_queries'] = []
        assert (await env.execute_tool(invalid, 'bad')).state == ActionStatusCode.API_ERROR
        invalid['arguments'] = {}
        assert (await env.execute_tool(invalid, 'bad')).state == ActionStatusCode.ARGS_ERROR
        invalid = dict(name='AsyncParallelSearch.fetch', arguments=dict(urls=['file:///etc/passwd'], objective='Read'))
        assert (await env.execute_tool(invalid, 'bad')).state == ActionStatusCode.API_ERROR
        assert not requests
        result = await env.execute_tool(search_call(), 'explicit')
        assert result.state == ActionStatusCode.SUCCESS

    asyncio.run(run())
    assert all(request.headers['authorization'] == 'Bearer explicit-key' for request, _ in requests)
