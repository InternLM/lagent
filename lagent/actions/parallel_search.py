import asyncio
import json
from typing import List, Union
from uuid import uuid4

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.version import __version__


class AsyncParallelSearch(AsyncActionMixin, BaseAction):
    """Search the web and fetch pages with anonymous Parallel Search MCP.

    Register both APIs in ``EnvAgent.stateful_tools`` so the host supplies its
    conversation ID. Call ``reset(session_id)`` when reusing a host ID for a new
    conversation. Host IDs are mapped to opaque UUIDs and never sent to Parallel.
    Requires the ``mcp`` installation extra.
    """

    def __init__(self, timeout: float = 30, headers: dict = None):
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        self.timeout = timeout
        self.headers = {'User-Agent': f'lagent/{__version__}', **(headers or {})}
        self._sessions = {}
        super().__init__()

    def reset(self, session_id: Union[str, int] = 0):
        """Forget the opaque ID when a host conversation is reset or discarded."""
        self._sessions.pop(session_id, None)

    async def _call(self, tool: str, arguments: dict, session_id: Union[str, int]):
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        if session_id not in self._sessions:
            self._sessions[session_id] = str(uuid4())
        arguments = {**arguments, 'session_id': self._sessions[session_id]}

        # The overall deadline includes connecting, initialization and reading.
        async def request():
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as http_client:
                async with streamable_http_client('https://search.parallel.ai/mcp', http_client=http_client) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        result = await client.call_tool(tool, arguments)
            return result

        try:
            result = await asyncio.wait_for(request(), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f'Parallel MCP request timed out after {self.timeout} seconds') from exc
        text = '\n'.join(block.text for block in result.content if block.type == 'text')
        if result.isError:
            raise RuntimeError(text or 'Parallel MCP tool failed')
        if result.structuredContent is not None:
            return result.structuredContent
        if not text:
            raise ValueError('Parallel MCP returned no text or structured content')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @tool_api
    async def search(self, objective: str, search_queries: List[str], session_id: Union[str, int] = 0):
        """Search for current information and return excerpts with source URLs.

        Args:
            objective (str): What the search should find.
            search_queries (List[str]): Related keyword queries for this objective.
            session_id (Union[str, int]): Host conversation ID, supplied by EnvAgent.
        """
        if not objective.strip() or not search_queries or any(not query.strip() for query in search_queries):
            raise ValueError('Provide an objective and at least one nonempty search query')
        return await self._call('web_search', dict(objective=objective, search_queries=search_queries), session_id)

    @tool_api
    async def fetch(self, urls: List[str], objective: str, session_id: Union[str, int] = 0):
        """Fetch source pages when search excerpts need more detail.

        Args:
            urls (List[str]): Public HTTP or HTTPS source URLs.
            objective (str): Information to extract from these pages.
            session_id (Union[str, int]): Host conversation ID, supplied by EnvAgent.
        """
        if not urls or any(not url.startswith(('https://', 'http://')) for url in urls):
            raise ValueError('Provide at least one HTTP or HTTPS URL')
        return await self._call('web_fetch', dict(urls=urls, objective=objective), session_id)
