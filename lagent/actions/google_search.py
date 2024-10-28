import os
from typing import List, Optional, Tuple, Type, Union

import aiohttp
import requests

from lagent.schema import ActionReturn, ActionStatusCode
from .base_action import AsyncActionMixin, BaseAction, tool_api
from .parser import BaseParser, JsonParser


class GoogleSearch(BaseAction):
    """Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Code is modified from lang-chain GoogleSerperAPIWrapper
    (https://github.com/langchain-ai/langchain/blob/ba5f
    baba704a2d729a4b8f568ed70d7c53e799bb/libs/langchain/
    langchain/utilities/google_serper.py)

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        timeout (int): Upper bound of waiting time for a serper request.
        search_type (str): Serper API support ['search', 'images', 'news',
            'places'] types of search, currently we only support 'search'.
        description (dict): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
    """
    result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 5,
        search_type: str = 'search',
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        api_key = os.environ.get('SERPER_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please set Serper API key either in the environment '
                'as SERPER_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout
        self.search_type = search_type

    @tool_api
    def run(self, query: str, k: int = 10) -> ActionReturn:
        """一个可以从谷歌搜索结果的API。当你需要对于一个特定问题找到简短明了的回答时，可以使用它。输入应该是一个搜索查询。

        Args:
            query (str): the search content
            k (int): select first k results in the search results as response
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query, k=k)
        # convert search results to ToolReturn format
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response, k)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _parse_results(self, results: dict, k: int) -> Union[str, List[str]]:
        """Parse the search results from Serper API.

        Args:
            results (dict): The search content from Serper API
                in json format.

        Returns:
            List[str]: The parsed search results.
        """

        snippets = []

        if results.get('answerBox'):
            answer_box = results.get('answerBox', {})
            if answer_box.get('answer'):
                return [answer_box.get('answer')]
            elif answer_box.get('snippet'):
                return [answer_box.get('snippet').replace('\n', ' ')]
            elif answer_box.get('snippetHighlighted'):
                return answer_box.get('snippetHighlighted')

        if results.get('knowledgeGraph'):
            kg = results.get('knowledgeGraph', {})
            title = kg.get('title')
            entity_type = kg.get('type')
            if entity_type:
                snippets.append(f'{title}: {entity_type}.')
            description = kg.get('description')
            if description:
                snippets.append(description)
            for attribute, value in kg.get('attributes', {}).items():
                snippets.append(f'{title} {attribute}: {value}.')

        for result in results[self.result_key_for_type[
                self.search_type]][:k]:
            if 'snippet' in result:
                snippets.append(result['snippet'])
            for attribute, value in result.get('attributes', {}).items():
                snippets.append(f'{attribute}: {value}.')

        if len(snippets) == 0:
            return ['No good Google Search Result was found']
        return snippets

    def _search(self,
                search_term: str,
                search_type: Optional[str] = None,
                **kwargs) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to Serper API.

        Args:
            search_term (str): The search query.
            search_type (str): search type supported by Serper API,
                default to 'search'.

        Returns:
            tuple: the return value is a tuple contains:
                - status_code (int): HTTP status code from Serper API.
                - response (dict): response context with json format.
        """
        headers = {
            'X-API-KEY': self.api_key or '',
            'Content-Type': 'application/json',
        }
        params = {
            'q': search_term,
            **{
                key: value
                for key, value in kwargs.items() if value is not None
            },
        }
        try:
            response = requests.post(
                f'https://google.serper.dev/{search_type or self.search_type}',
                headers=headers,
                params=params,
                timeout=self.timeout)
        except Exception as e:
            return -1, str(e)
        return response.status_code, response.json()


class AsyncGoogleSearch(AsyncActionMixin, GoogleSearch):
    """Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Code is modified from lang-chain GoogleSerperAPIWrapper
    (https://github.com/langchain-ai/langchain/blob/ba5f
    baba704a2d729a4b8f568ed70d7c53e799bb/libs/langchain/
    langchain/utilities/google_serper.py)

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        timeout (int): Upper bound of waiting time for a serper request.
        search_type (str): Serper API support ['search', 'images', 'news',
            'places'] types of search, currently we only support 'search'.
        description (dict): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
    """

    @tool_api
    async def run(self, query: str, k: int = 10) -> ActionReturn:
        """一个可以从谷歌搜索结果的API。当你需要对于一个特定问题找到简短明了的回答时，可以使用它。输入应该是一个搜索查询。

        Args:
            query (str): the search content
            k (int): select first k results in the search results as response
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = await self._search(query, k=k)
        # convert search results to ToolReturn format
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    async def _search(self,
                      search_term: str,
                      search_type: Optional[str] = None,
                      **kwargs) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to Serper API.

        Args:
            search_term (str): The search query.
            search_type (str): search type supported by Serper API,
                default to 'search'.

        Returns:
            tuple: the return value is a tuple contains:
                - status_code (int): HTTP status code from Serper API.
                - response (dict): response context with json format.
        """
        headers = {
            'X-API-KEY': self.api_key or '',
            'Content-Type': 'application/json',
        }
        params = {
            'q': search_term,
            **{
                key: value
                for key, value in kwargs.items() if value is not None
            },
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                        f'https://google.serper.dev/{search_type or self.search_type}',
                        headers=headers,
                        params=params) as resp:
                    code, ret = resp.status, await resp.json()
            except aiohttp.ClientError as e:
                code, ret = -1, str(e)
        return code, ret
