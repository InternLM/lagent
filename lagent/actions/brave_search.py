import os
from typing import List, Optional, Tuple, Type, Union

import requests

from lagent.schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction, tool_api
from .parser import BaseParser, JsonParser


class BraveSearch(BaseAction):
    """Wrapper around the Brave Search API.

    To use, you should pass your Brave API key to the constructor.

    Code is modified from lang-chain BraveSearchWrapper
    (https://github.com/daver987/langchain/blob/c5016e2
    b0b4878b0a920e809a5169d80b409288b/libs/community/
    langchain_community/utilities/brave_search.py)

    Args:
        api_key (str): API KEY to use brave search API,
            You can create a free API key at https://brave.com/search/api/.
        timeout (int): Upper bound of waiting time for a brave request.
        search_type (str): Brave API support ['web', 'images', 'news',
            'videos'] types of search.
        description (dict): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool): Whether the action is enabled. Defaults to ``True``.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 timeout: int = 5,
                 search_type: str = 'web',
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        super().__init__(description, parser, enable)
        api_key = os.environ.get('BRAVE_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please set BRAVE API key either in the environment '
                'as BRAVE_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout
        self.search_type = search_type

    @tool_api
    def run(self, query: str, k: int = 10) -> ActionReturn:
        """一个可以从BRAVE浏览器搜索结果的API。当你需要对于一个特定问题找到简短明了的回答时，可以使用它。输入应该是一个搜索查询。

        Args:
            query (str): the search content
            k (int): select first k results in the search results as response
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query, count=k)
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

    def _parse_results(self, results: dict) -> Union[str, List[str]]:
        """Parse the search results from Brave API.

        Args:
            results (dict): The search content from Brave API
                in json format.

        Returns:
            List[str]: The parsed search results.
        """

        if self.search_type=="web":
            filtered_result=results.get("web", {}).get("results", [])
        else:
            filtered_result=results.get("results", {})

        snippets = [
                {
                    "title": item.get("title"),
                    "snippets": " ".join(
                        filter(
                            None, [item.get("description"), *item.get("extra_snippets", [])]
                        )
                    ),
                }
                for item in filtered_result
            ]

        if len(snippets) == 0:
            return ['No good Brave Search Result was found']
        return snippets

    def _search(self,
                search_term: str,
                **kwargs) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to Brave API.

        Args:
            search_term (str): The search query.

        Returns:
            tuple: the return value is a tuple contains:
                - status_code (int): HTTP status code from Serper API.
                - response (dict): response context with json format.
        """
        headers = {
            "X-Subscription-Token": self.api_key or '',
            "Accept": "application/json",
        }

        params = {
            'q': search_term,
            **{
                key: value
                for key, value in kwargs.items() if value is not None
            },
        }
        try:
            response = requests.get(
                f'https://api.search.brave.com/res/v1/{self.search_type}/search',
                headers=headers,
                params=params,
                timeout=self.timeout)
        except Exception as e:
            return -1, str(e)
        return response.status_code, response.json()
