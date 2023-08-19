import os

import requests

from etangent.schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction

DEFAULT_DESCRIPTION = """一个可以从谷歌搜索结果的API。
当你需要对于一个特定问题找到简短明了的回答时，可以使用它。
输入应该是一个搜索查询。
"""


class SerperSearch(BaseAction):
    """Wrapper around the Serper.dev Google Search API.

    You can create a free API key at https://serper.dev.

    To use, you should pass your serper API key to the constructor.

    Code is modified from lang-chain GoogleSerperAPIWrapper
    (https://github.com/langchain-ai/langchain/blob/ba5f
    baba704a2d729a4b8f568ed70d7c53e799bb/libs/langchain/
    langchain/utilities/google_serper.py)
    """
    result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

    def __init__(self,
                 api_key=None,
                 timeout=5,
                 search_type='search',
                 k=10,
                 description=DEFAULT_DESCRIPTION,
                 name=None,
                 enable=True,
                 disable_description=None):
        super().__init__(description, name, enable, disable_description)

        api_key = os.environ.get('SERPER_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please Set Serper API key either in the environment '
                ' as SERPER_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout
        self.search_type = search_type
        self.k = k

    def __call__(self, query):
        tool_return = ActionReturn(url=None, args=None)
        status_code, response = self._search(
            query, search_type=self.search_type, k=self.k)
        # convert search results to ToolReturn format
        if status_code == -1:
            tool_return.errmsg = str(response)
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = dict(text=str(parsed_res))
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _parse_results(self, results):
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
                self.search_type]][:self.k]:
            if 'snippet' in result:
                snippets.append(result['snippet'])
            for attribute, value in result.get('attributes', {}).items():
                snippets.append(f'{attribute}: {value}.')

        if len(snippets) == 0:
            return ['No good Google Search Result was found']
        return snippets

    def _search(self, search_term, search_type='search', **kwargs):
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
                f'https://google.serper.dev/{search_type}',
                headers=headers,
                params=params,
                timeout=self.timeout)
        except Exception as e:
            return -1, e
        return response.status_code, response.json()
