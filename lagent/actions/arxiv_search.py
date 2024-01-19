import json
from typing import Optional, Type

import arxiv

from lagent.actions.base_action import BaseAction
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

DEFAULT_DESCRIPTION = dict(
    name='ArxivSearch',
    description='Search information from Arxiv.org '
    'Useful for when you need to answer questions about Physics, Mathematics, '
    'Computer Science, Quantitative Biology, Quantitative Finance, Statistics, '
    'Electrical Engineering, and Economics '
    'from scientific articles on arxiv.org',
    api_list=[
        dict(
            name='get_arxiv_article_information',
            description=
            'Run Arxiv search and get the article meta information.',
            parameters=[
                dict(
                    name='query',
                    type='STRING',
                    description='the content of search query')
            ],
            required=['query'],
            return_data=[
                dict(
                    name='content',
                    description='a list of 3 arxiv search papers'),
            ],
        )
    ],
)


class ArxivSearch(BaseAction):
    """ArxivSearch action"""

    def __init__(self,
                 top_k_results: int = 3,
                 max_query_len: int = 300,
                 doc_content_chars_max: int = 1500,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description or DEFAULT_DESCRIPTION, parser, enable)
        self.top_k_results = top_k_results
        self.max_query_len = max_query_len
        self.doc_content_chars_max = doc_content_chars_max

    def get_arxiv_article_information(self, **param) -> ActionReturn:
        query = param['query']
        try:
            results = arxiv.Search(  # type: ignore
                query[:self.max_query_len],
                max_results=self.top_k_results).results()
        except Exception as exc:
            return ActionReturn(
                param,
                errmsg=f'Arxiv exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        docs = [
            f'Published: {result.updated.date()}\nTitle: {result.title}\n'
            f'Authors: {", ".join(a.name for a in result.authors)}\n'
            f'Summary: {result.summary[:self.doc_content_chars_max]}'
            for result in results
        ]
        if docs:
            res = {'content': '\n\n'.join(docs)}
            return ActionReturn(
                param, result={'text': json.dumps(res, ensure_ascii=False)})
        res = {'content': 'No good Arxiv Result was found'}
        return ActionReturn(
            param, result={'text': json.dumps(res, ensure_ascii=False)})
