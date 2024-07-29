import json
import logging
import random
import time
import warnings
from typing import List, Optional, Type

from cachetools import TTLCache, cached
from duckduckgo_search import DDGS

from lagent.actions.base_browser import BaseBrowser
from lagent.actions.parser import BaseParser, JsonParser


class DuckDuckGoSearch:

    def __init__(self,
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ]):
        self.topk = topk
        self.black_list = black_list

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_ddgs(query, timeout=20)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from DuckDuckGo after retries.')

    def _call_ddgs(self, query: str, **kwargs) -> dict:
        ddgs = DDGS(**kwargs)
        response = ddgs.text(query.strip("'"), max_results=10)
        return response

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        for item in response:
            raw_results.append(
                (item['href'], item['description']
                 if 'description' in item else item['body'], item['title']))
        return self._filter_results(raw_results)

    def _filter_results(self, results: List[tuple]) -> dict:
        filtered_results = {}
        count = 0
        for url, snippet, title in results:
            if all(domain not in url
                   for domain in self.black_list) and not url.endswith('.pdf'):
                filtered_results[count] = {
                    'url': url,
                    'summ': json.dumps(snippet, ensure_ascii=False)[1:-1],
                    'title': title
                }
                count += 1
                if count >= self.topk:
                    break
        return filtered_results


class DuckDuckGoBrowser(BaseBrowser):
    """Wrapper around the Web Browser Tool.
    """

    def __init__(self,
                 timeout: int = 5,
                 black_list: Optional[List[str]] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 topk: int = 20,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        self.searcher = DuckDuckGoSearch(black_list=black_list, topk=topk)
        super().__init__(timeout, description, parser, enable)
