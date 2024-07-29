import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Type, Union

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached

from lagent.actions import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser


class ContentFetcher:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def fetch(self, url: str) -> Tuple[bool, str]:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.content
        except requests.RequestException as e:
            return False, str(e)

        text = BeautifulSoup(html, 'html.parser').get_text()
        cleaned_text = re.sub(r'\n+', '\n', text)
        return True, cleaned_text


class BaseBrowser(BaseAction):
    """Wrapper around the Web Browser Tool.
    """

    def __init__(self,
                 timeout: int = 5,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        self.searcher = None
        self.fetcher = ContentFetcher(timeout=timeout)
        self.search_results = None
        super().__init__(description, parser, enable)

    @tool_api
    def search(self, query: Union[str, List[str]]) -> dict:
        """Search API
        Args:
            query (List[str]): list of search query strings
        """

        if self.searcher is None:
            raise NotImplementedError

        queries = query if isinstance(query, list) else [query]
        search_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_query = {
                executor.submit(self.searcher.search, q): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                except Exception as exc:
                    warnings.warn(f'{query} generated an exception: {exc}')
                else:
                    for result in results.values():
                        if result['url'] not in search_results:
                            search_results[result['url']] = result
                        else:
                            search_results[
                                result['url']]['summ'] += f"\n{result['summ']}"

        self.search_results = {
            idx: result
            for idx, result in enumerate(search_results.values())
        }
        return self.search_results

    @tool_api
    def select(self, select_ids: List[int]) -> dict:
        """get the detailed content on the selected pages.

        Args:
            select_ids (List[int]): list of index to select. Max number of index to be selected is no more than 4.
        """
        if not self.search_results:
            raise ValueError('No search results to select from.')

        new_search_results = {}
        with ThreadPoolExecutor() as executor:
            future_to_id = {
                executor.submit(self.fetcher.fetch,
                                self.search_results[select_id]['url']):
                select_id
                for select_id in select_ids if select_id in self.search_results
            }

            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    web_success, web_content = future.result()
                except Exception as exc:
                    warnings.warn(f'{select_id} generated an exception: {exc}')
                else:
                    if web_success:
                        self.search_results[select_id][
                            'content'] = web_content[:8192]
                        new_search_results[select_id] = self.search_results[
                            select_id].copy()
                        new_search_results[select_id].pop('summ')

        return new_search_results

    @tool_api
    def open_url(self, url: str) -> dict:
        print(f'Start Browsing: {url}')
        web_success, web_content = self.fetcher.fetch(url)
        if web_success:
            return {'type': 'text', 'content': web_content}
        else:
            return {'error': web_content}
