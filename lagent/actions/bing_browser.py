import asyncio
import json
import logging
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Type, Union

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from duckduckgo_search import DDGS

from lagent.actions import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser


class BaseSearch:

    def __init__(self, topk: int = 3, black_list: List[str] = None):
        self.topk = topk
        self.black_list = black_list

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


class DuckDuckGoSearch(BaseSearch):

    def __init__(self,
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.proxy = kwargs.get('proxy')
        self.timeout = kwargs.get('timeout', 30)
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_ddgs(
                    query, timeout=self.timeout, proxy=self.proxy)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from DuckDuckGo after retries.')

    async def _async_call_ddgs(self, query: str, **kwargs) -> dict:
        ddgs = DDGS(**kwargs)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(ddgs.text, query.strip("'"), max_results=10),
                timeout=self.timeout)
            return response
        except asyncio.TimeoutError:
            logging.exception('Request to DDGS timed out.')
            raise

    def _call_ddgs(self, query: str, **kwargs) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self._async_call_ddgs(query, **kwargs))
            return response
        finally:
            loop.close()

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        for item in response:
            raw_results.append(
                (item['href'], item['description']
                 if 'description' in item else item['body'], item['title']))
        return self._filter_results(raw_results)


class BingSearch(BaseSearch):

    def __init__(self,
                 api_key: str,
                 region: str = 'zh-CN',
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.api_key = api_key
        self.market = region
        self.proxy = kwargs.get('proxy')
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_bing_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Bing Search after retries.')

    def _call_bing_api(self, query: str) -> dict:
        endpoint = 'https://api.bing.microsoft.com/v7.0/search'
        params = {'q': query, 'mkt': self.market, 'count': f'{self.topk * 2}'}
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        response = requests.get(
            endpoint, headers=headers, params=params, proxies=self.proxy)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict) -> dict:
        webpages = {
            w['id']: w
            for w in response.get('webPages', {}).get('value', [])
        }
        raw_results = []

        for item in response.get('rankingResponse',
                                 {}).get('mainline', {}).get('items', []):
            if item['answerType'] == 'WebPages':
                webpage = webpages.get(item['value']['id'])
                if webpage:
                    raw_results.append(
                        (webpage['url'], webpage['snippet'], webpage['name']))
            elif item['answerType'] == 'News' and item['value'][
                    'id'] == response.get('news', {}).get('id'):
                for news in response.get('news', {}).get('value', []):
                    raw_results.append(
                        (news['url'], news['description'], news['name']))

        return self._filter_results(raw_results)


class BraveSearch(BaseSearch):
    """
    Wrapper around the Brave Search API.

    To use, you should pass your Brave Search API key to the constructor.

    Args:
        api_key (str): API KEY to use Brave Search API.
            You can create a free API key at https://api.search.brave.com/app/keys.
        search_type (str): Brave Search API supports ['web', 'news', 'images', 'videos'],
            currently only supports 'news' and 'web'.
        topk (int): The number of search results returned in response from API search results.
        region (str): The country code string. Specifies the country where the search results come from.
        language (str): The language code string. Specifies the preferred language for the search results.
        extra_snippets (bool): Allows retrieving up to 5 additional snippets, which are alternative excerpts from the search results.
        **kwargs: Any other parameters related to the Brave Search API. Find more details at
            https://api.search.brave.com/app/documentation/web-search/get-started.
    """

    def __init__(self,
                 api_key: str,
                 region: str = 'ALL',
                 language: str = 'zh-hans',
                 extra_snippests: bool = True,
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.api_key = api_key
        self.market = region
        self.proxy = kwargs.get('proxy')
        self.language = language
        self.extra_snippests = extra_snippests
        self.search_type = kwargs.get('search_type', 'web')
        self.kwargs = kwargs
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_brave_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Brave Search after retries.')

    def _call_brave_api(self, query: str) -> dict:
        endpoint = f'https://api.search.brave.com/res/v1/{self.search_type}/search'
        params = {
            'q': query,
            'country': self.market,
            'search_lang': self.language,
            'extra_snippets': self.extra_snippests,
            'count': self.topk,
            **{
                key: value
                for key, value in self.kwargs.items() if value is not None
            },
        }
        headers = {
            'X-Subscription-Token': self.api_key or '',
            'Accept': 'application/json'
        }
        response = requests.get(
            endpoint, headers=headers, params=params, proxies=self.proxy)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict) -> dict:
        if self.search_type == 'web':
            filtered_result = response.get('web', {}).get('results', [])
        else:
            filtered_result = response.get('results', {})
        raw_results = []

        for item in filtered_result:
            raw_results.append((
                item.get('url', ''),
                ' '.join(
                    filter(None, [
                        item.get('description'),
                        *item.get('extra_snippets', [])
                    ])),
                item.get('title', ''),
            ))
        return self._filter_results(raw_results)


class GoogleSearch(BaseSearch):
    """
    Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Args:
        api_key (str): API KEY to use serper google search API.
            You can create a free API key at https://serper.dev.
        search_type (str): Serper API supports ['search', 'images', 'news',
            'places'] types of search, currently we only support 'search' and 'news'.
        topk (int): The number of search results returned in response from api search results.
        **kwargs: Any other parameters related to the Serper API. Find more details at
            https://serper.dev/playground
    """

    result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

    def __init__(self,
                 api_key: str,
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.api_key = api_key
        self.proxy = kwargs.get('proxy')
        self.search_type = kwargs.get('search_type', 'search')
        self.kwargs = kwargs
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_serper_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Google Serper Search after retries.'
        )

    def _call_serper_api(self, query: str) -> dict:
        endpoint = f'https://google.serper.dev/{self.search_type}'
        params = {
            'q': query,
            'num': self.topk,
            **{
                key: value
                for key, value in self.kwargs.items() if value is not None
            },
        }
        headers = {
            'X-API-KEY': self.api_key or '',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            endpoint, headers=headers, params=params, proxies=self.proxy)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict) -> dict:
        raw_results = []

        if response.get('answerBox'):
            answer_box = response.get('answerBox', {})
            if answer_box.get('answer'):
                raw_results.append(('', answer_box.get('answer'), ''))
            elif answer_box.get('snippet'):
                raw_results.append(
                    ('', answer_box.get('snippet').replace('\n', ' '), ''))
            elif answer_box.get('snippetHighlighted'):
                raw_results.append(
                    ('', answer_box.get('snippetHighlighted'), ''))

        if response.get('knowledgeGraph'):
            kg = response.get('knowledgeGraph', {})
            description = kg.get('description', '')
            attributes = '. '.join(
                f'{attribute}: {value}'
                for attribute, value in kg.get('attributes', {}).items())
            raw_results.append(
                (kg.get('descriptionLink', ''),
                 f'{description}. {attributes}' if attributes else description,
                 f"{kg.get('title', '')}: {kg.get('type', '')}."))

        for result in response[self.result_key_for_type[
                self.search_type]][:self.topk]:
            description = result.get('snippet', '')
            attributes = '. '.join(
                f'{attribute}: {value}'
                for attribute, value in result.get('attributes', {}).items())
            raw_results.append(
                (result.get('link', ''),
                 f'{description}. {attributes}' if attributes else description,
                 result.get('title', '')))

        return self._filter_results(raw_results)


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


class BingBrowser(BaseAction):
    """Wrapper around the Web Browser Tool.
    """

    def __init__(self,
                 searcher_type: str = 'DuckDuckGoSearch',
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
                 enable: bool = True,
                 **kwargs):
        self.searcher = eval(searcher_type)(
            black_list=black_list, topk=topk, **kwargs)
        self.fetcher = ContentFetcher(timeout=timeout)
        self.search_results = None
        super().__init__(description, parser, enable)

    @tool_api
    def search(self, query: Union[str, List[str]]) -> dict:
        """BING search API
        Args:
            query (List[str]): list of search query strings
        """
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
