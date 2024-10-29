import asyncio
import hashlib
import hmac
import json
import logging
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.client import HTTPSConnection
from typing import List, Optional, Tuple, Type, Union

import aiohttp
import aiohttp.client_exceptions
import requests
from asyncache import cached as acached
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from duckduckgo_search import DDGS, AsyncDDGS

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.utils import async_as_completed


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

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                ddgs = AsyncDDGS(timeout=self.timeout, proxy=self.proxy)
                response = await ddgs.atext(query.strip("'"), max_results=10)
                return self._parse_response(response)
            except Exception as e:
                if isinstance(e, asyncio.TimeoutError):
                    logging.exception('Request to DDGS timed out.')
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
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

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = await self._async_call_bing_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
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

    async def _async_call_bing_api(self, query: str) -> dict:
        endpoint = 'https://api.bing.microsoft.com/v7.0/search'
        params = {'q': query, 'mkt': self.market, 'count': f'{self.topk * 2}'}
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    proxy=self.proxy and
                (self.proxy.get('http') or self.proxy.get('https'))) as resp:
                return await resp.json()

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

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = await self._async_call_brave_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
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

    async def _async_call_brave_api(self, query: str) -> dict:
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
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    proxy=self.proxy and
                (self.proxy.get('http') or self.proxy.get('https'))) as resp:
                return await resp.json()

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

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = await self._async_call_serper_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
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

    async def _async_call_serper_api(self, query: str) -> dict:
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
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    proxy=self.proxy and
                (self.proxy.get('http') or self.proxy.get('https'))) as resp:
                return await resp.json()

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


class TencentSearch(BaseSearch):
    """Wrapper around the tencentclound Search API.

    To use, you should pass your secret_id and secret_key to the constructor.

    Args:
        secret_id (str): Your Tencent Cloud secret ID for accessing the API.
            For more details, refer to the documentation: https://cloud.tencent.com/document/product/598/40488.
        secret_key (str): Your Tencent Cloud secret key for accessing the API.
        api_key (str, optional): Additional API key, if required.
        action (str): The action for this interface, use `SearchCommon`.
        version (str): The API version, use `2020-12-29`.
        service (str): The service name, use `tms`.
        host (str): The API host, use `tms.tencentcloudapi.com`.
        topk (int): The maximum number of search results to return.
        tsn (int): Time filter for search results. Valid values:
            1 (within 1 day), 2 (within 1 week), 3 (within 1 month),
            4 (within 1 year), 5 (within 6 months), 6 (within 3 years).
        insite (str): Specify a site to search within (supports only a single site).
            If not specified, the entire web is searched. Example: `zhihu.com`.
        category (str): Vertical category for filtering results. Optional values include:
            `baike` (encyclopedia), `weather`, `calendar`, `medical`, `news`, `train`, `star` (horoscope).
        vrid (str): Result card type(s). Different `vrid` values represent different types of result cards.
            Supports multiple values separated by commas. Example: `30010255`.
    """

    def __init__(self,
                 secret_id: str = 'Your SecretId',
                 secret_key: str = 'Your SecretKey',
                 api_key: str = '',
                 action: str = 'SearchCommon',
                 version: str = '2020-12-29',
                 service: str = 'tms',
                 host: str = 'tms.tencentcloudapi.com',
                 topk: int = 3,
                 tsn: int = None,
                 insite: str = None,
                 category: str = None,
                 vrid: str = None,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ]):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.api_key = api_key
        self.action = action
        self.version = version
        self.service = service
        self.host = host
        self.tsn = tsn
        self.insite = insite
        self.category = category
        self.vrid = vrid
        super().__init__(topk, black_list=black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_tencent_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Bing Search after retries.')

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = await self._async_call_tencent_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Bing Search after retries.')

    def _get_headers_and_payload(self, query: str) -> tuple:

        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

        params = dict(Query=query)
        # if self.topk:
        #     params['Cnt'] = self.topk
        if self.tsn:
            params['Tsn'] = self.tsn
        if self.insite:
            params['Insite'] = self.insite
        if self.category:
            params['Category'] = self.category
        if self.vrid:
            params['Vrid'] = self.vrid
        payload = json.dumps(params)
        algorithm = 'TC3-HMAC-SHA256'
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

        # ************* 步骤 1：拼接规范请求串 *************
        http_request_method = 'POST'
        canonical_uri = '/'
        canonical_querystring = ''
        ct = 'application/json; charset=utf-8'
        canonical_headers = f'content-type:{ct}\nhost:{self.host}\nx-tc-action:{self.action.lower()}\n'
        signed_headers = 'content-type;host;x-tc-action'
        hashed_request_payload = hashlib.sha256(
            payload.encode('utf-8')).hexdigest()
        canonical_request = (
            http_request_method + '\n' + canonical_uri + '\n' +
            canonical_querystring + '\n' + canonical_headers + '\n' +
            signed_headers + '\n' + hashed_request_payload)

        # ************* 步骤 2：拼接待签名字符串 *************
        credential_scope = date + '/' + self.service + '/' + 'tc3_request'
        hashed_canonical_request = hashlib.sha256(
            canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = (
            algorithm + '\n' + str(timestamp) + '\n' + credential_scope +
            '\n' + hashed_canonical_request)

        # ************* 步骤 3：计算签名 *************
        secret_date = sign(('TC3' + self.secret_key).encode('utf-8'), date)
        secret_service = sign(secret_date, self.service)
        secret_signing = sign(secret_service, 'tc3_request')
        signature = hmac.new(secret_signing, string_to_sign.encode('utf-8'),
                             hashlib.sha256).hexdigest()

        # ************* 步骤 4：拼接 Authorization *************
        authorization = (
            algorithm + ' ' + 'Credential=' + self.secret_id + '/' +
            credential_scope + ', ' + 'SignedHeaders=' + signed_headers +
            ', ' + 'Signature=' + signature)

        # ************* 步骤 5：构造并发起请求 *************
        headers = {
            'Authorization': authorization,
            'Content-Type': 'application/json; charset=utf-8',
            'Host': self.host,
            'X-TC-Action': self.action,
            'X-TC-Timestamp': str(timestamp),
            'X-TC-Version': self.version
        }
        # if self.region:
        #     headers["X-TC-Region"] = self.region
        if self.api_key:
            headers['X-TC-Token'] = self.api_key
        return headers, payload

    def _call_tencent_api(self, query: str) -> dict:
        headers, payload = self._get_headers_and_payload(query)
        req = HTTPSConnection(self.host)
        req.request('POST', '/', headers=headers, body=payload.encode('utf-8'))
        resp = req.getresponse()
        try:
            resp = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            logging.warning(str(e))
            import ast
            resp = ast.literal_eval(resp)
        return resp.get('Response', dict())

    async def _async_call_tencent_api(self, query: str):
        headers, payload = self._get_headers_and_payload(query)
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(
                    'https://' + self.host.lstrip('/'),
                    headers=headers,
                    data=payload) as resp:
                return (await resp.json()).get('Response', {})

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        for item in response.get('Pages', []):
            display = json.loads(item['Display'])
            if not display['url']:
                continue
            raw_results.append((display['url'], display['content']
                                or display['abstract_info'], display['title']))
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

    @acached(cache=TTLCache(maxsize=100, ttl=600))
    async def afetch(self, url: str) -> Tuple[bool, str]:
        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True,
                    timeout=aiohttp.ClientTimeout(self.timeout)) as session:
                async with session.get(url) as resp:
                    html = await resp.text(errors='ignore')
                    text = BeautifulSoup(html, 'html.parser').get_text()
                    cleaned_text = re.sub(r'\n+', '\n', text)
                    return True, cleaned_text
        except Exception as e:
            return False, str(e)


class WebBrowser(BaseAction):
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
                 **kwargs):
        self.searcher = eval(searcher_type)(
            black_list=black_list, topk=topk, **kwargs)
        self.fetcher = ContentFetcher(timeout=timeout)
        self.search_results = None
        super().__init__(description, parser)

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
                executor.submit(self.fetcher.fetch, self.search_results[select_id]['url']): select_id
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


class AsyncWebBrowser(AsyncActionMixin, WebBrowser):
    """Wrapper around the Web Browser Tool.
    """

    @tool_api
    async def search(self, query: Union[str, List[str]]) -> dict:
        """BING search API
        
        Args:
            query (List[str]): list of search query strings
        """
        queries = query if isinstance(query, list) else [query]
        search_results = {}

        tasks = []
        for q in queries:
            task = asyncio.create_task(self.searcher.asearch(q))
            task.query = q
            tasks.append(task)
        async for future in async_as_completed(tasks):
            query = future.query
            try:
                results = await future
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
    async def select(self, select_ids: List[int]) -> dict:
        """get the detailed content on the selected pages.

        Args:
            select_ids (List[int]): list of index to select. Max number of index to be selected is no more than 4.
        """
        if not self.search_results:
            raise ValueError('No search results to select from.')

        new_search_results = {}
        tasks = []
        for select_id in select_ids:
            if select_id in self.search_results:
                task = asyncio.create_task(
                    self.fetcher.afetch(self.search_results[select_id]['url']))
                task.select_id = select_id
                tasks.append(task)
        async for future in async_as_completed(tasks):
            select_id = future.select_id
            try:
                web_success, web_content = await future
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
    async def open_url(self, url: str) -> dict:
        print(f'Start Browsing: {url}')
        web_success, web_content = await self.fetcher.afetch(url)
        if web_success:
            return {'type': 'text', 'content': web_content}
        else:
            return {'error': web_content}
