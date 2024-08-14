import hashlib
import hmac
import json
import logging
import random
import time
import warnings
from datetime import datetime
from http.client import HTTPSConnection

from cachetools import TTLCache, cached

from lagent.actions import BaseSearch


class TencentSearch(BaseSearch):
    """Wrapper around the tencentclound Search API.
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
                 vrid: str = None):
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
        super().__init__(topk, black_list=None)

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

    def _call_tencent_api(self, query: str) -> dict:

        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

        params = dict(Query=query)
        if self.topk:
            params['Cnt'] = self.topk
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
            'X-TC-Timestamp': timestamp,
            'X-TC-Version': self.version
        }
        # if region:
        #     headers["X-TC-Region"] = region
        if self.api_key:
            headers['X-TC-Token'] = self.api_key

        req = HTTPSConnection(self.host)
        req.request('POST', '/', headers=headers, body=payload.encode('utf-8'))
        resp = req.getresponse()
        return resp.read()

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        for item in response.get('Pages', []):
            display = json.loads(item['display'])
            raw_results.append(
                (display['url'], display['content'], display['title']))
        return self._filter_results(raw_results)
