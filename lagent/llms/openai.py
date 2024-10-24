import asyncio
import json
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Dict, List, Optional, Union

import aiohttp
import requests

from ..schema import ModelStatusCode
from ..utils import filter_suffix
from .base_api import AsyncBaseAPILLM, BaseAPILLM

warnings.simplefilter('default')

OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


class GPTAPI(BaseAPILLM):
    """Model wrapper around OpenAI's models.

    Args:
        model_type (str): The name of OpenAI's model.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str = 'gpt-3.5-turbo',
                 retry: int = 2,
                 json_mode: bool = False,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = [
                     dict(role='system', api_role='system'),
                     dict(role='user', api_role='user'),
                     dict(role='assistant', api_role='assistant'),
                     dict(role='environment', api_role='system')
                 ],
                 api_base: str = OPENAI_API_BASE,
                 proxies: Optional[Dict] = None,
                 **gen_params):
        if 'top_k' in gen_params:
            warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.',
                          DeprecationWarning)
            gen_params.pop('top_k')
        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params)
        self.gen_params.pop('top_k')
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = api_base
        self.model_type = model_type
        self.proxies = proxies
        self.json_mode = json_mode

    def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): a list of messages
                or list of lists of messages
            gen_params: additional generation configuration

        Returns:
            Union[str, List[str]]: generated string(s)
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = {**self.gen_params, **gen_params}
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat,
                                self.template_parser._prompt2api(messages),
                                **gen_params)
                for messages in (
                    [inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    def stream_chat(
        self,
        inputs: List[dict],
        **gen_params,
    ):
        """Generate responses given the contexts.

        Args:
            inputs (List[dict]): a list of messages
            gen_params: additional generation configuration

        Returns:
            str: generated string
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = self.update_gen_params(**gen_params)
        gen_params['stream'] = True

        resp = ''
        finished = False
        stop_words = gen_params.get('stop_words')
        if stop_words is None:
            stop_words = []
        # mapping to role that openai supports
        messages = self.template_parser._prompt2api(inputs)
        for text in self._stream_chat(messages, **gen_params):
            if self.model_type.lower().startswith('qwen'):
                resp = text
            else:
                resp += text
            if not resp:
                continue
            # remove stop_words
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

    def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode)

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]
                header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    proxies=self.proxies)
                response = raw_response.json()
                return response['choices'][0]['message']['content'].strip()
            except requests.ConnectionError:
                errmsg = 'Got connection error ' + str(traceback.format_exc())
                self.logger.error(errmsg)
                continue
            except requests.JSONDecodeError:
                errmsg = 'JsonDecode error, got ' + str(raw_response.content)
                self.logger.error(errmsg)
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    errmsg = 'Find error message in response: ' + str(
                        response['error'])
                    self.logger.error(errmsg)
            except Exception as error:
                errmsg = str(error) + '\n' + str(traceback.format_exc())
                self.logger.error(errmsg)
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           f'details. errmsg: {errmsg}')

    def _stream_chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """

        def streaming(raw_response):
            for chunk in raw_response.iter_lines(
                    chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
                if chunk:
                    decoded = chunk.decode('utf-8')
                    if decoded.startswith('data: [DONE]'):
                        return
                    if decoded[:5] == 'data:':
                        decoded = decoded[5:]
                        if decoded[0] == ' ':
                            decoded = decoded[1:]
                    else:
                        print(decoded)
                        continue
                    try:
                        response = json.loads(decoded)
                        if 'code' in response and response['code'] == -20003:
                            # Context exceeds maximum length
                            yield ''
                            return
                        if self.model_type.lower().startswith('qwen'):
                            choice = response['output']['choices'][0]
                            yield choice['message']['content']
                            if choice['finish_reason'] == 'stop':
                                return
                        else:
                            choice = response['choices'][0]
                            if choice['finish_reason'] == 'stop':
                                return
                            yield choice['delta'].get('content', '')
                    except Exception as exc:
                        msg = f'response {decoded} lead to exception of {str(exc)}'
                        self.logger.error(msg)
                        raise Exception(msg) from exc

        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode)

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            if len(self.invalid_keys) == len(self.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0

                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            key = self.keys[self.key_ctr]
            header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                self.org_ctr += 1
                if self.org_ctr == len(self.orgs):
                    self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    proxies=self.proxies)
                return streaming(raw_response)
            except requests.ConnectionError:
                errmsg = 'Got connection error ' + str(traceback.format_exc())
                self.logger.error(errmsg)
                continue
            except requests.JSONDecodeError:
                errmsg = 'JsonDecode error, got ' + str(raw_response.content)
                self.logger.error(errmsg)
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    errmsg = 'Find error message in response: ' + str(
                        response['error'])
                    self.logger.error(errmsg)
            except Exception as error:
                errmsg = str(error) + '\n' + str(traceback.format_exc())
                self.logger.error(errmsg)
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           f'details. errmsg: {errmsg}')

    def generate_request_data(self,
                              model_type,
                              messages,
                              gen_params,
                              json_mode=False):
        """
        Generates the request data for different model types.

        Args:
            model_type (str): The type of the model (e.g., 'gpt', 'internlm', 'qwen').
            messages (list): The list of messages to be sent to the model.
            gen_params (dict): The generation parameters.
            json_mode (bool): Flag to determine if the response format should be JSON.

        Returns:
            tuple: A tuple containing the header and the request data.
        """
        # Copy generation parameters to avoid modifying the original dictionary
        gen_params = gen_params.copy()

        # Hold out 100 tokens due to potential errors in token calculation
        max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
        if max_tokens <= 0:
            return '', ''

        # Initialize the header
        header = {
            'content-type': 'application/json',
        }

        # Common parameters processing
        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        if 'repetition_penalty' in gen_params:
            gen_params['frequency_penalty'] = gen_params.pop(
                'repetition_penalty')

        # Model-specific processing
        data = {}
        if model_type.lower().startswith('gpt'):
            if 'top_k' in gen_params:
                warnings.warn(
                    '`top_k` parameter is deprecated in OpenAI APIs.',
                    DeprecationWarning)
                gen_params.pop('top_k')
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            data = {
                'model': model_type,
                'messages': messages,
                'n': 1,
                **gen_params
            }
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('internlm'):
            data = {
                'model': model_type,
                'messages': messages,
                'n': 1,
                **gen_params
            }
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('qwen'):
            header['X-DashScope-SSE'] = 'enable'
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            if 'frequency_penalty' in gen_params:
                gen_params['repetition_penalty'] = gen_params.pop(
                    'frequency_penalty')
            gen_params['result_format'] = 'message'
            data = {
                'model': model_type,
                'input': {
                    'messages': messages
                },
                'parameters': {
                    **gen_params
                }
            }
        else:
            raise NotImplementedError(
                f'Model type {model_type} is not supported')

        return header, data

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompt.

        Args:
            prompt (str): Input string.

        Returns:
            list: token ids
        """
        import tiktoken
        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return enc.encode(prompt)


class AsyncGPTAPI(AsyncBaseAPILLM):
    """Model wrapper around OpenAI's models.

    Args:
        model_type (str): The name of OpenAI's model.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str = 'gpt-3.5-turbo',
                 retry: int = 2,
                 json_mode: bool = False,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = [
                     dict(role='system', api_role='system'),
                     dict(role='user', api_role='user'),
                     dict(role='assistant', api_role='assistant')
                 ],
                 api_base: str = OPENAI_API_BASE,
                 proxies: Optional[Dict] = None,
                 **gen_params):
        if 'top_k' in gen_params:
            warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.',
                          DeprecationWarning)
            gen_params.pop('top_k')
        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params)
        self.gen_params.pop('top_k')
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = api_base
        self.model_type = model_type
        self.proxies = proxies or {}
        self.json_mode = json_mode

    async def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        session_ids: Union[int, List[int]] = None,
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): a list of messages
                or list of lists of messages
            gen_params: additional generation configuration

        Returns:
            Union[str, List[str]]: generated string(s)
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = {**self.gen_params, **gen_params}
        tasks = [
            self._chat(messages, **gen_params) for messages in (
                [inputs] if isinstance(inputs[0], dict) else inputs)
        ]
        ret = await asyncio.gather(*tasks)
        return ret[0] if isinstance(inputs[0], dict) else ret

    async def stream_chat(
        self,
        inputs: List[dict],
        **gen_params,
    ):
        """Generate responses given the contexts.

        Args:
            inputs (List[dict]): a list of messages
            gen_params: additional generation configuration

        Returns:
            str: generated string
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = self.update_gen_params(**gen_params)
        gen_params['stream'] = True

        resp = ''
        finished = False
        stop_words = gen_params.get('stop_words')
        if stop_words is None:
            stop_words = []
        # mapping to role that openai supports
        messages = self.template_parser._prompt2api(inputs)
        async for text in self._stream_chat(messages, **gen_params):
            if self.model_type.lower().startswith('qwen'):
                resp = text
            else:
                resp += text
            if not resp:
                continue
            # remove stop_words
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode)

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            if len(self.invalid_keys) == len(self.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0

                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            key = self.keys[self.key_ctr]
            header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                self.org_ctr += 1
                if self.org_ctr == len(self.orgs):
                    self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            self.url,
                            headers=header,
                            json=data,
                            proxy=self.proxies.get(
                                'https', self.proxies.get('http'))) as resp:
                        response = await resp.json()
                        return response['choices'][0]['message'][
                            'content'].strip()
            except aiohttp.ClientConnectionError:
                errmsg = 'Got connection error ' + str(traceback.format_exc())
                self.logger.error(errmsg)
                continue
            except aiohttp.ClientResponseError as e:
                errmsg = 'Response error, got ' + str(e)
                self.logger.error(errmsg)
                continue
            except json.JSONDecodeError:
                errmsg = 'JsonDecode error, got ' + (await resp.text(
                    errors='replace'))
                self.logger.error(errmsg)
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    errmsg = 'Find error message in response: ' + str(
                        response['error'])
                    self.logger.error(errmsg)
            except Exception as error:
                errmsg = str(error) + '\n' + str(traceback.format_exc())
                self.logger.error(errmsg)
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           f'details. errmsg: {errmsg}')

    async def _stream_chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """

        async def streaming(raw_response):
            async for chunk in raw_response.content:
                if chunk:
                    decoded = chunk.decode('utf-8')
                    if decoded.startswith('data: [DONE]'):
                        return
                    if decoded[:5] == 'data:':
                        decoded = decoded[5:]
                        if decoded[0] == ' ':
                            decoded = decoded[1:]
                    else:
                        print(decoded)
                        continue
                    try:
                        response = json.loads(decoded)
                        if 'code' in response and response['code'] == -20003:
                            # Context exceeds maximum length
                            yield ''
                            return
                        if self.model_type.lower().startswith('qwen'):
                            choice = response['output']['choices'][0]
                            yield choice['message']['content']
                            if choice['finish_reason'] == 'stop':
                                return
                        else:
                            choice = response['choices'][0]
                            if choice['finish_reason'] == 'stop':
                                return
                            yield choice['delta'].get('content', '')
                    except Exception as exc:
                        msg = f'response {decoded} lead to exception of {str(exc)}'
                        self.logger.error(msg)
                        raise Exception(msg) from exc

        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode)

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            if len(self.invalid_keys) == len(self.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0

                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            key = self.keys[self.key_ctr]
            header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                self.org_ctr += 1
                if self.org_ctr == len(self.orgs):
                    self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            self.url,
                            headers=header,
                            json=data,
                            proxy=self.proxies.get(
                                'https',
                                self.proxies.get('http'))) as raw_response:
                        async for msg in streaming(raw_response):
                            yield msg
                        return
            except aiohttp.ClientConnectionError:
                errmsg = 'Got connection error ' + str(traceback.format_exc())
                self.logger.error(errmsg)
                continue
            except aiohttp.ClientResponseError as e:
                errmsg = 'Response error, got ' + str(e)
                self.logger.error(errmsg)
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    errmsg = 'Find error message in response: ' + str(
                        response['error'])
                    self.logger.error(errmsg)
            except Exception as error:
                errmsg = str(error) + '\n' + str(traceback.format_exc())
                self.logger.error(errmsg)
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           f'details. errmsg: {errmsg}')

    def generate_request_data(self,
                              model_type,
                              messages,
                              gen_params,
                              json_mode=False):
        """
        Generates the request data for different model types.

        Args:
            model_type (str): The type of the model (e.g., 'gpt', 'internlm', 'qwen').
            messages (list): The list of messages to be sent to the model.
            gen_params (dict): The generation parameters.
            json_mode (bool): Flag to determine if the response format should be JSON.

        Returns:
            tuple: A tuple containing the header and the request data.
        """
        # Copy generation parameters to avoid modifying the original dictionary
        gen_params = gen_params.copy()

        # Hold out 100 tokens due to potential errors in token calculation
        max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
        if max_tokens <= 0:
            return '', ''

        # Initialize the header
        header = {
            'content-type': 'application/json',
        }

        # Common parameters processing
        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        if 'repetition_penalty' in gen_params:
            gen_params['frequency_penalty'] = gen_params.pop(
                'repetition_penalty')

        # Model-specific processing
        data = {}
        if model_type.lower().startswith('gpt'):
            if 'top_k' in gen_params:
                warnings.warn(
                    '`top_k` parameter is deprecated in OpenAI APIs.',
                    DeprecationWarning)
                gen_params.pop('top_k')
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            data = {
                'model': model_type,
                'messages': messages,
                'n': 1,
                **gen_params
            }
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('internlm'):
            data = {
                'model': model_type,
                'messages': messages,
                'n': 1,
                **gen_params
            }
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('qwen'):
            header['X-DashScope-SSE'] = 'enable'
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            if 'frequency_penalty' in gen_params:
                gen_params['repetition_penalty'] = gen_params.pop(
                    'frequency_penalty')
            gen_params['result_format'] = 'message'
            data = {
                'model': model_type,
                'input': {
                    'messages': messages
                },
                'parameters': {
                    **gen_params
                }
            }
        else:
            raise NotImplementedError(
                f'Model type {model_type} is not supported')

        return header, data

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompt.

        Args:
            prompt (str): Input string.

        Returns:
            list: token ids
        """
        import tiktoken
        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return enc.encode(prompt)
