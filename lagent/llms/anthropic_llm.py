import asyncio
import json
import os
from typing import Dict, List, Optional, Union

import anthropic
import httpcore
import httpx
from anthropic import NOT_GIVEN
from requests.exceptions import ProxyError

from .base_api import AsyncBaseAPILLM, BaseAPILLM


class ClaudeAPI(BaseAPILLM):

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'claude-3-5-sonnet-20241022',
        retry: int = 5,
        key: Union[str, List[str]] = 'ENV',
        proxies: Optional[Dict] = None,
        meta_template: Optional[Dict] = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='user'),
        ],
        temperature: float = NOT_GIVEN,
        max_new_tokens: int = 512,
        top_p: float = NOT_GIVEN,
        top_k: int = NOT_GIVEN,
        repetition_penalty: float = 0.0,
        stop_words: Union[List[str], str] = None,
    ):

        super().__init__(
            meta_template=meta_template,
            model_type=model_type,
            retry=retry,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words,
        )

        key = os.getenv('Claude_API_KEY') if key == 'ENV' else key

        if isinstance(key, str):
            self.keys = [key]
        else:
            self.keys = list(set(key))
        self.clients = {key: anthropic.AsyncAnthropic(proxies=proxies, api_key=key) for key in self.keys}

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        self.model_type = model_type
        self.proxies = proxies

    def chat(
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
        gen_params = {**self.gen_params, **gen_params}
        import nest_asyncio

        nest_asyncio.apply()

        async def run_async_tasks():
            tasks = [
                self._chat(self.template_parser(messages), **gen_params)
                for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
            ]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_running_loop()
            # If the event loop is already running, schedule the task
            future = asyncio.ensure_future(run_async_tasks())
            ret = loop.run_until_complete(future)
        except RuntimeError:
            # If no running event loop, start a new one
            ret = asyncio.run(run_async_tasks())
        return ret[0] if isinstance(inputs[0], dict) else ret

    def generate_request_data(self, model_type, messages, gen_params):
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
        gen_params.pop('repetition_penalty')
        if 'stop_words' in gen_params:
            gen_params['stop_sequences'] = gen_params.pop('stop_words')
        # Common parameters processing
        gen_params['max_tokens'] = max_tokens
        gen_params.pop('skip_special_tokens', None)
        gen_params.pop('session_id', None)

        system = None
        if messages[0]['role'] == 'system':
            system = messages.pop(0)
            system = system['content']
        for message in messages:
            message.pop('name', None)
        data = {'model': model_type, 'messages': messages, **gen_params}
        if system:
            data['system'] = system
        return data

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        data = self.generate_request_data(model_type=self.model_type, messages=messages, gen_params=gen_params)
        max_num_retries = 0

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
            client = self.clients[key]

            try:
                response = await client.messages.create(**data)
                response = json.loads(response.json())
                return response['content'][0]['text'].strip()
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
                print(f'API请求错误: {e}')
                await asyncio.sleep(5)

            except (httpcore.ProxyError, ProxyError) as e:

                print(f'代理服务器错误: {e}')
                await asyncio.sleep(5)
            except httpx.TimeoutException as e:
                print(f'请求超时: {e}')
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                raise

            except Exception as error:
                if error.body['error']['message'] == 'invalid x-api-key':
                    self.invalid_keys.add(key)
                    self.logger.warn(f'invalid key: {key}')
                elif error.body['error']['type'] == 'overloaded_error':
                    await asyncio.sleep(5)
                elif error.body['error']['message'] == 'Internal server error':
                    await asyncio.sleep(5)
                elif error.body['error']['message'] == (
                    'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to '
                    'upgrade or purchase credits.'
                ):
                    self.invalid_keys.add(key)
                    print(f'API has no quota: {key}, Valid keys: {len(self.keys) - len(self.invalid_keys)}')
            max_num_retries += 1

        raise RuntimeError(
            'Calling Claude failed after retrying for ' f'{max_num_retries} times. Check the logs for ' 'details.'
        )


class AsyncClaudeAPI(AsyncBaseAPILLM):

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'claude-3-5-sonnet-20241022',
        retry: int = 5,
        key: Union[str, List[str]] = 'ENV',
        proxies: Optional[Dict] = None,
        meta_template: Optional[Dict] = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='user'),
        ],
        temperature: float = NOT_GIVEN,
        max_new_tokens: int = 512,
        top_p: float = NOT_GIVEN,
        top_k: int = NOT_GIVEN,
        repetition_penalty: float = 0.0,
        stop_words: Union[List[str], str] = None,
    ):

        super().__init__(
            model_type=model_type,
            retry=retry,
            meta_template=meta_template,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words,
        )

        key = os.getenv('Claude_API_KEY') if key == 'ENV' else key

        if isinstance(key, str):
            self.keys = [key]
        else:
            self.keys = list(set(key))
        self.clients = {key: anthropic.AsyncAnthropic(proxies=proxies, api_key=key) for key in self.keys}

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        self.model_type = model_type
        self.proxies = proxies

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
        gen_params = {**self.gen_params, **gen_params}
        tasks = [
            self._chat(messages, **gen_params) for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
        ]
        ret = await asyncio.gather(*tasks)
        return ret[0] if isinstance(inputs[0], dict) else ret

    def generate_request_data(self, model_type, messages, gen_params):
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
        gen_params.pop('repetition_penalty')
        if 'stop_words' in gen_params:
            gen_params['stop_sequences'] = gen_params.pop('stop_words')
        # Common parameters processing
        gen_params['max_tokens'] = max_tokens
        gen_params.pop('skip_special_tokens', None)
        gen_params.pop('session_id', None)

        system = None
        if messages[0]['role'] == 'system':
            system = messages.pop(0)
            system = system['content']
        for message in messages:
            message.pop('name', None)
        data = {'model': model_type, 'messages': messages, **gen_params}
        if system:
            data['system'] = system
        return data

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)
        messages = self.template_parser(messages)
        data = self.generate_request_data(model_type=self.model_type, messages=messages, gen_params=gen_params)
        max_num_retries = 0

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
            client = self.clients[key]

            try:
                response = await client.messages.create(**data)
                response = json.loads(response.json())
                return response['content'][0]['text'].strip()
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
                print(f'API请求错误: {e}')
                await asyncio.sleep(5)

            except (httpcore.ProxyError, ProxyError) as e:

                print(f'代理服务器错误: {e}')
                await asyncio.sleep(5)
            except httpx.TimeoutException as e:
                print(f'请求超时: {e}')
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                raise

            except Exception as error:
                if error.body['error']['message'] == 'invalid x-api-key':
                    self.invalid_keys.add(key)
                    self.logger.warn(f'invalid key: {key}')
                elif error.body['error']['type'] == 'overloaded_error':
                    await asyncio.sleep(5)
                elif error.body['error']['message'] == 'Internal server error':
                    await asyncio.sleep(5)
                elif error.body['error']['message'] == (
                    'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to'
                    ' upgrade or purchase credits.'
                ):
                    self.invalid_keys.add(key)
                    print(f'API has no quota: {key}, Valid keys: {len(self.keys) - len(self.invalid_keys)}')
                else:
                    raise error
            max_num_retries += 1

        raise RuntimeError(
            'Calling Claude failed after retrying for ' f'{max_num_retries} times. Check the logs for ' 'details.'
        )
