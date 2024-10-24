import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Dict, Generator, List, Optional, Tuple, Union

import requests

from lagent.schema import ModelStatusCode
from lagent.utils.util import filter_suffix
from .base_api import BaseAPILLM

warnings.simplefilter('default')

SENSENOVA_API_BASE = 'https://api.sensenova.cn/v1/llm/chat-completions'

sensechat_models = {'SenseChat-5': 131072, 'SenseChat-5-Cantonese': 32768}


class SensenovaAPI(BaseAPILLM):
    """Model wrapper around SenseTime's models.

    Args:
        model_type (str): The name of SenseTime's model.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): SenseTime key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $SENSENOVA_API_KEY. If it's a list, the keys will be
            used in round-robin manner. Defaults to 'ENV'.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        sensenova_api_base (str): The base url of SenseTime's API. Defaults to
            'https://api.sensenova.cn/v1/llm/chat-completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'SenseChat-5-Cantonese',
        retry: int = 2,
        json_mode: bool = False,
        key: Union[str, List[str]] = 'ENV',
        meta_template: Optional[Dict] = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='system'),
        ],
        sensenova_api_base: str = SENSENOVA_API_BASE,
        proxies: Optional[Dict] = None,
        **gen_params,
    ):

        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params,
        )
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            # First, apply for SenseNova's ak and sk from SenseTime staff
            # Then, generated SENSENOVA_API_KEY using lagent.utils.gen_key.auto_gen_jwt_token(ak, sk)
            self.keys = [
                os.getenv('SENSENOVA_API_KEY') if key == 'ENV' else key
            ]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        self.url = sensenova_api_base
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
    ) -> Generator[Tuple[ModelStatusCode, str, Optional[str]], None, None]:
        """Generate responses given the contexts.

        Args:
            inputs (List[dict]): a list of messages
            gen_params: additional generation configuration

        Yields:
            Tuple[ModelStatusCode, str, Optional[str]]: Status code, generated string, and optional metadata
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = self.update_gen_params(**gen_params)
        gen_params['stream'] = True

        resp = ''
        finished = False
        stop_words = gen_params.get('stop_words') or []
        messages = self.template_parser._prompt2api(inputs)
        for text in self._stream_chat(messages, **gen_params):
            # TODO 测试 resp = text 还是 resp += text
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
            json_mode=self.json_mode,
        )

        max_num_retries = 0
        while max_num_retries < self.retry:
            self._wait()

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

            response = dict()
            try:
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    proxies=self.proxies,
                )
                response = raw_response.json()
                return response['choices'][0]['message']['content'].strip()
            except requests.ConnectionError:
                print('Got connection error, retrying...')
                continue
            except requests.JSONDecodeError:
                print('JsonDecode error, got', str(raw_response.content))
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

                    print('Find error message in response: ',
                          str(response['error']))
            except Exception as error:
                print(str(error))
            max_num_retries += 1

        raise RuntimeError('Calling SenseTime failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _stream_chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """

        def streaming(raw_response):
            for chunk in raw_response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = chunk.decode('utf-8')
                        # print(f"Decoded chunk: {decoded_chunk}")

                        if decoded_chunk == 'data:[DONE]':
                            # print("Stream ended")
                            break

                        if decoded_chunk.startswith('data:'):
                            json_str = decoded_chunk[5:]
                            chunk_data = json.loads(json_str)

                            if 'data' in chunk_data and 'choices' in chunk_data[
                                    'data']:
                                choice = chunk_data['data']['choices'][0]
                                if 'delta' in choice:
                                    content = choice['delta']
                                    yield content
                        else:
                            print(f'Unexpected format: {decoded_chunk}')

                    except json.JSONDecodeError as e:
                        print(f'JSON parsing error: {e}')
                    except Exception as e:
                        print(
                            f'An error occurred while processing the chunk: {e}'
                        )

        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode,
        )

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
            header['Authorization'] = f'Bearer {key}'

            response = dict()
            try:
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    proxies=self.proxies,
                )
                return streaming(raw_response)
            except requests.ConnectionError:
                print('Got connection error, retrying...')
                continue
            except requests.JSONDecodeError:
                print('JsonDecode error, got', str(raw_response.content))
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

                    print('Find error message in response: ',
                          str(response['error']))
            except Exception as error:
                print(str(error))
            max_num_retries += 1

        raise RuntimeError('Calling SenseTime failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def generate_request_data(self,
                              model_type,
                              messages,
                              gen_params,
                              json_mode=False):
        """
        Generates the request data for different model types.

        Args:
            model_type (str): The type of the model (e.g., 'sense').
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
        if model_type.lower().startswith('sense'):
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
        enc = self.tiktoken.encoding_for_model('gpt-4o')
        return enc.encode(prompt)
