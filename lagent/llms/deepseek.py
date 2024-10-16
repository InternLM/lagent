import asyncio
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Dict, List, Optional, Union

import aiohttp
import requests

from lagent.schema import ModelStatusCode
from lagent.utils import filter_suffix
from lagent.llms.base_api import AsyncBaseAPILLM, BaseAPILLM
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN

import importlib.util
import os
import site


def load_official_openai():
    site_packages = site.getsitepackages()

    for sp in site_packages:
        openai_init = os.path.join(sp, 'openai', '__init__.py')
        if os.path.exists(openai_init):
            spec = importlib.util.spec_from_file_location("official_openai", openai_init)
            official_openai = importlib.util.module_from_spec(spec)
            sys.modules["official_openai"] = official_openai
            try:
                spec.loader.exec_module(official_openai)
                return official_openai
            except Exception as e:
                print(f"Error loading official openai: {e}")
                del sys.modules["official_openai"]
                raise
    raise ImportError("Official openai is not found.")


warnings.simplefilter('default')

Deepseek_API_BASE = "https://api.deepseek.com"


class DeepseekAPI(BaseAPILLM):
    """
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
        meta_template (Dict, optional): The model's meta prompts
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str = 'deepseek-chat',
                 retry: int = 2,
                 json_mode: bool = False,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = [
                     dict(role='system', api_role='system'),
                     dict(role='user', api_role='user'),
                     dict(role='assistant', api_role='assistant')
                 ],
                 api_base: str = Deepseek_API_BASE,
                 proxies: Optional[Dict] = None,
                 max_tokens: Optional[int] = None,
                 **gen_params):
        # if 'top_k' in gen_params:
        #     warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.',
        #                   DeprecationWarning)
        #     gen_params.pop('top_k')
        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params)
        # self.gen_params.pop('top_k')
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [os.getenv('DEEPSEEK_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        self.org_ctr = 0
        self.url = api_base
        self.model_type = model_type
        self.proxies = proxies
        self.json_mode = json_mode
        self.max_tokens = max_tokens or DEFAULT_LLM_MAX_TOKEN

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
        gen_params = {**self.gen_params, **gen_params}
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat, messages, **gen_params)
                for messages in (
                    [inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompts dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        max_num_retries = 0
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
                official_openai = load_official_openai()

                key = self.keys[self.key_ctr]
                client = official_openai.OpenAI(api_key=key, base_url="https://api.deepseek.com")
                response = dict()
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=gen_params.get('temperature', 0.7),
                        stream=False
                    )
                    return response.choices[0].message.content.strip()
                except requests.ConnectionError:
                    print('Got connection error, retrying...')
                    continue
                except requests.JSONDecodeError:
                    print('JsonDecode error, got', str(response.content))
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

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompts.

        Args:
            prompt (str): Input string.

        Returns:
            list: token ids
        """
        import tiktoken
        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return enc.encode(prompt)

