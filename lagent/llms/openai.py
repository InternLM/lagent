import json
import os
import time
# from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Dict, List, Optional, Union

import requests

from .base_api import BaseAPIModel

OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


class GPTAPI(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        model_type (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
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
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = [
                     dict(role='system', api_role='system'),
                     dict(role='user', api_role='user'),
                     dict(role='assistant', api_role='assistant')
                 ],
                 openai_api_base: str = OPENAI_API_BASE,
                 temperature: Optional[float] = None):

        super().__init__(
            model_type=model_type,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            retry=retry)
        self.logger = getLogger(__name__)
        self.temperature = temperature

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
        self.url = openai_api_base
        self.model_type = model_type

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.model_type:
            context_window = 32768
        elif '16k' in self.model_type:
            context_window = 16384
        elif 'gpt-4' in self.model_type:
            context_window = 8192
        self.context_window = context_window

    def generate(
        self,
        inputs: Union[List, str],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or List]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature
        return self._generate(inputs, max_out_len, temperature)

    def _generate(self, input: str or List, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or List): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, list, dict))

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        elif isinstance(input, dict):
            messages = [input]
        else:
            messages = input

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        max_out_len = min(
            max_out_len,
            self.context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

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

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            try:
                data = dict(
                    model=self.model_type,
                    messages=messages,
                    max_tokens=max_out_len,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
                raw_response = requests.post(
                    self.url, headers=header, data=json.dumps(data))
            except requests.ConnectionError:
                print('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                print('JsonDecode error, got', str(raw_response.content))
                continue
            try:
                return response['choices'][0]['message']['content'].strip()
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
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        import tiktoken
        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return len(enc.encode(prompt))
