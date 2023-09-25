import re
import threading
import warnings
from abc import abstractclassmethod
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

from .base_llm import BaseModel


class BaseAPIModel(BaseModel):
    """Base class for API model wrapper.

    Args:
        model_type (str): The type of model.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str,
                 query_per_second: int = 1,
                 retry: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None):
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.meta_template = meta_template
        self.retry = retry
        self.query_per_second = query_per_second
        self.token_bucket = TokenBucket(query_per_second)
        self.template_parser = APITemplateParser(meta_template)

    @abstractclassmethod
    def generate(self, inputs, max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or list]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)

        # Count English words
        english_count = sum(len(part.split()) for part in english_parts)

        # Count Chinese words
        chinese_count = sum(len(part) for part in chinese_parts)

        return english_count + chinese_count

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def to(self, device):
        pass


class APITemplateParser:
    """Intermidate prompt template parser, specifically for API models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        # Check meta template
        if meta_template:
            assert isinstance(meta_template, list)
            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for item in meta_template:
                assert isinstance(item, dict)
                assert item['role'] not in self.roles, \
                    'role in meta prompt must be unique!'
                self.roles[item['role']] = item.copy()

    def parse_template(self, dialog: List[Union[str, List]]):
        """Parse the intermidate prompt template, and wrap it with meta
        template if applicable. When the meta template is set and the input is
        a list, the return value will be a list containing the full
        conversation history. Each item looks like:

        .. code-block:: python

            {'role': 'user', 'content': '...'}).

        Args:
            dialog (List[str or list]): An intermidate prompt
                template (potentially before being wrapped by meta template).

        Returns:
            List[str or list]: The finalized prompt or a conversation.
        """
        assert isinstance(dialog, (str, list))
        if isinstance(dialog, str):
            return dialog
        if self.meta_template:

            prompt = list()
            # Whether to keep generating the prompt
            generate = True
            for i, item in enumerate(dialog):
                if not generate:
                    break
                if isinstance(item, str):
                    if item.strip():
                        # TODO: logger
                        warnings.warn('Non-empty string in prompt template '
                                      'will be ignored in API models.')
                else:
                    api_prompts = self._prompt2api(item)
                    prompt.append(api_prompts)

            # merge the consecutive prompts assigned to the same role
            new_prompt = list([prompt[0]])
            last_role = prompt[0]['role']
            for item in prompt[1:]:
                if item['role'] == last_role:
                    new_prompt[-1]['content'] += '\n' + item['content']
                else:
                    last_role = item['role']
                    new_prompt.append(item)
            prompt = new_prompt

        else:
            # in case the model does not have any meta template
            prompt = ''
            last_sep = ''
            for item in dialog:
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get('content', ''):
                    prompt += last_sep + item.get('content', '')
                last_sep = '\n'
        return prompt

    def _prompt2api(self, prompts: Union[List, str]) -> Tuple[str, bool]:
        """Convert the prompts to a API-style prompts, given an updated
        role_dict.

        Args:
            prompts (Union[List, str]): The prompts to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[str, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        if isinstance(prompts, str):
            return prompts
        elif isinstance(prompts, dict):
            api_role = self._role2api_role(prompts)
            return api_role

        res = []
        for prompt in prompts:
            if isinstance(prompt, str):
                raise TypeError('Mixing str without explicit role is not '
                                'allowed in API models!')
            else:
                api_role = self._role2api_role(prompt)
                res.append(api_role)
        return res

    def _role2api_role(self, role_prompt: Dict) -> Tuple[str, bool]:

        merged_prompt = self.roles.get(
            role_prompt['role'],
            self.roles.get(
                self.roles[role_prompt['role']].get('fallback_role')))
        res = {}
        res['role'] = merged_prompt['api_role']
        res['content'] = merged_prompt.get('begin', '')
        res['content'] += role_prompt.get('content', '')
        res['content'] += merged_prompt.get('end', '')
        return res


class TokenBucket:
    """A token bucket for rate limiting.

    Args:
        rate (float): The rate of the token bucket.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = threading.Semaphore(0)
        self.started = False

    def _add_tokens(self):
        """Add tokens to the bucket."""
        while True:
            if self._tokens._value < self._rate:
                self._tokens.release()
            sleep(1 / self._rate)

    def get_token(self):
        """Get a token from the bucket."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._add_tokens, daemon=True).start()
        self._tokens.acquire()
