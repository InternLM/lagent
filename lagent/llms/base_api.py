import warnings
from typing import Dict, List, Optional, Tuple, Union

from lagent.llms.base_llm import AsyncLLMMixin, BaseLLM


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

    def __call__(self, dialog: List[Union[str, List]]):
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
        merged_prompt = self.roles[role_prompt['role']]
        if merged_prompt.get('fallback_role'):
            merged_prompt = self.roles[self.roles[
                merged_prompt['fallback_role']]]
        res = role_prompt.copy()
        res['role'] = merged_prompt['api_role']
        res['content'] = merged_prompt.get('begin', '')
        res['content'] += role_prompt.get('content', '')
        res['content'] += merged_prompt.get('end', '')
        return res


class BaseAPILLM(BaseLLM):
    """Base class for API model wrapper.

    Args:
        model_type (str): The type of model.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(self,
                 model_type: str,
                 retry: int = 2,
                 template_parser: 'APITemplateParser' = APITemplateParser,
                 meta_template: Optional[Dict] = None,
                 *,
                 max_new_tokens: int = 512,
                 top_p: float = 0.8,
                 top_k: int = 40,
                 temperature: float = 0.8,
                 repetition_penalty: float = 0.0,
                 stop_words: Union[List[str], str] = None):
        self.model_type = model_type
        self.meta_template = meta_template
        self.retry = retry
        if template_parser:
            self.template_parser = template_parser(meta_template)

        if isinstance(stop_words, str):
            stop_words = [stop_words]
        self.gen_params = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words,
            skip_special_tokens=False)


class AsyncBaseAPILLM(AsyncLLMMixin, BaseAPILLM):
    pass
