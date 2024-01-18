from abc import abstractclassmethod
from typing import Dict, List, Optional, Tuple, Union


class LMTemplateParser:
    """Intermidate prompt template parser, specifically for language models.

    Args:
        meta_template (list of dict, optional): The meta template for the
            model.
    """

    def __init__(self, meta_template: Optional[List[Dict]] = None):
        self.meta_template = meta_template
        if meta_template:
            assert isinstance(meta_template, list)
            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for item in meta_template:
                assert isinstance(item, dict)
                assert item['role'] not in self.roles, \
                    'role in meta prompt must be unique!'
                self.roles[item['role']] = item.copy()

    def parse_template(self, dialog) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.

        Args:
            dialog (List[str or PromptList]): A prompt
                template (potentially before being wrapped by meta template).

        Returns:
            str: The final string.
        """
        assert isinstance(dialog, (str, list))
        if isinstance(dialog, str):
            return dialog
        if self.meta_template:

            prompt = ''
            for index, item in enumerate(dialog):
                if isinstance(item, str):
                    prompt += item
                else:
                    new_str = self._prompt2str(item, index == len(dialog) - 1)
                    prompt += new_str
        else:
            # in case the model does not have any meta template
            prompt = ''
            last_sep = ''
            for item in dialog:
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get('content', ''):
                    prompt += last_sep + item.get('prompt', '')
                last_sep = '\n'
        return prompt

    def _format_begin(self, role_cfg, message):
        name = message.get('name', None)
        if name is not None:
            begin = role_cfg['begin'].get('with_name', '')
            if name in role_cfg['begin'].get('name', {}):
                begin = begin.format(name=role_cfg['begin']['name'][name])
            else:
                begin = begin.format(name=name)
        else:
            if isinstance(role_cfg.get('begin', ''), str):
                begin = role_cfg.get('begin', '')
            elif isinstance(role_cfg['begin'], dict):
                begin = role_cfg['begin'].get('without_name', '')
        return begin

    def _prompt2str(self,
                    prompt: Union[str, Dict],
                    last: bool = False) -> Tuple[str, bool]:
        if isinstance(prompt, str):
            return prompt
        merged_prompt = self.roles.get(prompt['role'])

        if merged_prompt.get('fallback_role'):
            merged_prompt = self.roles.get(merged_prompt['fallback_role'])
        begin = self._format_begin(merged_prompt, prompt)
        res = begin
        if last and merged_prompt.get('generate', False):
            res += prompt.get('content', '')
            return res
        res += prompt.get('content', '') + merged_prompt.get('end', '')
        if last and merged_prompt['role'] != 'assistant':
            res += self._format_begin(self.roles['assistant'], {})
            return res
        return res


class BaseModel:
    """Base class for model wrapper.

    Args:
        path (str): The path to the model.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        meta_template (list of dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = False

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 template_parser: 'LMTemplateParser' = LMTemplateParser,
                 meta_template: Optional[List[Dict]] = None):
        self.path = path
        self.max_seq_len = max_seq_len
        self.tokenizer_only = tokenizer_only
        # meta template
        self.template_parser = template_parser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    @abstractclassmethod
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

    def parse_template(self, dialog) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.

        Args:
            dialog (List[str or PromptList]): A prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            str: The final string.
        """
        return self.template_parser.parse_template(dialog)

    def generate_from_template(self, templates, max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates)
        return self.generate(inputs, max_out_len=max_out_len, **kwargs)

    def to(self, device):
        self.model.to(device)
