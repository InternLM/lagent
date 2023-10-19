from typing import Dict, List, Optional

import torch

from .base_llm import BaseModel


class HFTransformer(BaseModel):
    """Model wrapper around HuggingFace general models.

    Adapted from OpenCompass (https://github.com/InternLM/opencompass
    /blob/main/opencompass/models/huggingface.py)

    Args:
        path (str): The name or path to HuggingFace's model.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(
            self,
            path: str,
            max_seq_len: int = 2048,
            tokenizer_path: Optional[str] = None,
            tokenizer_kwargs: dict = dict(),
            tokenizer_only: bool = False,
            model_kwargs: dict = dict(device_map='auto'),
            meta_template: Optional[Dict] = [
                dict(role='system', begin='<|System|>:', end='\n'),
                dict(role='user', begin='<|User|>:', end='\n'),
                dict(
                    role='assistant',
                    begin='<|Bot|>:',
                    end='<eoa>\n',
                    generate=True)
            ],  # default meta template for InternLM-7b
            extract_pred_after_decode: bool = False,
            batch_padding: bool = False):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template)
        self._load_tokenizer(
            path=path,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path, model_kwargs=model_kwargs)

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path,
            trust_remote_code=True,
            **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, path: str, model_kwargs: dict):
        from transformers import AutoModel
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, **model_kwargs)
        self.model.eval()

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        input_ids = self.tokenizer(
            inputs, truncation=True,
            max_length=self.max_seq_len - max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_out_len, **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds[0]

    def generate_from_template(self, templates, max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates)
        response = self.generate(inputs, max_out_len=max_out_len, **kwargs)
        return response.replace(
            self.template_parser.roles['assistant']['end'].strip(),
            '').strip()


class HFTransformerCasualLM(HFTransformer):

    def _load_model(self, path: str, model_kwargs: dict):
        from transformers import AutoModelForCausalLM
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, **model_kwargs)
        self.model.eval()
