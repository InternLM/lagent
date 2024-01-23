from typing import Dict, List, Optional, Callable, Union
import copy
import warnings

import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

from .base_llm import BaseModel

logger = logging.get_logger(__name__)


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
            batch_padding: bool = False,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            additional_eos_token_id: Optional[Union[int, List[int]]] = None,
            **kwargs):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
            **kwargs)
        self._load_tokenizer(
            path=path,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path, model_kwargs=model_kwargs)
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.additional_eos_token_id = additional_eos_token_id
        if not self.additional_eos_token_id:
            stop_words_id = []
            for sw in self.gen_params.get('stop_words', []):
                stop_words_id.append(self.tokenizer([sw])['input_ids'][0])
            self.additional_eos_token_id = stop_words_id

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

    @torch.inference_mode()
    def generate(
        self,
        inputs: List[str],
        **kwargs,
    ):
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        input_length = len(inputs["input_ids"][0])
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        input_ids = inputs["input_ids"]
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]  # noqa: F841  # pylint: disable=W0612
        generation_config = self.model.generation_config
        generation_config = copy.deepcopy(generation_config)

        new_gen_params = self.update_gen_params(**kwargs)
        model_kwargs = generation_config.update(**new_gen_params)
        bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
            generation_config.bos_token_id,
            generation_config.eos_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if self.additional_eos_token_id is not None:
            eos_token_id.extend(self.additional_eos_token_id)
        has_default_max_length = (kwargs.get("max_length") is None
                                  and generation_config.max_length is not None)
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (generation_config.max_new_tokens +
                                            input_ids_seq_length)
            if not has_default_max_length:
                logger.warn(  # pylint: disable=W4902
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (self.logits_processor if self.logits_processor is
                            not None else LogitsProcessorList())
        stopping_criteria = (self.stopping_criteria if self.stopping_criteria
                             is not None else StoppingCriteriaList())

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=self.stopping_criteria
        )
        logits_warper = self.model._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids,
                **model_kwargs
            )
            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False
            )
            unfinished_sequences = unfinished_sequences.mul(
                (min(next_tokens != i for i in eos_token_id)).long()
            )

            output_token_ids = input_ids[0].cpu().tolist()
            output_token_ids = output_token_ids[input_length:]
            for each_eos_token_id in eos_token_id:
                if output_token_ids[-1] == each_eos_token_id:
                    output_token_ids = output_token_ids[:-1]
            response = self.tokenizer.decode(output_token_ids)

            if not batched:
                yield response[0]
            else:
                yield response
            # stop when each sentence is finished, or if we exceed the maximum length
            if (unfinished_sequences.max() == 0 or
                stopping_criteria(input_ids, scores)
            ):
                break


class HFTransformerCasualLM(HFTransformer):

    def _load_model(self, path: str, model_kwargs: dict):
        from transformers import AutoModelForCausalLM
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, **model_kwargs)
        self.model.eval()
