import asyncio
from typing import List, Union

from lagent.llms.base_llm import AsyncBaseLLM, BaseLLM
from lagent.utils.util import filter_suffix


def asdict_completion(output):
    return {
        key: getattr(output, key)
        for key in [
            'text', 'token_ids', 'cumulative_logprob', 'logprobs',
            'finish_reason', 'stop_reason'
        ]
    }


class VllmModel(BaseLLM):
    """
    A wrapper of vLLM model.

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a huggingface model.
                    - ii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        tp (int): tensor parallel
        vllm_cfg (dict): Other kwargs for vllm model initialization.
    """

    def __init__(self, path: str, tp: int = 1, vllm_cfg=dict(), **kwargs):

        super().__init__(path=path, **kwargs)
        from vllm import LLM
        self.model = LLM(
            model=self.path,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            **vllm_cfg)

    def generate(self,
                 inputs: Union[str, List[str]],
                 do_preprocess: bool = None,
                 skip_special_tokens: bool = False,
                 return_dict: bool = False,
                 **kwargs):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from vllm import SamplingParams

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        stop_words = gen_params.pop('stop_words')

        sampling_config = SamplingParams(
            skip_special_tokens=skip_special_tokens,
            max_tokens=max_new_tokens,
            stop=stop_words,
            **gen_params)
        response = self.model.generate(prompt, sampling_params=sampling_config)
        texts = [resp.outputs[0].text for resp in response]
        # remove stop_words
        texts = filter_suffix(texts, self.gen_params.get('stop_words'))
        for resp, text in zip(response, texts):
            resp.outputs[0].text = text
        if batched:
            return [asdict_completion(resp.outputs[0])
                    for resp in response] if return_dict else texts
        return asdict_completion(
            response[0].outputs[0]) if return_dict else texts[0]


class AsyncVllmModel(AsyncBaseLLM):
    """
    A asynchronous wrapper of vLLM model.

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a huggingface model.
                    - ii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        tp (int): tensor parallel
        vllm_cfg (dict): Other kwargs for vllm model initialization.
    """

    def __init__(self, path: str, tp: int = 1, vllm_cfg=dict(), **kwargs):
        super().__init__(path=path, **kwargs)
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=self.path,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            **vllm_cfg)
        self.model = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self,
                       inputs: Union[str, List[str]],
                       session_ids: Union[int, List[int]] = None,
                       do_preprocess: bool = None,
                       skip_special_tokens: bool = False,
                       return_dict: bool = False,
                       **kwargs):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from vllm import SamplingParams

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        if session_ids is None:
            session_ids = list(range(len(inputs)))
        elif isinstance(session_ids, (int, str)):
            session_ids = [session_ids]
        assert len(inputs) == len(session_ids)

        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        stop_words = gen_params.pop('stop_words')

        sampling_config = SamplingParams(
            skip_special_tokens=skip_special_tokens,
            max_tokens=max_new_tokens,
            stop=stop_words,
            **gen_params)

        async def _inner_generate(uid, text):
            resp, generator = '', self.model.generate(
                text, sampling_params=sampling_config, request_id=uid)
            async for out in generator:
                resp = out.outputs[0]
            return resp

        response = await asyncio.gather(*[
            _inner_generate(sid, inp) for sid, inp in zip(session_ids, prompt)
        ])
        texts = [resp.text for resp in response]
        # remove stop_words
        texts = filter_suffix(texts, self.gen_params.get('stop_words'))
        for resp, text in zip(response, texts):
            resp.text = text
        if batched:
            return [asdict_completion(resp)
                    for resp in response] if return_dict else texts
        return asdict_completion(response[0]) if return_dict else texts[0]
