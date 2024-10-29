import asyncio
import copy
import logging
from dataclasses import asdict
from typing import List, Optional, Union

import aiohttp

from lagent.llms.base_llm import AsyncLLMMixin, BaseLLM
from lagent.schema import ModelStatusCode
from lagent.utils.util import filter_suffix


class TritonClient(BaseLLM):
    """TritonClient is a wrapper of TritonClient for LLM.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
            triton inference server
        model_name (str): the name of the model
        session_len (int): the context size
        max_tokens (int): the expected generated token numbers
    """

    def __init__(self,
                 tritonserver_addr: str,
                 model_name: str,
                 session_len: int = 32768,
                 log_level: str = 'WARNING',
                 **kwargs):
        super().__init__(path=None, **kwargs)
        try:
            from lmdeploy.serve.turbomind.chatbot import Chatbot, StatusCode
        except Exception as e:
            logging.error(f'{e}')
            raise RuntimeError('DO NOT use turbomind.chatbot since it has '
                               'been removed by lmdeploy since v0.5.2')
        self.state_map = {
            StatusCode.TRITON_STREAM_END: ModelStatusCode.END,
            StatusCode.TRITON_SERVER_ERR: ModelStatusCode.SERVER_ERR,
            StatusCode.TRITON_SESSION_CLOSED: ModelStatusCode.SESSION_CLOSED,
            StatusCode.TRITON_STREAM_ING: ModelStatusCode.STREAM_ING,
            StatusCode.TRITON_SESSION_OUT_OF_LIMIT:
            ModelStatusCode.SESSION_OUT_OF_LIMIT,
            StatusCode.TRITON_SESSION_INVALID_ARG:
            ModelStatusCode.SESSION_INVALID_ARG,
            StatusCode.TRITON_SESSION_READY: ModelStatusCode.SESSION_READY
        }
        self.chatbot = Chatbot(
            tritonserver_addr=tritonserver_addr,
            model_name=model_name,
            session_len=session_len,
            log_level=log_level,
            **kwargs)

    def generate(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 request_id: str = '',
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 skip_special_tokens: bool = False,
                 **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            inputs (str, List[str]): user's prompt(s) in this round
            session_id (int): the identical id of a session
            request_id (str): the identical id of this round conversation
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.serve.turbomind.chatbot import Session, get_logger
        if isinstance(inputs, str):
            inputs = [inputs]
        prompt = inputs

        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        self.chatbot.cfg = self._update_gen_params(**kwargs)
        max_new_tokens = self.chatbot.cfg.max_new_tokens

        logger = get_logger('service.ft', log_level=self.chatbot.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'max_out_len {max_new_tokens}')

        if self.chatbot._session is None:
            sequence_start = True
            self.chatbot._session = Session(session_id=session_id)
        elif self.chatbot._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            return ''

        self.chatbot._session.status = 1
        self.chatbot._session.request_id = request_id
        self.chatbot._session.response = ''

        status, res, _ = None, '', 0
        for status, res, _ in self.chatbot._stream_infer(
                self.chatbot._session,
                prompt,
                max_new_tokens,
                sequence_start,
                sequence_end,
                skip_special_tokens=skip_special_tokens):
            status = self.state_map.get(status)
            if status < ModelStatusCode.END:
                return ''
            elif status == ModelStatusCode.END:
                self.chatbot._session.histories = (
                    self.chatbot._session.histories +
                    self.chatbot._session.prompt +
                    self.chatbot._session.response)
                # remove stop_words
                res = filter_suffix(res, self.gen_params.get('stop_words'))
                return res

    def stream_chat(self,
                    inputs: List[dict],
                    session_id: int = 2967,
                    request_id: str = '',
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    skip_special_tokens: bool = False,
                    **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in stream mode.

        Args:
            session_id (int): the identical id of a session
            inputs (List[dict]): user's inputs in this round conversation
            request_id (str): the identical id of this round conversation
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        from lmdeploy.serve.turbomind.chatbot import Session, get_logger
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        self.chatbot.cfg = self._update_gen_params(**kwargs)
        max_new_tokens = self.chatbot.cfg.max_new_tokens

        logger = get_logger('service.ft', log_level=self.chatbot.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'max_out_len {max_new_tokens}')

        if self.chatbot._session is None:
            sequence_start = True
            self.chatbot._session = Session(session_id=session_id)
        elif self.chatbot._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            return ModelStatusCode.SESSION_CLOSED, '', 0

        self.chatbot._session.status = 1
        self.chatbot._session.request_id = request_id
        self.chatbot._session.response = ''

        prompt = self.template_parser(inputs)
        status, res, _ = None, '', 0
        for status, res, _ in self.chatbot._stream_infer(
                self.chatbot._session,
                prompt,
                max_new_tokens,
                sequence_start,
                sequence_end,
                skip_special_tokens=skip_special_tokens):
            status = self.state_map.get(status)
            # The stop symbol also appears in the output of the last STREAM_ING state.
            res = filter_suffix(res, self.gen_params.get('stop_words'))
            if status < ModelStatusCode.END:
                return status, res, _
            elif status == ModelStatusCode.END:  # remove stop_words
                self.chatbot._session.histories = (
                    self.chatbot._session.histories +
                    self.chatbot._session.prompt +
                    self.chatbot._session.response)
                yield status, res, _
                break
            else:
                yield status, res, _

    def _update_gen_params(self, **kwargs):
        import mmengine
        new_gen_params = self.update_gen_params(**kwargs)
        self.gen_params['stop_words'] = new_gen_params.pop('stop_words')
        stop_words = self.chatbot._stop_words(
            self.gen_params.get('stop_words'))
        cfg = mmengine.Config(
            dict(
                session_len=self.chatbot.model.session_len,
                stop_words=stop_words,
                bad_words=self.chatbot.cfg.bad_words,
                **new_gen_params))
        return cfg


class LMDeployPipeline(BaseLLM):
    """

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
        tp (int): tensor parallel
        pipeline_cfg (dict): config of pipeline
    """

    def __init__(self,
                 path: str,
                 model_name: Optional[str] = None,
                 tp: int = 1,
                 pipeline_cfg=dict(),
                 **kwargs):
        import lmdeploy
        from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline, version_info

        self.str_version = lmdeploy.__version__
        self.version = version_info
        self.do_sample = kwargs.pop('do_sample', None)
        if self.do_sample is not None and self.version < (0, 6, 0):
            raise RuntimeError(
                '`do_sample` parameter is not supported by lmdeploy until '
                f'v0.6.0, but currently using lmdeloy {self.str_version}')
        super().__init__(path=path, **kwargs)
        backend_config = copy.deepcopy(pipeline_cfg)
        backend_config.update(tp=tp)
        backend_config = {
            k: v
            for k, v in backend_config.items()
            if hasattr(TurbomindEngineConfig, k)
        }
        backend_config = TurbomindEngineConfig(**backend_config)
        chat_template_config = ChatTemplateConfig(
            model_name=model_name) if model_name else None
        self.model = pipeline(
            model_path=self.path,
            backend_config=backend_config,
            chat_template_config=chat_template_config,
            log_level='WARNING')

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
        from lmdeploy.messages import GenerationConfig
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        do_sample = kwargs.pop('do_sample', None)
        gen_params = self.update_gen_params(**kwargs)

        if do_sample is None:
            do_sample = self.do_sample
        if do_sample is not None and self.version < (0, 6, 0):
            raise RuntimeError(
                '`do_sample` parameter is not supported by lmdeploy until '
                f'v0.6.0, but currently using lmdeloy {self.str_version}')
        if self.version >= (0, 6, 0):
            if do_sample is None:
                do_sample = gen_params['top_k'] > 1 or gen_params[
                    'temperature'] > 0
            gen_params.update(do_sample=do_sample)

        gen_config = GenerationConfig(
            skip_special_tokens=skip_special_tokens, **gen_params)
        response = self.model.batch_infer(
            prompt, gen_config=gen_config, do_preprocess=do_preprocess)
        texts = [resp.text for resp in response]
        # remove stop_words
        texts = filter_suffix(texts, self.gen_params.get('stop_words'))
        for resp, text in zip(response, texts):
            resp.text = text
        if batched:
            return [asdict(resp)
                    for resp in response] if return_dict else texts
        return asdict(response[0]) if return_dict else texts[0]


class LMDeployServer(BaseLLM):
    """

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
        server_name (str): host ip for serving
        server_port (int): server port
        tp (int): tensor parallel
        log_level (str): set log level whose value among
            [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    """

    def __init__(self,
                 path: str,
                 model_name: Optional[str] = None,
                 server_name: str = '0.0.0.0',
                 server_port: int = 23333,
                 tp: int = 1,
                 log_level: str = 'WARNING',
                 serve_cfg=dict(),
                 **kwargs):
        super().__init__(path=path, **kwargs)
        self.model_name = model_name
        # TODO get_logger issue in multi processing
        import lmdeploy
        self.client = lmdeploy.serve(
            model_path=self.path,
            model_name=model_name,
            server_name=server_name,
            server_port=server_port,
            tp=tp,
            log_level=log_level,
            **serve_cfg)

    def generate(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 ignore_eos: bool = False,
                 skip_special_tokens: Optional[bool] = False,
                 timeout: int = 30,
                 **kwargs) -> List[str]:
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            inputs (str, List[str]): user's prompt(s) in this round
            session_id (int): the identical id of a session
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
            timeout (int): max time to wait for response
        Returns:
            (a list of/batched) text/chat completion
        """

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False

        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        gen_params.update(max_tokens=max_new_tokens)

        resp = [''] * len(inputs)
        for text in self.client.completions_v1(
                self.model_name,
                inputs,
                session_id=session_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                stream=False,
                ignore_eos=ignore_eos,
                skip_special_tokens=skip_special_tokens,
                timeout=timeout,
                **gen_params):
            resp = [
                resp[i] + item['text']
                for i, item in enumerate(text['choices'])
            ]
        # remove stop_words
        resp = filter_suffix(resp, self.gen_params.get('stop_words'))
        if not batched:
            return resp[0]
        return resp

    def stream_chat(self,
                    inputs: List[dict],
                    session_id=0,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    stream: bool = True,
                    ignore_eos: bool = False,
                    skip_special_tokens: Optional[bool] = False,
                    timeout: int = 30,
                    **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in stream mode.

        Args:
            session_id (int): the identical id of a session
            inputs (List[dict]): user's inputs in this round conversation
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            stream (bool): return in a streaming format if enabled
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
            timeout (int): max time to wait for response
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        gen_params.update(max_tokens=max_new_tokens)
        prompt = self.template_parser(inputs)

        resp = ''
        finished = False
        stop_words = self.gen_params.get('stop_words')
        for text in self.client.completions_v1(
                self.model_name,
                prompt,
                session_id=session_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                stream=stream,
                ignore_eos=ignore_eos,
                skip_special_tokens=skip_special_tokens,
                timeout=timeout,
                **gen_params):
            resp += text['choices'][0]['text']
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


class LMDeployClient(LMDeployServer):
    """

    Args:
        url (str): communicating address 'http://<ip>:<port>' of
            api_server
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
    """

    def __init__(self, url: str, model_name: str, **kwargs):
        BaseLLM.__init__(self, path=url, **kwargs)
        from lmdeploy.serve.openai.api_client import APIClient
        self.client = APIClient(url)
        self.model_name = model_name


class AsyncLMDeployPipeline(AsyncLLMMixin, LMDeployPipeline):
    """

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
        tp (int): tensor parallel
        pipeline_cfg (dict): config of pipeline
    """

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
        from lmdeploy.messages import GenerationConfig, Response

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
        gen_config = GenerationConfig(
            skip_special_tokens=skip_special_tokens, **gen_params)

        async def _inner_generate(uid, text):
            resp = Response('', 0, 0, uid)
            async for out in self.model.generate(
                    text,
                    uid,
                    gen_config,
                    stream_response=True,
                    sequence_start=True,
                    sequence_end=True,
                    do_preprocess=do_preprocess,
                    **kwargs):
                resp.text += out.response
                resp.generate_token_len = out.generate_token_len
                resp.input_token_len = out.input_token_len
                resp.finish_reason = out.finish_reason
                if out.token_ids:
                    resp.token_ids.extend(out.token_ids)
                if out.logprobs:
                    if resp.logprobs is None:
                        resp.logprobs = []
                    resp.logprobs.extend(out.logprobs)
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
            return [asdict(resp)
                    for resp in response] if return_dict else texts
        return asdict(response[0]) if return_dict else texts[0]


class AsyncLMDeployServer(AsyncLLMMixin, LMDeployServer):
    """

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
        server_name (str): host ip for serving
        server_port (int): server port
        tp (int): tensor parallel
        log_level (str): set log level whose value among
            [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    """

    async def generate(
        self,
        inputs: Union[str, List[str]],
        session_ids: Union[int, List[int]] = None,
        sequence_start: bool = True,
        sequence_end: bool = True,
        ignore_eos: bool = False,
        skip_special_tokens: Optional[bool] = False,
        timeout: int = 30,
        **kwargs,
    ):
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            inputs (str, List[str]): user's prompt(s) in this round
            session_ids (int, List[int]): session id(s)
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
            timeout (int): max time to wait for response
        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.serve.openai.api_client import json_loads

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False

        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        gen_params.update(max_tokens=max_new_tokens)

        responses = [''] * len(inputs)
        pload = dict(
            model=self.model_name,
            prompt=inputs,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            stream=False,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
            timeout=timeout,
            **gen_params)
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(3 * 3600)) as session:
            async with session.post(
                    self.client.completions_v1_url,
                    headers=self.client.headers,
                    json=pload) as resp:
                async for chunk in resp.content:
                    if chunk:
                        decoded = chunk.decode('utf-8')
                        output = json_loads(decoded)
                        responses = [
                            response + item['text'] for response, item in zip(
                                responses, output['choices'])
                        ]
        # remove stop_words
        responses = filter_suffix(responses, self.gen_params.get('stop_words'))
        if not batched:
            return responses[0]
        return responses

    async def stream_chat(
        self,
        inputs: List[dict],
        session_id: int = None,
        sequence_start: bool = True,
        sequence_end: bool = True,
        stream: bool = True,
        ignore_eos: bool = False,
        skip_special_tokens: Optional[bool] = False,
        timeout: int = 30,
        **kwargs,
    ):
        """Start a new round conversation of a session. Return the chat
        completions in stream mode.

        Args:
            inputs (List[dict]): user's inputs in this round conversation
            session_id (int): session id
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
            stream (bool): return in a streaming format if enabled
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
            timeout (int): max time to wait for response
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        from lmdeploy.serve.openai.api_client import json_loads

        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        gen_params.update(max_tokens=max_new_tokens)
        prompt = self.template_parser(inputs)

        response = ''
        finished = False
        stop_words = self.gen_params.get('stop_words')

        pload = dict(
            model=self.model_name,
            prompt=prompt,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            stream=stream,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
            timeout=timeout,
            **gen_params)
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(3 * 3600)) as session:
            async with session.post(
                    self.client.completions_v1_url,
                    headers=self.client.headers,
                    json=pload) as resp:
                async for chunk in resp.content:
                    if chunk:
                        decoded = chunk.decode('utf-8')
                        if not decoded.strip() or decoded.rstrip(
                        ) == 'data: [DONE]':
                            continue
                        if decoded[:6] == 'data: ':
                            decoded = decoded[6:]
                        output = json_loads(decoded)
                        response += output['choices'][0]['text']
                        if not response:
                            continue
                        # remove stop_words
                        for sw in stop_words:
                            if sw in response:
                                response = filter_suffix(response, stop_words)
                                finished = True
                                break
                        yield ModelStatusCode.STREAM_ING, response, None
                        if finished:
                            break
                yield ModelStatusCode.END, response, None


class AsyncLMDeployClient(AsyncLMDeployServer):
    """

    Args:
        url (str): communicating address 'http://<ip>:<port>' of
            api_server
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
    """

    def __init__(self, url: str, model_name: str, **kwargs):
        BaseLLM.__init__(self, path=url, **kwargs)
        from lmdeploy.serve.openai.api_client import APIClient
        self.client = APIClient(url)
        self.model_name = model_name
