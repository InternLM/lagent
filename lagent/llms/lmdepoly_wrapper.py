from typing import List, Optional, Union

from lagent.llms.base_llm import BaseModel
from lagent.schema import ModelStatusCode
from lagent.utils.util import filter_suffix


class TritonClient(BaseModel):
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
        from lmdeploy.serve.turbomind.chatbot import Chatbot, StatusCode
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


class LMDeployPipeline(BaseModel):
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

        super().__init__(path=path, **kwargs)
        from lmdeploy import pipeline
        self.model = pipeline(
            model_path=self.path, model_name=model_name, tp=tp, **pipeline_cfg)

    def generate(self,
                 inputs: Union[str, List[str]],
                 do_preprocess: bool = None,
                 skip_special_tokens: bool = False,
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
        gen_params = self.update_gen_params(**kwargs)
        gen_config = GenerationConfig(
            skip_special_tokens=skip_special_tokens, **gen_params)
        response = self.model.batch_infer(
            prompt, gen_config=gen_config, do_preprocess=do_preprocess)
        response = [resp.text for resp in response]
        # remove stop_words
        response = filter_suffix(response, self.gen_params.get('stop_words'))
        if batched:
            return response
        return response[0]


class LMDeployServer(BaseModel):
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


# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Any, Dict, Iterable, List, Optional, Union

import requests

# from lmdeploy.utils import get_logger


def get_model_list(api_url: str):
    """Get model list from api server."""
    response = requests.get(api_url)
    if hasattr(response, 'text'):
        model_list = json.loads(response.text)
        model_list = model_list.pop('data', [])
        return [item['id'] for item in model_list]
    return None


def json_loads(content):
    """Loads content to json format."""
    try:
        content = json.loads(content)
        return content
    except:  # noqa
        # logger = get_logger('lmdeploy')
        # logger.warning(f'weird json content {content}')
        return ''


class APIClient:
    """Chatbot for LLaMA series models with turbomind as inference engine.

    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
        api_key (str | None): api key. Default to None, which means no
            api key will be used.
    """

    def __init__(self,
                 api_server_url: str,
                 api_key: Optional[str] = None,
                 **kwargs):
        self.api_server_url = api_server_url
        self.chat_intractive_v1_url = f'{api_server_url}/v1/chat/interactive'
        self.chat_completions_v1_url = f'{api_server_url}/v1/chat/completions'
        self.completions_v1_url = f'{api_server_url}/v1/completions'
        self.models_v1_url = f'{api_server_url}/v1/models'
        self.encode_v1_url = f'{api_server_url}/v1/encode'
        self._available_models = None
        self.api_key = api_key
        self.headers = {'content-type': 'application/json'}
        if api_key is not None:
            self.headers['Authorization'] = f'Bearer {api_key}'

    @property
    def available_models(self):
        """Show available models."""
        if self._available_models is not None:
            return self._available_models
        response = requests.get(self.models_v1_url)
        if hasattr(response, 'text'):
            model_list = json_loads(response.text)
            model_list = model_list.pop('data', [])
            self._available_models = [item['id'] for item in model_list]
            return self._available_models
        return None

    def encode(self,
               input: Union[str, List[str]],
               do_preprocess: Optional[bool] = False,
               add_bos: Optional[bool] = True):
        """Encode prompts.

        Args:
            input: the prompt to be encoded. In str or List[str] format.
            do_preprocess: whether do preprocess or not. Default to False.
            add_bos: True when it is the beginning of a conversation. False
                when it is not. Default to True.
        Return: (input_ids, length)
        """
        response = requests.post(self.encode_v1_url,
                                 headers=self.headers,
                                 json=dict(input=input,
                                           do_preprocess=do_preprocess,
                                           add_bos=add_bos),
                                 stream=False)
        if hasattr(response, 'text'):
            output = json_loads(response.text)
            return output['input_ids'], output['length']
        return None, None

    def chat_completions_v1(self,
                            model: str,
                            messages: Union[str, List[Dict[str, str]]],
                            temperature: Optional[float] = 0.7,
                            top_p: Optional[float] = 1.0,
                            n: Optional[int] = 1,
                            max_tokens: Optional[int] = 512,
                            stop: Optional[Union[str, List[str]]] = None,
                            stream: Optional[bool] = False,
                            presence_penalty: Optional[float] = 0.0,
                            frequency_penalty: Optional[float] = 0.0,
                            user: Optional[str] = None,
                            repetition_penalty: Optional[float] = 1.0,
                            session_id: Optional[int] = -1,
                            ignore_eos: Optional[bool] = False,
                            skip_special_tokens: Optional[bool] = True,
                            **kwargs):
        """Chat completion v1.

        Args:
            model: model name. Available from self.available_models.
            messages: string prompt or chat history in OpenAI format. Chat
                history example: `[{"role": "user", "content": "hi"}]`.
            temperature (float): to modulate the next token probability
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            n (int): How many chat completion choices to generate for each
                input message. Only support one here.
            stream: whether to stream the results or not. Default to false.
            max_tokens (int): output token nums
            stop (str | List[str] | None): To stop generating further
              tokens. Only accept stop words that's encoded to one token idex.
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            session_id (int): if not specified, will set random value

        Yields:
            json objects in openai formats
        """
        pload = {
            k: v
            for k, v in locals().copy().items()
            if k[:2] != '__' and k not in ['self']
        }
        response = requests.post(self.chat_completions_v1_url,
                                 headers=self.headers,
                                 json=pload,
                                 stream=stream)
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json_loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json_loads(decoded)
                    yield output

    def chat_interactive_v1(self,
                            prompt: Union[str, List[Dict[str, str]]],
                            session_id: int = -1,
                            interactive_mode: bool = False,
                            stream: bool = False,
                            stop: Optional[Union[str, List[str]]] = None,
                            request_output_len: int = 512,
                            top_p: float = 0.8,
                            top_k: int = 40,
                            temperature: float = 0.8,
                            repetition_penalty: float = 1.0,
                            ignore_eos: bool = False,
                            skip_special_tokens: Optional[bool] = True,
                            **kwargs):
        """Interactive completions.

        - On interactive mode, the chat history is kept on the server. Please
        set `interactive_mode = True`.
        - On normal mode, no chat history is kept on the server. Set
        `interactive_mode = False`.

        Args:
            prompt: the prompt to use for the generation.
            session_id: determine which instance will be called.
                If not specified with a value other than -1, using random value
                directly.
            interactive_mode (bool): turn on interactive mode or not. On
                interactive mode, session history is kept on the server (and
                vice versa).
            stream: whether to stream the results or not.
            stop (str | List[str] | None): To stop generating further tokens.
                Only accept stop words that's encoded to one token idex.
            request_output_len (int): output token nums
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.

        Yields:
            json objects consist of text, tokens, finish_reason
        """
        pload = {
            k: v
            for k, v in locals().copy().items()
            if k[:2] != '__' and k not in ['self']
        }
        response = requests.post(self.chat_intractive_v1_url,
                                 headers=self.headers,
                                 json=pload,
                                 stream=stream)
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                output = json_loads(decoded)
                yield output

    def completions_v1(
            self,
            model: str,
            prompt: Union[str, List[Any]],
            suffix: Optional[str] = None,
            temperature: Optional[float] = 0.7,
            n: Optional[int] = 1,
            max_tokens: Optional[int] = 16,
            stream: Optional[bool] = False,
            stop: Optional[Union[str, List[str]]] = None,
            top_p: Optional[float] = 1.0,
            top_k: Optional[int] = 40,
            user: Optional[str] = None,
            # additional argument of lmdeploy
            repetition_penalty: Optional[float] = 1.0,
            session_id: Optional[int] = -1,
            ignore_eos: Optional[bool] = False,
            skip_special_tokens: Optional[bool] = True,
            **kwargs):
        """Chat completion v1.

        Args:
            model (str): model name. Available from /v1/models.
            prompt (str): the input prompt.
            suffix (str): The suffix that comes after a completion of inserted
                text.
            max_tokens (int): output token nums
            temperature (float): to modulate the next token probability
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            n (int): How many chat completion choices to generate for each
                input message. Only support one here.
            stream: whether to stream the results or not. Default to false.
            stop (str | List[str] | None): To stop generating further
              tokens. Only accept stop words that's encoded to one token idex.
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            user (str): A unique identifier representing your end-user.
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            session_id (int): if not specified, will set random value

        Yields:
            json objects in openai formats
        """
        pload = {
            k: v
            for k, v in locals().copy().items()
            if k[:2] != '__' and k not in ['self']
        }
        response = requests.post(self.completions_v1_url,
                                 headers=self.headers,
                                 json=pload,
                                 stream=stream)
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json_loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json_loads(decoded)
                    yield output

    def chat(self,
             prompt: str,
             session_id: int,
             request_output_len: int = 512,
             stream: bool = False,
             top_p: float = 0.8,
             top_k: int = 40,
             temperature: float = 0.8,
             repetition_penalty: float = 1.0,
             ignore_eos: bool = False):
        """Chat with a unique session_id.

        Args:
            prompt: the prompt to use for the generation.
            session_id: determine which instance will be called.
                If not specified with a value other than -1, using random value
                directly.
            stream: whether to stream the results or not.
            stop: whether to stop the session response or not.
            request_output_len (int): output token nums
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos

        Yields:
            text, tokens, finish_reason
        """
        assert session_id != -1, 'please set a value other than -1'
        for outputs in self.chat_interactive_v1(
                prompt,
                session_id=session_id,
                request_output_len=request_output_len,
                interactive_mode=True,
                stream=stream,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos):
            if outputs['finish_reason'] == 'length':
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
            yield outputs['text'], outputs['tokens'], outputs['finish_reason']

    def end_session(self, session_id: int):
        """End the session with a unique session_id.

        Args:
            session_id: determine which instance will be called.
                If not specified with a value other than -1, using random value
                directly.
        """
        for out in self.chat_interactive_v1(prompt='',
                                            session_id=session_id,
                                            request_output_len=0,
                                            interactive_mode=False):
            pass


def input_prompt():
    """Input a prompt in the consolo interface."""
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def get_streaming_response(
        prompt: str,
        api_url: str,
        session_id: int,
        request_output_len: int = 512,
        stream: bool = True,
        interactive_mode: bool = False,
        ignore_eos: bool = False,
        cancel: bool = False,
        top_p: float = 0.8,
        temperature: float = 0.7,
        api_key: Optional[str] = None) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'
    pload = {
        'prompt': prompt,
        'stream': stream,
        'session_id': session_id,
        'request_output_len': request_output_len,
        'interactive_mode': interactive_mode,
        'ignore_eos': ignore_eos,
        'cancel': cancel,
        'top_p': top_p,
        'temperature': temperature
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b'\n'):
        if chunk:
            data = json_loads(chunk.decode('utf-8'))
            output = data.pop('text', '')
            tokens = data.pop('tokens', 0)
            finish_reason = data.pop('finish_reason', None)
            yield output, tokens, finish_reason


# def main(api_server_url: str,
#          session_id: int = 0,
#          api_key: Optional[str] = None):
#     """Main function to chat in terminal."""
#     api_client = APIClient(api_server_url, api_key=api_key)
#     while True:
#         prompt = input_prompt()
#         if prompt in ['exit', 'end']:
#             api_client.end_session(session_id)
#             if prompt == 'exit':
#                 exit(0)
#         else:
#             for text, tokens, finish_reason in api_client.chat(
#                     prompt,
#                     session_id=session_id,
#                     request_output_len=512,
#                     stream=True):
#                 if finish_reason == 'length':
#                     continue
#                 print(text, end='')


# if __name__ == '__main__':
#     import fire

#     fire.Fire(main)

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
        BaseModel.__init__(self, path=url, **kwargs)
        # from lmdeploy.serve.openai.api_client import APIClient
        self.client = APIClient(url)
        self.model_name = model_name
