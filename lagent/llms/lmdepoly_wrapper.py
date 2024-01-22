import json
import requests
from typing import List, Optional, Union

import mmengine

from lagent.llms.base_llm import BaseModel
from lagent.schema import AgentStatusCode
from lagent.utils.util import filter_suffix


class TritonClient(BaseModel):
    """TritonClient is a wrapper of TritonClient for LLM.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
            triton inference server
        model_name (str): the name of the model
        session_len (int): the context size
        max_out_len (int): the expected generated token numbers
    """

    def __init__(self,
                 tritonserver_addr: str,
                 model_name: str,
                 session_len: int = 32768,
                 **kwargs):
        super().__init__(path=None, **kwargs)
        from lmdeploy.serve.turbomind.chatbot import Chatbot, StatusCode
        self.state_map = {
            StatusCode.TRITON_STREAM_END: AgentStatusCode.END,
            StatusCode.TRITON_SERVER_ERR: AgentStatusCode.SERVER_ERR,
            StatusCode.TRITON_SESSION_CLOSED: AgentStatusCode.SESSION_CLOSED,
            StatusCode.TRITON_STREAM_ING: AgentStatusCode.STREAM_ING,
            StatusCode.TRITON_SESSION_OUT_OF_LIMIT:
            AgentStatusCode.SESSION_OUT_OF_LIMIT,
            StatusCode.TRITON_SESSION_INVALID_ARG:
            AgentStatusCode.SESSION_INVALID_ARG,
            StatusCode.TRITON_SESSION_READY: AgentStatusCode.SESSION_READY
        }
        self.chatbot = Chatbot(
            tritonserver_addr=tritonserver_addr,
            model_name=model_name,
            session_len=session_len,
            **kwargs)

    def generate(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 request_id: str = '',
                 max_out_len: int = None,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            inputs (str, List[str]): user's prompt(s) in this round
            session_id (int): the identical id of a session
            request_id (str): the identical id of this round conversation
            max_out_len (int): the expected generated token numbers
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session

        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.serve.turbomind.chatbot import Session, get_logger
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs

        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'max_out_len {max_out_len}')

        if self._session is None:
            sequence_start = True
            self._session = Session(session_id=session_id)
        elif self._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            return ''

        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''

        self.chatbot.cfg = self._update_gen_params(
            max_out_len=max_out_len, **kwargs)

        status, res, _ = None, '', 0
        for status, res, _ in self.chatbot._stream_infer(
                self._session, prompt, max_out_len, sequence_start,
                sequence_end):
            if status.value < 0:
                break
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
            if not batched:
                return res[0]
            return res
        else:
            return ''

    def stream_chat(self,
                    inputs: List[dict],
                    session_id: int = 2967,
                    request_id: str = '',
                    max_out_len: int = None,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            session_id (int): the identical id of a session
            inputs (List[dict]): user's inputs in this round conversation
            request_id (str): the identical id of this round conversation
            max_out_len (int): the expected generated token numbers
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session

        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        from lmdeploy.serve.turbomind.chatbot import Session, StatusCode, get_logger
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'max_out_len {max_out_len}')

        if self._session is None:
            sequence_start = True
            self._session = Session(session_id=session_id)
        elif self._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            return ''

        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''

        self.chatbot.cfg = self._update_gen_params(
            max_out_len=max_out_len, **kwargs)
        prompt = self.template_parser(inputs)

        status, res, _ = None, '', 0
        for status, res, _ in self.chatbot._stream_infer(
                self._session, prompt, max_out_len, sequence_start,
                sequence_end):
            if status == StatusCode.TRITON_STREAM_END:  # remove stop_words
                res = filter_suffix(res, self.model.stop_words)
            if status.value < 0:
                break
            else:
                yield self.state_map.get(status), res, _
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
            yield self.state_map.get(status), res, _
        else:
            return ''

    def _update_gen_params(self, **kwargs):
        new_gen_params = self.update_gen_params(**kwargs)
        cfg = mmengine.Config(
            dict(
                session_len=self.chatbot.session_len,
                bad_words=self.chatbot.bad_words,
                **new_gen_params))
        return cfg


class LMDeployClient(BaseModel):
    """

    Args:
        path (str): The path to the model.
        url (str):
    """

    def __init__(self, path: str, url: str, **kwargs):
        super().__init__(path=path, **kwargs)
        self.url = url

    def generate(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 ignore_eos: bool = False,
                 timeout: int = 30,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or List]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.

        Returns:
            Union[str, List[str]]: (A list of) generated strings.
        """
        from lmdeploy.serve.openai.api_client import APIClient
        if getattr(self, 'client', None) is None:
            self.client = APIClient(self.url)

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False

        gen_params = self.update_gen_params(**kwargs)

        resp = [""] * len(inputs)
        for text in self.client.completions_v1(
            self.path,
            inputs,
            session_id=session_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            stream=False,
            ignore_eos=ignore_eos,
            timeout=timeout,
            **gen_params
        ):
            resp = [resp[i] + item['text'] for i, item in enumerate(text['choices'])]
        # remove stop_words
        resp = filter_suffix(resp, self.stop_words)
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
                    timeout: int = 30,
                    **kwargs):
        from lmdeploy.serve.openai.api_client import APIClient
        if getattr(self, 'client', None) is None:
            self.client = APIClient(self.url)

        gen_params = self.update_gen_params(**kwargs)

        resp = ""
        finished = False
        for text in self.client.completions_v1(
            self.path,
            inputs,
            session_id=session_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            stream=stream,
            ignore_eos=ignore_eos,
            timeout=timeout,
            **gen_params
        ):
            resp += text['choices'][0]['text']
            if not resp:
                continue
            # remove stop_words
            for sw in self.stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, self.stop_words)
                    finished = True
                    break
            yield AgentStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield AgentStatusCode.END, resp, None


class LMDeployPipeline(BaseModel):
    """

    Args:
        path (str): The path to the model.
        model_name (str): the name of the model
        tp (int):
        pipeline_cfg (dict):
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
            model_path=self.path,
            model_name=model_name,
            tp=tp,
            **pipeline_cfg)

    def generate(self,
                 inputs: Union[str, List[str]],
                 do_preprocess=None,
                 **kwargs):
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        response = self.model.batch_infer(
            prompt, do_preprocess=do_preprocess, **gen_params)
        if batched:
            return response
        return response[0]


class LMDeployServer(BaseModel):
    """

    Args:
        path (str): The path to the model.
        model_name (str): the name of the model
        server_name (str):
        server_port (int):
        tp (int):
        log_level (str):
    """

    def __init__(self,
                 path: str,
                 model_name: Optional[str] = None,
                 server_name: str = '0.0.0.0',
                 server_port: int = 23333,
                 tp: int = 1,
                 log_level: str = 'ERROR',
                 **serve_cfg):
        super().__init__(path=path, **serve_cfg)
        import lmdeploy
        self.model = lmdeploy.serve(
            model_path=self.path,
            model_name=model_name,
            server_name=server_name,
            server_port=server_port,
            tp=tp,
            log_level=log_level,
            **serve_cfg)

    def generate(self, inputs: Union[str, List[str]], **kwargs):
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        response = None
        for chunk in self.model.completions_v1(prompt, **gen_params):
            response = chunk
        if batched:
            return response
        return response[0]
