import json
import requests
from typing import List, Optional, Union

import mmengine

from lagent.llms.base_llm import BaseModel
from lagent.schema import STATE_MAP
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
        from lmdeploy.serve.turbomind.chatbot import Chatbot
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

        self.chatbot.cfg = self._update_completion_params(
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

        self.chatbot.cfg = self._update_completion_params(
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
                yield STATE_MAP.get(status), res, _
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
            yield STATE_MAP.get(status), res, _
        else:
            return ''

    def _update_completion_params(self, **kwargs):
        new_completion_params = self.update_completion_params(**kwargs)
        cfg = mmengine.Config(
            dict(
                session_len=self.chatbot.session_len,
                bad_words=self.chatbot.bad_words,
                **new_completion_params))
        return cfg


class LMDeployServerAPI(BaseModel):
    """TritonClient is a wrapper of TritonClient for LLM.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of triton
            inference server
        model_name (str): the name of the model
        session_len (int): the context size
        max_out_len (int): the expected generated token numbers
    """

    def __init__(self, path: str, url: str, retry=2, **kwargs):
        super().__init__(path=path, **kwargs)

        self.retry = retry
        self.url = url

    def generate(self,
                 inputs: Union[str, List[str]],
                 session_id: int = 2967,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
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
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs

        max_num_retries = 0
        completion_params = self.update_completion_params(**kwargs)
        while max_num_retries < self.retry:

            header = {
                'content-type': 'application/json',
            }
            session_id = (session_id + 1) % 1000000

            try:
                data = dict(
                    model=self.path,
                    session_id=session_id,
                    prompt=prompt,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    **completion_params)
                raw_response = requests.post(
                    self.url,
                    headers=header,
                    data=json.dumps(data),
                    timeout=timeout)
            except requests.ConnectionError:
                print('Got connection error, retrying...')
                max_num_retries += 1
                continue
            try:
                # import pdb
                # pdb.set_trace()
                response = raw_response.json()
            except requests.JSONDecodeError:
                print('JsonDecode error, got', str(raw_response.content))
                max_num_retries += 1
                continue
            try:
                if 'completion' in self.url:
                    if batched:
                        return [
                            item['text'].strip()
                            for item in response['choices']
                        ]
                    return response['choices'][0]['text'].strip()
                else:
                    assert not batched
                    return response['text'].strip()
            except KeyError:
                max_num_retries += 1
                pass

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def stream_chat(self,
                    inputs: List[dict],
                    session_id=0,
                    sequence_start: bool = True,
                    sequence_end: bool = True,
                    stream: bool = True,
                    ignore_eos: bool = False,
                    timeout: int = 30,
                    **kwargs):
        from lmdeploy.serve.turbomind.chatbot import StatusCode
        header = {
            'content-type': 'application/json',
        }
        session_id = (session_id + 1) % 1000000

        prompt = self.template_parser(inputs)
        completion_params = self.update_completion_params(**kwargs)
        data = dict(
            model=self.path,
            session_id=session_id,  #
            prompt=prompt,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            stream=stream,
            ignore_eos=ignore_eos,
            **completion_params)
        response = requests.post(
            f'{self.url}/v1/completions',
            headers=header,
            data=json.dumps(data),
            stream=stream,
            timeout=timeout)
        resp = ''
        finished = False
        for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded == 'data: [DONE]':
                    continue
                if decoded == '[DONE]':
                    continue
                if decoded[:6] == 'data: ':
                    decoded = decoded[6:]
                try:
                    output = json.loads(decoded)
                except Exception as e:
                    yield STATE_MAP.get(
                        StatusCode.TRITON_SERVER_ERR), str(e), None
                    return
                resp += output['choices'][0]['text']
                if not resp:
                    continue
                # remove stop_words
                for sw in self.stop_words:
                    if sw in resp:
                        resp = filter_suffix(resp, self.stop_words)
                        finished = True
                        break
                yield STATE_MAP.get(StatusCode.TRITON_STREAM_ING), resp, None
                if finished:
                    break
        yield STATE_MAP.get(StatusCode.TRITON_STREAM_END), resp, None

        # from lmdeploy.serve.openai.api_client import APIClient
        # if getattr(self, 'client', None) is None:
        #     self.client = APIClient(self.url)
        # resp = ""
        # finished = False
        # for text in self.client.completions_v1(
        #         self.path,
        #         inputs,
        #         session_id=self._session_id,
        #         max_tokens=512,
        #         stream=True,
        #         ignore_eos=False):
        #     resp += text['choices'][0]['text']
        #     if not resp:
        #         continue
        #     # remove stop_words
        #     for sw in self.stop_words:
        #         if sw in resp:
        #             resp = filter_suffix(resp, self.stop_words)
        #             finished = True
        #             break
        #     yield StatusCode.TRITON_STREAM_ING, resp, None
        #     if finished:
        #         break
        # yield StatusCode.TRITON_STREAM_END, resp, None


class LMDeployPipeline(BaseModel):
    """TritonClient is a wrapper of TritonClient for LLM.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
            triton inference server
        model_name (str): the name of the model
        session_len (int): the context size
        max_out_len (int): the expected generated token numbers
    """

    def __init__(self,
                 path,
                 model_name=None,
                 instance_num: int = 4,
                 tp: int = 1,
                 pipeline_cfg=dict(),
                 **kwargs):

        super().__init__(path=path, **kwargs)
        from lmdeploy import pipeline
        self.model = pipeline(
            model_path=self.path,
            model_name=model_name,
            instance_num=instance_num,
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
        completion_params = self.update_completion_params(**kwargs)
        response = self.model.batch_infer(
            prompt, do_preprocess=do_preprocess, **completion_params)
        if batched:
            return response
        return response[0]


class LMDeployServer(BaseModel):
    """TritonClient is a wrapper of TritonClient for LLM.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
            triton inference server
        model_name (str): the name of the model
        session_len (int): the context size
        max_out_len (int): the expected generated token numbers
    """

    def __init__(self,
                 path: str,
                 model_name: Optional[str] = None,
                 server_name: str = '0.0.0.0',
                 server_port: int = 23333,
                 instance_num: int = 64,
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
            instance_num=instance_num,
            tp=tp,
            log_level=log_level,
            **serve_cfg)

    def generate(self, inputs: Union[str, List[str]], **kwargs):
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        completion_params = self.update_completion_params(**kwargs)
        response = None
        for chunk in self.model.completions_v1(prompt, **completion_params):
            response = chunk
        if batched:
            return response
        return response[0]
