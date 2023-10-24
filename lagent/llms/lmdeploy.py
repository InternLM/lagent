import dataclasses
import os.path as osp
import random

import lmdeploy.turbomind.chat as tm_chat
from lmdeploy.serve.turbomind.chatbot import Chatbot, Session, get_logger

from .base_llm import BaseModel


class TritonClient(Chatbot, BaseModel):

    def __init__(self, meta_template=None, **kwargs):
        """TritonClient is a wrapper of TritonClient for LLM.

        Args:
            model_name (str): the name of the model
            max_out_len (int): the expected generated token numbers
            log_level (str): log level
        """
        BaseModel.__init__(self, meta_template=meta_template, path=None)
        Chatbot.__init__(self, **kwargs)

    def generate(self,
                 prompt: str,
                 session_id: int = 2967,
                 request_id: str = '',
                 max_out_len: int = None,
                 sequence_start: bool = True,
                 sequence_end: bool = True,
                 *args,
                 **kwargs):
        """Start a new round conversation of a session. Return the chat
        completions in non-stream mode.

        Args:
            session_id (int): the identical id of a session
            prompt (str): user's prompt in this round conversation
            request_id (str): the identical id of this round conversation
            max_out_len (int): the expected generated token numbers
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session

        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
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

        status, res, _ = None, '', 0
        for status, res, _ in self._stream_infer(self._session, prompt,
                                                 max_out_len, sequence_start,
                                                 sequence_end):
            if status.value < 0:
                break
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
            return res
        else:
            return ''

    def generate_from_template(self, templates, max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates)
        response = self.generate(inputs, max_out_len=max_out_len, **kwargs)
        # The return of tuibomind contains <eoa>, here we hard code removes it.
        response = response.replace(
            self.template_parser.roles['assistant']['end'].strip(),
            '').strip()
        return response


class TurboMind(BaseModel):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 meta_template=None,
                 tp=1,
                 **kwargs):
        super().__init__(path, max_seq_len, tokenizer_only, meta_template)
        tokenizer_model_path = osp.join(path, 'triton_models', 'tokenizer')
        self.tokenizer = tm_chat.Tokenizer(tokenizer_model_path)
        self.tm_model = tm_chat.tm.TurboMind(
            path, eos_id=self.tokenizer.eos_token_id, tp=tp)
        self.generator = self.tm_model.create_instance()

        model_name = self.tm_model.model_name
        self.model = tm_chat.MODELS.get(model_name)(
            capability='completion', **kwargs)

    def generate(self, prompt, **kwargs):
        seed = random.getrandbits(64)
        input_ids = self.tokenizer.encode(prompt)
        gen_param = tm_chat.get_gen_param(
            'completion', self.model.sampling_param, step=0, nth_round=1)

        response_size = 0
        for outputs in self.generator.stream_infer(
                session_id=1,
                input_ids=[input_ids],
                stream_output=False,
                **dataclasses.asdict(gen_param),
                ignore_eos=False,
                random_seed=seed):
            res, tokens = outputs[0]
            # decode res
            response = self.tokenizer.decode(
                res.tolist(), offset=response_size)
            response = tm_chat.valid_str(response)

        return response
