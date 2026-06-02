import json
import os
import time
import traceback
from logging import getLogger
from typing import AsyncGenerator, Dict, List, Optional, Union

from ..schema import ModelStatusCode
from ..utils import filter_suffix
from .base_api import AsyncBaseAPILLM, BaseAPILLM


class LiteLLMAPI(BaseAPILLM):
    """Model wrapper around LiteLLM's unified completion interface.

    LiteLLM provides access to 100+ LLM providers (OpenAI, Anthropic, Google,
    Azure, Bedrock, Ollama, etc.) through a single ``litellm.completion()``
    call. Use any model identifier that LiteLLM supports, e.g.
    ``"gpt-4o-mini"``, ``"anthropic/claude-sonnet-4-20250514"``,
    ``"gemini/gemini-2.5-flash"``, ``"azure/gpt-4o"``.

    See https://docs.litellm.ai/docs/providers for the full provider list.

    Args:
        model_type (str): LiteLLM model identifier.
        retry (int): Number of retries on transient errors. Defaults to 2.
        key (str or List[str]): API key(s). ``'ENV'`` lets LiteLLM read
            provider-specific env vars (``OPENAI_API_KEY``,
            ``ANTHROPIC_API_KEY``, etc.) automatically. Defaults to ``'ENV'``.
        api_base (str, optional): Override the provider's default base URL
            (useful for Azure deployments or self-hosted endpoints).
        meta_template (Dict, optional): Role-mapping meta template.
        json_mode (bool): Request JSON output format. Defaults to False.
        gen_params: Default generation parameters (``max_new_tokens``,
            ``temperature``, ``top_p``, etc.).
    """

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'gpt-4o-mini',
        retry: int = 2,
        json_mode: bool = False,
        key: Union[str, List[str]] = 'ENV',
        api_base: Optional[str] = None,
        meta_template: Optional[Dict] = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='system'),
        ],
        **gen_params,
    ):
        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params,
        )
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [None if key == 'ENV' else key]
        else:
            self.keys = key

        self.key_ctr = 0
        self.invalid_keys = set()
        self.model_type = model_type
        self.api_base = api_base
        self.json_mode = json_mode

    def _get_completion_kwargs(self, messages, gen_params):
        """Build kwargs dict for ``litellm.completion()``."""
        gen_params = gen_params.copy()

        max_tokens = min(gen_params.pop('max_new_tokens', 512), 4096)
        if max_tokens <= 0:
            return None

        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        gen_params.pop('repetition_penalty', None)
        gen_params.pop('top_k', None)
        gen_params.pop('top_p', None)
        gen_params.pop('skip_special_tokens', None)
        gen_params.pop('session_id', None)

        kwargs = {
            'model': self.model_type,
            'messages': messages,
            'drop_params': True,
            **gen_params,
        }

        if self.json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        if self.api_base:
            kwargs['api_base'] = self.api_base

        # Forward API key only when explicitly set
        key = self.keys[self.key_ctr % len(self.keys)]
        if key is not None:
            kwargs['api_key'] = key

        return kwargs

    def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): a list of messages
                or list of lists of messages
            gen_params: additional generation configuration

        Returns:
            Union[str, List[str]]: generated string(s)
        """
        assert isinstance(inputs, list)
        gen_params = {**self.gen_params, **gen_params}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat, messages, **gen_params)
                for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    def stream_chat(self, inputs: List[dict], **gen_params):
        """Generate responses with streaming."""
        assert isinstance(inputs, list)
        gen_params = self.update_gen_params(**gen_params)
        gen_params['stream'] = True

        resp = ''
        finished = False
        stop_words = gen_params.get('stop_words')
        if stop_words is None:
            stop_words = []
        messages = self.template_parser(inputs)
        for text in self._stream_chat(messages, **gen_params):
            resp += text
            if not resp:
                continue
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

    def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of messages."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                'litellm is required for LiteLLMAPI. '
                'Install it with: pip install litellm'
            ) from e

        assert isinstance(messages, list)
        messages = self.template_parser(messages)
        kwargs = self._get_completion_kwargs(messages, gen_params)
        if kwargs is None:
            return ''

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content
                return (content or '').strip()
            except Exception as error:
                errmsg = str(error) + '\n' + traceback.format_exc()
                self.logger.error(errmsg)
                qualname = f'{type(error).__module__}.{type(error).__name__}'
                if qualname in (
                    'litellm.exceptions.RateLimitError',
                    'litellm.exceptions.APIConnectionError',
                    'litellm.exceptions.Timeout',
                    'litellm.exceptions.InternalServerError',
                    'litellm.exceptions.ServiceUnavailableError',
                ):
                    time.sleep(1)
                    max_num_retries += 1
                    continue
                raise
        raise RuntimeError(
            f'Calling LiteLLM failed after retrying for '
            f'{max_num_retries} times. errmsg: {errmsg}'
        )

    def _stream_chat(self, messages: List[dict], **gen_params):
        """Generate streaming completion."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                'litellm is required for LiteLLMAPI. '
                'Install it with: pip install litellm'
            ) from e

        kwargs = self._get_completion_kwargs(messages, gen_params)
        if kwargs is None:
            return
        kwargs['stream'] = True

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            try:
                response = litellm.completion(**kwargs)
                for chunk in response:
                    choice = chunk.choices[0]
                    if choice.finish_reason == 'stop':
                        return
                    delta = choice.delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as error:
                errmsg = str(error) + '\n' + traceback.format_exc()
                self.logger.error(errmsg)
                qualname = f'{type(error).__module__}.{type(error).__name__}'
                if qualname in (
                    'litellm.exceptions.RateLimitError',
                    'litellm.exceptions.APIConnectionError',
                    'litellm.exceptions.Timeout',
                    'litellm.exceptions.InternalServerError',
                    'litellm.exceptions.ServiceUnavailableError',
                ):
                    time.sleep(1)
                    max_num_retries += 1
                    continue
                raise
        raise RuntimeError(
            f'Calling LiteLLM failed after retrying for '
            f'{max_num_retries} times. errmsg: {errmsg}'
        )


class AsyncLiteLLMAPI(AsyncBaseAPILLM):
    """Async version of :class:`LiteLLMAPI`.

    Uses ``litellm.acompletion()`` for non-blocking calls.
    """

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'gpt-4o-mini',
        retry: int = 2,
        json_mode: bool = False,
        key: Union[str, List[str]] = 'ENV',
        api_base: Optional[str] = None,
        meta_template: Optional[Dict] = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='system'),
        ],
        **gen_params,
    ):
        super().__init__(
            model_type=model_type,
            meta_template=meta_template,
            retry=retry,
            **gen_params,
        )
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [None if key == 'ENV' else key]
        else:
            self.keys = key

        self.key_ctr = 0
        self.invalid_keys = set()
        self.model_type = model_type
        self.api_base = api_base
        self.json_mode = json_mode

    def _get_completion_kwargs(self, messages, gen_params):
        """Build kwargs dict for ``litellm.acompletion()``."""
        gen_params = gen_params.copy()

        max_tokens = min(gen_params.pop('max_new_tokens', 512), 4096)
        if max_tokens <= 0:
            return None

        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        gen_params.pop('repetition_penalty', None)
        gen_params.pop('top_k', None)
        gen_params.pop('top_p', None)
        gen_params.pop('skip_special_tokens', None)
        gen_params.pop('session_id', None)

        kwargs = {
            'model': self.model_type,
            'messages': messages,
            'drop_params': True,
            **gen_params,
        }

        if self.json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        if self.api_base:
            kwargs['api_base'] = self.api_base

        key = self.keys[self.key_ctr % len(self.keys)]
        if key is not None:
            kwargs['api_key'] = key

        return kwargs

    async def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        session_ids: Union[int, List[int]] = None,
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts."""
        import asyncio

        assert isinstance(inputs, list)
        gen_params = {**self.gen_params, **gen_params}
        tasks = [
            self._chat(messages, **gen_params)
            for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
        ]
        ret = await asyncio.gather(*tasks)
        return ret[0] if isinstance(inputs[0], dict) else ret

    async def stream_chat(self, inputs: List[dict], **gen_params):
        """Generate responses with streaming."""
        assert isinstance(inputs, list)
        gen_params = self.update_gen_params(**gen_params)
        gen_params['stream'] = True

        resp = ''
        finished = False
        stop_words = gen_params.get('stop_words')
        if stop_words is None:
            stop_words = []
        messages = self.template_parser(inputs)
        async for text in self._stream_chat(messages, **gen_params):
            resp += text
            if not resp:
                continue
            for sw in stop_words:
                if sw in resp:
                    resp = filter_suffix(resp, stop_words)
                    finished = True
                    break
            yield ModelStatusCode.STREAM_ING, resp, None
            if finished:
                break
        yield ModelStatusCode.END, resp, None

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of messages."""
        import asyncio

        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                'litellm is required for AsyncLiteLLMAPI. '
                'Install it with: pip install litellm'
            ) from e

        assert isinstance(messages, list)
        messages = self.template_parser(messages)
        kwargs = self._get_completion_kwargs(messages, gen_params)
        if kwargs is None:
            return ''

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            try:
                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content
                return (content or '').strip()
            except Exception as error:
                errmsg = str(error) + '\n' + traceback.format_exc()
                self.logger.error(errmsg)
                qualname = f'{type(error).__module__}.{type(error).__name__}'
                if qualname in (
                    'litellm.exceptions.RateLimitError',
                    'litellm.exceptions.APIConnectionError',
                    'litellm.exceptions.Timeout',
                    'litellm.exceptions.InternalServerError',
                    'litellm.exceptions.ServiceUnavailableError',
                ):
                    await asyncio.sleep(1)
                    max_num_retries += 1
                    continue
                raise
        raise RuntimeError(
            f'Calling LiteLLM failed after retrying for '
            f'{max_num_retries} times. errmsg: {errmsg}'
        )

    async def _stream_chat(self, messages: List[dict], **gen_params) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        import asyncio

        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                'litellm is required for AsyncLiteLLMAPI. '
                'Install it with: pip install litellm'
            ) from e

        kwargs = self._get_completion_kwargs(messages, gen_params)
        if kwargs is None:
            return
        kwargs['stream'] = True

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            try:
                response = await litellm.acompletion(**kwargs)
                async for chunk in response:
                    choice = chunk.choices[0]
                    if choice.finish_reason == 'stop':
                        return
                    delta = choice.delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as error:
                errmsg = str(error) + '\n' + traceback.format_exc()
                self.logger.error(errmsg)
                qualname = f'{type(error).__module__}.{type(error).__name__}'
                if qualname in (
                    'litellm.exceptions.RateLimitError',
                    'litellm.exceptions.APIConnectionError',
                    'litellm.exceptions.Timeout',
                    'litellm.exceptions.InternalServerError',
                    'litellm.exceptions.ServiceUnavailableError',
                ):
                    await asyncio.sleep(1)
                    max_num_retries += 1
                    continue
                raise
        raise RuntimeError(
            f'Calling LiteLLM failed after retrying for '
            f'{max_num_retries} times. errmsg: {errmsg}'
        )
