"""Model wrappers around [OrcaRouter](https://www.orcarouter.ai).

OrcaRouter is a gateway that fronts open-weight models from many vendors
through a single OpenAI-compatible endpoint. These wrappers mirror the
``GPTAPI`` / ``AsyncGPTAPI`` wrappers with OrcaRouter defaults so that
Lagent agents can route to the gateway as a named provider.
"""

import os
import warnings
from typing import Dict, List, Optional, Union

from .openai import AsyncGPTAPI, GPTAPI

ORCAROUTER_API_BASE = 'https://api.orcarouter.ai/v1/chat/completions'
ORCAROUTER_API_KEY_ENV = 'ORCAROUTER_API_KEY'
ORCAROUTER_DEFAULT_MODEL = 'orcarouter/auto'

_DEFAULT_META_TEMPLATE = [
    dict(role='system', api_role='system'),
    dict(role='user', api_role='user'),
    dict(role='assistant', api_role='assistant'),
    dict(role='environment', api_role='system'),
]


def _gateway_request_data(model_type, messages, gen_params, json_mode=False):
    """Build the request payload for gateway models.

    OrcaRouter is a multi-vendor gateway: accept gateway aliases
    (``orcarouter/*``) as well as vendor-qualified model names
    (``vendor/model``) by routing them through the generic Chat Completions
    path. Return ``None`` so the caller can fall back to the OpenAI wrapper's
    validation for any other model.
    """
    if not (model_type.lower().startswith('orcarouter') or '/' in model_type):
        return None

    gen_params = gen_params.copy()
    max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
    if max_tokens <= 0:
        return '', ''

    header = {
        'content-type': 'application/json',
    }

    gen_params['max_tokens'] = max_tokens
    if 'stop_words' in gen_params:
        gen_params['stop'] = gen_params.pop('stop_words')
    if 'repetition_penalty' in gen_params:
        gen_params['frequency_penalty'] = gen_params.pop('repetition_penalty')
    if 'top_k' in gen_params:
        warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.', DeprecationWarning)
        gen_params.pop('top_k')
    gen_params.pop('skip_special_tokens', None)
    gen_params.pop('session_id', None)

    data = {'model': model_type, 'messages': messages, 'n': 1, **gen_params}
    if json_mode:
        data['response_format'] = {'type': 'json_object'}
    return header, data


class OrcaRouterAPI(GPTAPI):
    """Model wrapper around OrcaRouter's OpenAI-compatible gateway.

    Args:
        model_type (str): The model to use. Defaults to 'orcarouter/auto',
            which routes to a suitable open-weight model on the gateway.
        retry (int): Number of retries if the API call fails. Defaults to 2.
        key (str or List[str]): OrcaRouter key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $ORCAROUTER_API_KEY (keys start with ``sk-orca-``).
            If it's a list, the keys will be used in round-robin manner.
            Defaults to 'ENV'.
        org (str or List[str], optional): OpenAI organization(s), passed
            through for compatibility. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt template.
        api_base (str): The base url of OrcaRouter's OpenAI-compatible API.
            Defaults to 'https://api.orcarouter.ai/v1/chat/completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(
        self,
        model_type: str = ORCAROUTER_DEFAULT_MODEL,
        retry: int = 2,
        json_mode: bool = False,
        key: Union[str, List[str]] = 'ENV',
        org: Optional[Union[str, List[str]]] = None,
        meta_template: Optional[Dict] = _DEFAULT_META_TEMPLATE,
        api_base: str = ORCAROUTER_API_BASE,
        proxies: Optional[Dict] = None,
        **gen_params,
    ):
        super().__init__(
            model_type=model_type,
            retry=retry,
            json_mode=json_mode,
            key=key,
            org=org,
            meta_template=meta_template,
            api_base=api_base,
            proxies=proxies,
            **gen_params,
        )
        # Read OrcaRouter keys from the dedicated env var instead of
        # $OPENAI_API_KEY.
        if isinstance(key, str):
            self.keys = [os.getenv(ORCAROUTER_API_KEY_ENV) if key == 'ENV' else key]
        else:
            self.keys = key

    def generate_request_data(self, model_type, messages, gen_params, json_mode=False):
        result = _gateway_request_data(model_type, messages, gen_params, json_mode)
        if result is not None:
            return result
        return super().generate_request_data(model_type, messages, gen_params, json_mode)


class AsyncOrcaRouterAPI(AsyncGPTAPI):
    """Asynchronous variant of :class:`OrcaRouterAPI`.

    Args:
        model_type (str): The model to use. Defaults to 'orcarouter/auto'.
        retry (int): Number of retries if the API call fails. Defaults to 2.
        key (str or List[str]): OrcaRouter key(s). When set to "ENV", the key
            is fetched from the environment variable $ORCAROUTER_API_KEY.
            Defaults to 'ENV'.
        org (str or List[str], optional): OpenAI organization(s), passed
            through for compatibility. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt template.
        api_base (str): The base url of OrcaRouter's OpenAI-compatible API.
            Defaults to 'https://api.orcarouter.ai/v1/chat/completions'.
        gen_params: Default generation configuration which could be overridden
            on the fly of generation.
    """

    is_api: bool = True

    def __init__(
        self,
        model_type: str = ORCAROUTER_DEFAULT_MODEL,
        retry: int = 2,
        json_mode: bool = False,
        key: Union[str, List[str]] = 'ENV',
        org: Optional[Union[str, List[str]]] = None,
        meta_template: Optional[Dict] = _DEFAULT_META_TEMPLATE,
        api_base: str = ORCAROUTER_API_BASE,
        proxies: Optional[Dict] = None,
        **gen_params,
    ):
        super().__init__(
            model_type=model_type,
            retry=retry,
            json_mode=json_mode,
            key=key,
            org=org,
            meta_template=meta_template,
            api_base=api_base,
            proxies=proxies,
            **gen_params,
        )
        # Read OrcaRouter keys from the dedicated env var instead of
        # $OPENAI_API_KEY.
        if isinstance(key, str):
            self.keys = [os.getenv(ORCAROUTER_API_KEY_ENV) if key == 'ENV' else key]
        else:
            self.keys = key

    def generate_request_data(self, model_type, messages, gen_params, json_mode=False):
        result = _gateway_request_data(model_type, messages, gen_params, json_mode)
        if result is not None:
            return result
        return super().generate_request_data(model_type, messages, gen_params, json_mode)
