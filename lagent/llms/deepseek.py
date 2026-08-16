# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Union

from .openai import AsyncGPTAPI, GPTAPI


DEEPSEEK_API_BASE = 'https://api.deepseek.com/v1/chat/completions'


class DeepSeekAPI(GPTAPI):
    """OpenAI-compatible wrapper for DeepSeek Chat APIs.

    DeepSeek exposes an OpenAI-compatible chat completions endpoint. This class
    defaults ``api_base`` and env key handling so DeepSeek works without manually
    configuring ``GPTAPI``.

    Example:
        ```python
        from lagent.llms import DeepSeekAPI
        llm = DeepSeekAPI(model_type='deepseek-chat', key='ENV')
        ```
    """

    def __init__(
        self,
        model_type: str = 'deepseek-chat',
        key: Union[str, List[str]] = 'ENV',
        api_base: str = DEEPSEEK_API_BASE,
        **gen_params,
    ):
        if key == 'ENV':
            env_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
            if env_key:
                key = env_key
        super().__init__(model_type=model_type, key=key, api_base=api_base, **gen_params)


class AsyncDeepSeekAPI(AsyncGPTAPI):
    """Async OpenAI-compatible wrapper for DeepSeek Chat APIs."""

    def __init__(
        self,
        model_type: str = 'deepseek-chat',
        key: Union[str, List[str]] = 'ENV',
        api_base: str = DEEPSEEK_API_BASE,
        **gen_params,
    ):
        if key == 'ENV':
            env_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
            if env_key:
                key = env_key
        super().__init__(model_type=model_type, key=key, api_base=api_base, **gen_params)
