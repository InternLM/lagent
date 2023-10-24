from lagent.utils import is_module_exist
from .base_api import BaseAPIModel
from .base_llm import BaseModel
from .openai import GPTAPI

__all__ = ['BaseModel', 'BaseAPIModel', 'GPTAPI']

if is_module_exist('transformers'):
    from .huggingface import HFTransformer, HFTransformerCasualLM  # noqa: F401
    __all__.extend(['HFTransformer', 'HFTransformerCasualLM'])

if is_module_exist('lmdeploy'):
    from .lmdeploy import TritonClient, TurboMind  # noqa: F401
    __all__.extend(['TritonClient', 'TurboMind'])
