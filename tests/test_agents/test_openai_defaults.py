"""Regression coverage for the deprecated OpenAI defaults (issue #378)."""

import ast
from pathlib import Path

import pytest

from lagent.llms import AsyncGPTAPI, GPTAPI


@pytest.mark.parametrize('model_class', [GPTAPI, AsyncGPTAPI])
def test_default_openai_model_builds_a_chat_completion_request(model_class):
    """Both wrappers select the same supported, low-cost chat model by default."""
    llm = model_class(key='test-key')
    messages = [{'role': 'user', 'content': 'hello'}]
    header, payload = llm.generate_request_data(llm.model_type, messages, llm.gen_params)

    assert llm.model_type == 'gpt-4o-mini'
    assert header['content-type'] == 'application/json'
    assert payload['model'] == 'gpt-4o-mini'
    assert payload['messages'] == messages
    assert payload['max_tokens'] > 0


@pytest.mark.parametrize('model_class', [GPTAPI, AsyncGPTAPI])
def test_explicit_openai_model_still_overrides_default(model_class):
    llm = model_class(model_type='gpt-4o', key='test-key')
    assert llm.model_type == 'gpt-4o'


def test_example_configs_use_supported_models_and_constructor_keyword():
    root = Path(__file__).resolve().parents[2]
    react_source = (root / 'lagent' / 'agents' / 'react.py').read_text(encoding='utf-8')
    assert 'gpt-4o-2024-05-13' not in react_source

    example = ast.parse((root / 'examples' / 'run_async_agent_openai.py').read_text(encoding='utf-8'))
    async_llm = next(
        node.value
        for node in example.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'async_llm' for target in node.targets)
    )
    assert isinstance(async_llm, ast.Call)
    config = {keyword.arg: keyword.value for keyword in async_llm.keywords}
    assert 'model' not in config
    assert ast.literal_eval(config['model_type']) == 'gpt-4o'
