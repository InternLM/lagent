import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lagent.llms.litellm_llm import AsyncLiteLLMAPI, LiteLLMAPI

_CHAT_PATH = 'lagent.llms.litellm_llm.LiteLLMAPI._chat'


def _resp(content='hello'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _null_resp():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
    )


class TestLiteLLMAPI:
    def test_chat_dispatches_correctly(self):
        model = LiteLLMAPI(model_type='anthropic/claude-sonnet-4-20250514')
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch(_CHAT_PATH, return_value='hello') as mock:
            result = model.chat(messages)
        assert result == 'hello'

    def test_chat_batch(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        batch = [
            [{'role': 'user', 'content': 'a'}],
            [{'role': 'user', 'content': 'b'}],
        ]
        with patch(_CHAT_PATH, return_value='ok'):
            results = model.chat(batch)
        assert len(results) == 2

    def test_null_response_returns_empty_string(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch('litellm.completion', return_value=_null_resp()):
            result = model._chat(messages)
        assert result == ''

    def test_response_stripped(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch('litellm.completion', return_value=_resp('  hello  ')):
            result = model._chat(messages)
        assert result == 'hello'

    def test_drop_params_always_set(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert kwargs['drop_params'] is True

    def test_api_key_forwarded_when_set(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini', key='sk-test')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert kwargs['api_key'] == 'sk-test'

    def test_api_key_omitted_when_env(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini', key='ENV')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert 'api_key' not in kwargs

    def test_api_base_forwarded(self):
        model = LiteLLMAPI(
            model_type='azure/gpt-4o',
            api_base='https://my-resource.openai.azure.com',
        )
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert kwargs['api_base'] == 'https://my-resource.openai.azure.com'

    def test_json_mode(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini', json_mode=True)
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert kwargs['response_format'] == {'type': 'json_object'}

    def test_gen_params_translated(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 200, 'stop_words': ['END'], 'top_k': 5, 'repetition_penalty': 1.1},
        )
        assert kwargs['max_tokens'] == 200
        assert kwargs['stop'] == ['END']
        assert 'stop_words' not in kwargs
        assert 'top_k' not in kwargs
        assert 'repetition_penalty' not in kwargs

    def test_zero_max_tokens_returns_none(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 0},
        )
        assert kwargs is None

    def test_exception_propagates(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch('litellm.completion', side_effect=ValueError('bad')):
            with pytest.raises(ValueError, match='bad'):
                model._chat(messages)

    def test_import_error(self):
        model = LiteLLMAPI(model_type='gpt-4o-mini')
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch.dict('sys.modules', {'litellm': None}):
            with pytest.raises(ImportError, match='litellm is required'):
                model._chat(messages)

    def test_registered_in_init(self):
        from lagent.llms import LiteLLMAPI as Imported, AsyncLiteLLMAPI as AsyncImported

        assert Imported is LiteLLMAPI
        assert AsyncImported is AsyncLiteLLMAPI


class TestAsyncLiteLLMAPI:
    def test_init(self):
        model = AsyncLiteLLMAPI(model_type='anthropic/claude-sonnet-4-20250514', key='sk-test')
        assert model.model_type == 'anthropic/claude-sonnet-4-20250514'
        assert model.keys == ['sk-test']

    def test_completion_kwargs(self):
        model = AsyncLiteLLMAPI(model_type='gpt-4o-mini', api_base='http://localhost:4000')
        kwargs = model._get_completion_kwargs(
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 100},
        )
        assert kwargs['drop_params'] is True
        assert kwargs['api_base'] == 'http://localhost:4000'


@pytest.mark.skipif(
    'ANTHROPIC_FOUNDRY_API_KEY' not in os.environ,
    reason='Live E2E requires ANTHROPIC_FOUNDRY_API_KEY',
)
class TestLiveE2E:
    def test_live_chat(self):
        model = LiteLLMAPI(
            model_type='anthropic/' + os.environ.get('ANTHROPIC_DEFAULT_SONNET_MODEL', 'claude-sonnet-4-20250514'),
            key=os.environ['ANTHROPIC_FOUNDRY_API_KEY'],
            api_base=os.environ.get('ANTHROPIC_FOUNDRY_BASE_URL'),
            temperature=0.7,
        )
        result = model.chat([{'role': 'user', 'content': 'Say OK and nothing else.'}])
        assert isinstance(result, str)
        assert len(result) > 0
        print(f'Live E2E response: {result!r}')
