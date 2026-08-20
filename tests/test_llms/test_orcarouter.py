import os
import unittest
from unittest import mock

from lagent.llms.orcarouter import (
    ORCAROUTER_API_BASE,
    ORCAROUTER_DEFAULT_MODEL,
    ORCAROUTER_API_KEY_ENV,
    AsyncOrcaRouterAPI,
    OrcaRouterAPI,
)


class FakeResponse:

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class TestOrcaRouterAPI(unittest.TestCase):

    def tearDown(self):
        os.environ.pop(ORCAROUTER_API_KEY_ENV, None)

    def test_defaults(self):
        llm = OrcaRouterAPI()
        self.assertEqual(llm.url, ORCAROUTER_API_BASE)
        self.assertEqual(llm.model_type, ORCAROUTER_DEFAULT_MODEL)

    def test_key_from_env(self):
        os.environ[ORCAROUTER_API_KEY_ENV] = 'sk-orca-test'
        llm = OrcaRouterAPI(key='ENV')
        self.assertEqual(llm.keys, ['sk-orca-test'])

    def test_explicit_key(self):
        llm = OrcaRouterAPI(key='sk-orca-explicit')
        self.assertEqual(llm.keys, ['sk-orca-explicit'])

    def test_generate_request_data_gateway_model(self):
        llm = OrcaRouterAPI()
        header, data = llm.generate_request_data(
            'orcarouter/auto',
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 512},
        )
        self.assertEqual(data['model'], 'orcarouter/auto')
        self.assertEqual(data['messages'], [{'role': 'user', 'content': 'hi'}])
        self.assertEqual(data['max_tokens'], 512)
        self.assertNotIn('response_format', data)

    def test_generate_request_data_json_mode(self):
        llm = OrcaRouterAPI(json_mode=True)
        _, data = llm.generate_request_data(
            'orcarouter/auto',
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 512},
            json_mode=True,
        )
        self.assertEqual(data['response_format'], {'type': 'json_object'})

    def test_generate_request_data_vendor_qualified_model(self):
        llm = OrcaRouterAPI()
        _, data = llm.generate_request_data(
            'deepseek/deepseek-v4-pro',
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 512},
        )
        self.assertEqual(data['model'], 'deepseek/deepseek-v4-pro')

    def test_generate_request_data_non_gateway_model(self):
        llm = OrcaRouterAPI()
        # A model without a gateway prefix is validated by the base wrapper.
        with self.assertRaises(NotImplementedError):
            llm.generate_request_data(
                'not-a-model',
                [{'role': 'user', 'content': 'hi'}],
                {'max_new_tokens': 512},
            )

    @mock.patch('lagent.llms.openai.requests.post')
    def test_chat(self, mock_post):
        os.environ[ORCAROUTER_API_KEY_ENV] = 'sk-orca-test'
        mock_post.return_value = FakeResponse(
            {'choices': [{'message': {'content': 'ORCA-OK'}}]}
        )
        llm = OrcaRouterAPI(key='ENV')
        ret = llm.chat([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(ret, 'ORCA-OK')
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs['url'] if 'url' in kwargs else mock_post.call_args[0][0],
                         ORCAROUTER_API_BASE)
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer sk-orca-test')


class TestAsyncOrcaRouterAPI(unittest.TestCase):

    def tearDown(self):
        os.environ.pop(ORCAROUTER_API_KEY_ENV, None)

    def test_defaults(self):
        llm = AsyncOrcaRouterAPI()
        self.assertEqual(llm.url, ORCAROUTER_API_BASE)
        self.assertEqual(llm.model_type, ORCAROUTER_DEFAULT_MODEL)

    def test_key_from_env(self):
        os.environ[ORCAROUTER_API_KEY_ENV] = 'sk-orca-test'
        llm = AsyncOrcaRouterAPI(key='ENV')
        self.assertEqual(llm.keys, ['sk-orca-test'])

    def test_generate_request_data_gateway_model(self):
        llm = AsyncOrcaRouterAPI()
        _, data = llm.generate_request_data(
            'orcarouter/auto',
            [{'role': 'user', 'content': 'hi'}],
            {'max_new_tokens': 512},
        )
        self.assertEqual(data['model'], 'orcarouter/auto')
