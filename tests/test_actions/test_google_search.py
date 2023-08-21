import json
from unittest import TestCase, mock

from lagent.actions import GoogleSearch
from lagent.schema import ActionStatusCode


class TestGoogleSearch(TestCase):

    @mock.patch.object(GoogleSearch, '_search')
    def test_search_tool(self, mock_search_func):
        mock_response = (200, json.load('tests/data/search.json'))
        mock_search_func.return_value = mock_response
        search_tool = GoogleSearch(api_key='abc')
        tool_return = search_tool.run("What's the capital of China?")
        self.assertEqual(tool_return.state, ActionStatusCode.SUCCESS)
        self.assertDictEqual(tool_return.result, dict(text="['Beijing']"))

    @mock.patch.object(GoogleSearch, '_search')
    def test_api_error(self, mock_search_func):
        mock_response = (403, {'message': 'bad requests'})
        mock_search_func.return_value = mock_response
        search_tool = GoogleSearch(api_key='abc')
        tool_return = search_tool.run("What's the capital of China?")
        self.assertEqual(tool_return.state, ActionStatusCode.API_ERROR)
        self.assertEqual(tool_return.errmsg, str(403))

    @mock.patch.object(GoogleSearch, '_search')
    def test_http_error(self, mock_search_func):
        mock_response = (-1, 'HTTPSConnectionPool')
        mock_search_func.return_value = mock_response
        search_tool = GoogleSearch(api_key='abc')
        tool_return = search_tool.run("What's the capital of China?")
        self.assertEqual(tool_return.state, ActionStatusCode.HTTP_ERROR)
        self.assertEqual(tool_return.errmsg, 'HTTPSConnectionPool')
