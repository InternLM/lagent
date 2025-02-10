import os
from unittest import TestCase, mock

from lagent.actions.web_browser import SearxngSearch


class TestGoogleSearch(TestCase):

    @mock.patch.object(SearxngSearch, 'search')
    def test_search_tool(self, mock_search_func):
        # mock_response = (200, json.load('tests/data/search.json'))
        # mock_search_func.return_value = mock_response

        os.environ['SEARXNG_URL'] = 'http://192.168.26.xx:18080/search'
        search_tool = SearxngSearch(api_key='abc')
        tool_return = search_tool.search("What's the capital of China?")
        self.assertGreater(len(tool_return), 0)
