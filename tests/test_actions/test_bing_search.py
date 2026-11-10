from unittest import TestCase, mock

from lagent.actions.web_browser import BingSearch


class TestBingSearch(TestCase):

    def test_default_endpoint(self):
        search_tool = BingSearch(api_key='abc')
        self.assertEqual(search_tool.endpoint, 'https://api.bing.microsoft.com/v7.0/search')

    @mock.patch('lagent.actions.web_browser.requests.get')
    def test_custom_endpoint(self, mock_get):
        mock_get.return_value.json.return_value = {}
        search_tool = BingSearch(api_key='abc', endpoint='https://search.example.com/v7.0/search')
        search_tool._call_bing_api("What's the capital of China?")
        self.assertEqual(mock_get.call_args.args[0], 'https://search.example.com/v7.0/search')
        self.assertEqual(mock_get.call_args.kwargs['headers'], {'Ocp-Apim-Subscription-Key': 'abc'})
