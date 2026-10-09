import asyncio
import json
from unittest import TestCase

from lagent.actions import AsyncActionMixin, BaseAction, tool_api
from lagent.schema import ActionStatusCode


class EchoAction(BaseAction):
    """Return a value without changing its text."""

    @tool_api
    def run(self, value):
        """Return the supplied value.

        Args:
            value (Any): The value to return.
        """
        return value


class AsyncEchoAction(AsyncActionMixin, BaseAction):
    """Return a value asynchronously without changing its text."""

    @tool_api
    async def run(self, value):
        """Return the supplied value.

        Args:
            value (Any): The value to return.
        """
        return value


class TestParserUnicodeOutputs(TestCase):

    def test_string_outputs(self):
        action = EchoAction()
        for text in ['plain ASCII', '中文 café', '', 'done ✅', 'नमस्ते', '𠮷田']:
            with self.subTest(text=text):
                result = action({'value': text})
                self.assertEqual(result.state, ActionStatusCode.SUCCESS)
                self.assertEqual(result.format_result(), text)

    def test_dictionary_outputs(self):
        value = {'status': '✅', 'text': 'नमस्ते', '𠮷': '中文', 'number': 0}
        result = EchoAction()({'value': value})
        self.assertEqual(json.loads(result.format_result()), value)

    def test_non_string_outputs(self):
        action = EchoAction()
        for value in [0, False, None, ['✅', 'नमस्ते']]:
            with self.subTest(value=value):
                self.assertEqual(action({'value': value}).format_result(), str(value))

    def test_async_action_preserves_tool_output(self):
        result = asyncio.run(AsyncEchoAction()({'value': {'text': '✅ नमस्ते'}}))
        self.assertEqual(result.state, ActionStatusCode.SUCCESS)
        self.assertEqual(json.loads(result.format_result()), {'text': '✅ नमस्ते'})

    def test_invalid_arguments_still_return_args_error(self):
        result = EchoAction()({'missing': '✅'})
        self.assertEqual(result.state, ActionStatusCode.ARGS_ERROR)
