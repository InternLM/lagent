from unittest import TestCase

from lagent.actions.python_interpreter import PythonInterpreter
from lagent.schema import ActionStatusCode


class TestPythonInterpreter(TestCase):

    def test_python_executor(self):
        python_executor = PythonInterpreter()
        tool_return = python_executor(
            '```python\ndef solution():\n    return 1\n```')
        self.assertEqual(tool_return.state, ActionStatusCode.SUCCESS)
        self.assertDictEqual(tool_return.result, dict(text='1'))

    def test_timeout(self):
        python_executor = PythonInterpreter(timeout=2)
        tool_return = python_executor(
            '```python\ndef solution():\n    while True:\n        pass\n```')
        self.assertEqual(tool_return.state, ActionStatusCode.API_ERROR)
        self.assertIn('FunctionTimedOut', tool_return.errmsg)
