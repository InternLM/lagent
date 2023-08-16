import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional

from func_timeout import FunctionTimedOut, func_set_timeout

from etangent.actions.base_action import BaseAction
from etangent.schema import ActionReturn, ActionStatusCode


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


class PythonExecutor(BaseAction):

    def __init__(self,
                 description='',
                 answer_symbol: Optional[str] = None,
                 answer_expr: Optional[str] = 'solution()',
                 answer_from_stdout: bool = False,
                 name=None,
                 enable=True,
                 disable_description=None,
                 timeout=30):
        super().__init__(description, name, enable, disable_description)

        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout

    def __call__(self, command):
        self.runtime = GenericRuntime()
        tool_return = ActionReturn(url=None, args=None)
        try:
            tool_return = func_set_timeout(self.timeout)(self._call)(command)
        except FunctionTimedOut as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _call(self, command):
        tool_return = ActionReturn(url=None, args=None)
        try:
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]
            tool_return.args = dict(text='```python\n' + command + '\n```')
            command = command.split('\n')

            if self.answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(command))
                program_io.seek(0)
                res = program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(command[:-1]))
                res = self.runtime.eval_code(command[-1])
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
            return tool_return
        try:
            tool_return.result = dict(text=str(res))
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
