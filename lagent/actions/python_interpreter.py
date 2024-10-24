# flake8: noqa: E501
import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type

from aioify import aioify

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


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


class PythonInterpreter(BaseAction):
    """A Python executor that can execute Python scripts.

    Args:
        answer_symbol (str, Optional): the answer symbol from LLM. Defaults to ``None``.
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        answer_from_stdout (boolean, Optional): whether the execution results is from
            stdout. Defaults to ``False``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
    """

    def __init__(
        self,
        answer_symbol: Optional[str] = None,
        answer_expr: Optional[str] = 'solution()',
        answer_from_stdout: bool = False,
        timeout: int = 20,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout

    @tool_api
    def run(self, command: str) -> ActionReturn:
        """用来执行Python代码。代码必须是一个函数，函数名必须得是 'solution'，代码对应你的思考过程。代码实例格式如下：

        ```python
        # import 依赖包
        import xxx
        def solution():
            # 初始化一些变量
            variable_names_with_real_meaning = xxx
            # 步骤一
            mid_variable = func(variable_names_with_real_meaning)
            # 步骤 x
            mid_variable = func(mid_variable)
            # 最后结果
            final_answer =  func(mid_variable)
            return final_answer
        ```

        Args:
            command (:class:`str`): Python code snippet
        """
        from func_timeout import FunctionTimedOut, func_set_timeout
        self.runtime = GenericRuntime()
        try:
            tool_return = func_set_timeout(self.timeout)(self._call)(command)
        except FunctionTimedOut as e:
            tool_return = ActionReturn(type=self.name)
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _call(self, command: str) -> ActionReturn:
        tool_return = ActionReturn(type=self.name)
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
            tool_return.result = [dict(type='text', content=str(res))]
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return


class AsyncPythonInterpreter(AsyncActionMixin, PythonInterpreter):
    """A Python executor that can execute Python scripts.

    Args:
        answer_symbol (str, Optional): the answer symbol from LLM. Defaults to ``None``.
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        answer_from_stdout (boolean, Optional): whether the execution results is from
            stdout. Defaults to ``False``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
    """

    @tool_api
    @aioify
    def run(self, command: str) -> ActionReturn:
        """用来执行Python代码。代码必须是一个函数，函数名必须得是 'solution'，代码对应你的思考过程。代码实例格式如下：

        ```python
        # import 依赖包
        import xxx
        def solution():
            # 初始化一些变量
            variable_names_with_real_meaning = xxx
            # 步骤一
            mid_variable = func(variable_names_with_real_meaning)
            # 步骤 x
            mid_variable = func(mid_variable)
            # 最后结果
            final_answer =  func(mid_variable)
            return final_answer
        ```

        Args:
            command (:class:`str`): Python code snippet
        """
        return super().run(command)
