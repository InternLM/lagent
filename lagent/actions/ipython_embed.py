import asyncio
import re
import sys
from collections import defaultdict
from io import StringIO
from typing import List, Optional, Type, Union

from IPython.terminal.embed import InteractiveShellEmbed

from ..schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction
from .parser import BaseParser, JsonParser


class IPythonInteractiveEmbed(BaseAction):

    def __init__(
        self,
        timeout: int = 30,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ):
        super().__init__(description, parser, enable)
        self.timeout = timeout
        self.sessions = defaultdict(InteractiveShellEmbed)
        self._highlighting = re.compile(r'\x1b\[\d{,3}(;\d{,3}){,3}m')

    def __call__(self,
                 commands: Union[str, List[str]],
                 indexes: Union[int, List[int]] = None) -> ActionReturn:
        if isinstance(commands, list):
            batch_size = len(commands)
            is_batch = True
        else:
            batch_size = 1
            commands = [commands]
            is_batch = False
        if indexes is None:
            indexes = range(batch_size)
        elif isinstance(indexes, int):
            indexes = [indexes]
        if len(indexes) != batch_size or len(indexes) != len(set(indexes)):
            raise ValueError(
                'the size of `indexes` must equal that of `commands`')
        results = self.run_code_blocks([
            (index, command) for index, command in zip(indexes, commands)
        ])
        if not is_batch:
            return results[0]
        return results

    async def exec_code(self, index: str, code: str):
        """在对应的IPython实例中执行代码，并维护状态."""
        shell = self.sessions[index]  # 为每个索引获取或创建一个新的shell实例
        try:
            with StringIO() as io:
                old_stdout = sys.stdout
                sys.stdout = io
                await shell.run_cell_async(code)
                sys.stdout = old_stdout
                output = self._highlighting.sub('', io.getvalue().strip())
                output = re.sub(r'^Out\[\d+\]: ', '', output)
            if 'Error' in output or 'Traceback' in output:
                output = output.lstrip('-').strip()
                return {'status': 'FAILURE', 'msg': output}
            return {'status': 'SUCCESS', 'value': output}
        except Exception as e:
            return {'status': 'FAILURE', 'msg': str(e)}

    async def exec_code_with_timeout(self, index, code, timeout=None):
        try:
            ret = await asyncio.wait_for(
                self.exec_code(index, code), timeout or self.timeout)
        except asyncio.TimeoutError:
            ret = {
                'status': 'FAILURE',
                'msg': 'The code interpreter encountered a timeout error.'
            }
        return ret

    async def process_code(self, index_code_pairs, timeout=None):
        """处理一系列索引和代码对."""
        tasks = []
        for index, code in index_code_pairs:
            code = self.extract_code(code)
            task = self.exec_code_with_timeout(index, code, timeout)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    def run_code_blocks(self, index_code_pairs):
        """启动代码块的异步执行."""
        results = asyncio.run(self.process_code(index_code_pairs))
        out = []
        for (_, code), result in zip(index_code_pairs, results):
            if result['status'] == 'SUCCESS':
                out.append(
                    ActionReturn(
                        args={'command': code},
                        type=self.name,
                        result=[{
                            'type': 'text',
                            'content': result['value']
                        }],
                        state=ActionStatusCode.SUCCESS))
            else:
                out.append(
                    ActionReturn(
                        args={'command': code},
                        type=self.name,
                        errmsg=result['msg'],
                        state=ActionStatusCode.API_ERROR))
        return out

    @staticmethod
    def extract_code(text: str) -> str:
        """Extract Python code from markup languages.

        Args:
            text (:class:`str`): Markdown-formatted text

        Returns:
            :class:`str`: Python code
        """
        import json5

        # Match triple backtick blocks first
        triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
        # Match single backtick blocks second
        single_match = re.search(r'`([^`]*)`', text, re.DOTALL)
        if triple_match:
            text = triple_match.group(1)
        elif single_match:
            text = single_match.group(1)
        else:
            try:
                text = json5.loads(text)['code']
            except Exception:
                pass
        # If no code blocks found, return original text
        return text

    def reset(self):
        for interpreter in self.sessions.values():
            interpreter.reset()
