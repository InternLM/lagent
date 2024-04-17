import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from io import StringIO
from multiprocessing import Process, Queue
from typing import List, Optional, Type, Union

from filelock import FileLock
from IPython import InteractiveShell
from timeout_decorator import timeout as tm

from ..schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction
from .parser import BaseParser, JsonParser


class IPythonProcess(Process):

    def __init__(self,
                 in_q: Queue,
                 out_q: Queue,
                 timeout: int = 20,
                 ci_lock: str = None,
                 daemon: bool = True):
        super().__init__(daemon=daemon)
        self.in_q = in_q
        self.out_q = out_q
        self.timeout = timeout
        self.session_id2shell = defaultdict(InteractiveShell)
        self.ci_lock = FileLock(
            ci_lock) if ci_lock else nullcontext()  # avoid core corruption
        self._highlighting = re.compile(r'\x1b\[\d{,3}(;\d{,3}){,3}m')

    def run(self):
        while True:
            msg = self.in_q.get()
            if msg == 'reset':
                with self.ci_lock:
                    for session_id, shell in self.session_id2shell.items():
                        try:
                            shell.reset(new_session=False)
                        except Exception:
                            self.session_id2shell[
                                session_id] = InteractiveShell()
                self.out_q.put('ok')
            elif isinstance(msg, tuple) and len(msg) == 3:
                i, session_id, code = msg
                res = tm(self.timeout)(self.exec)(session_id, code)
                self.out_q.put((i, session_id, res))

    def exec(self, session_id, code):
        try:
            with StringIO() as io:
                old_stdout = sys.stdout
                sys.stdout = io
                self.session_id2shell[session_id].run_cell(
                    self.extract_code(code))
                sys.stdout = old_stdout
                output = self._highlighting.sub('', io.getvalue().strip())
                output = re.sub(r'^Out\[\d+\]: ', '', output)
            if 'Error' in output or 'Traceback' in output:
                output = output.lstrip('-').strip()
                if output.startswith('TimeoutError'):
                    output = 'The code interpreter encountered a timeout error.'
                return {'status': 'FAILURE', 'msg': output}
            return {'status': 'SUCCESS', 'value': output}
        except Exception as e:
            return {'status': 'FAILURE', 'msg': str(e)}

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


class IPythonInteractiveManager(BaseAction):

    def __init__(
        self,
        max_workers: int = 50,
        timeout: int = 30,
        ci_lock: str = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ):
        super().__init__(description, parser, enable)
        self.max_workers = max_workers
        self.timeout = timeout
        self.ci_lock = ci_lock
        self.id2queue = defaultdict(Queue)
        self.id2process = {}
        self.out_queue = Queue()

    def __call__(self,
                 commands: Union[str, List[str]],
                 session_ids: Union[int, List[int]] = None):
        if isinstance(commands, list):
            batch_size = len(commands)
            is_batch = True
        else:
            batch_size = 1
            commands = [commands]
            is_batch = False
        if session_ids is None:
            session_ids = range(batch_size)
        elif isinstance(session_ids, int):
            session_ids = [session_ids]
        if len(session_ids) != batch_size or len(session_ids) != len(
                set(session_ids)):
            raise ValueError(
                'the size of `session_ids` must equal that of `commands`')
        try:
            exec_results = self.run_code_blocks([
                (session_id, command)
                for session_id, command in zip(session_ids, commands)
            ])
        except KeyboardInterrupt:
            self.clear()
            exit(1)
        action_returns = []
        for result, code in zip(exec_results, commands):
            action_return = ActionReturn({'command': code}, type=self.name)
            if result['status'] == 'SUCCESS':
                action_return.result = [
                    dict(type='text', content=result['value'])
                ]
                action_return.state = ActionStatusCode.SUCCESS
            else:
                action_return.errmsg = result['msg']
                action_return.state = ActionStatusCode.API_ERROR
            action_returns.append(action_return)
        if not is_batch:
            return action_returns[0]
        return action_returns

    def process_code(self, index, session_id, code):
        ipy_id = session_id % self.max_workers
        input_queue = self.id2queue[ipy_id]
        proc = self.id2process.setdefault(
            ipy_id,
            IPythonProcess(
                input_queue,
                self.out_queue,
                self.timeout,
                self.ci_lock,
                daemon=True))
        if not proc.is_alive():
            proc.start()
        input_queue.put((index, session_id, code))

    def run_code_blocks(self, session_code_pairs):
        size = len(session_code_pairs)
        for index, (session_id, code) in enumerate(session_code_pairs):
            self.process_code(index, session_id, code)
        results = []
        while len(results) < size:
            msg = self.out_queue.get()
            if isinstance(msg, tuple) and len(msg) == 3:
                index, _, result = msg
                results.append((index, result))
        results.sort()
        return [item[1] for item in results]

    def clear(self):
        self.id2queue.clear()
        for proc in self.id2process.values():
            proc.terminate()
        self.id2process.clear()
        while not self.out_queue.empty():
            self.out_queue.get()

    def reset(self):
        cnt = 0
        for q in self.id2queue.values():
            q.put('reset')
            cnt += 1
        while cnt > 0:
            msg = self.out_queue.get()
            if msg == 'ok':
                cnt -= 1
