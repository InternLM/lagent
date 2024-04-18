# flake8: noqa: E501
import base64
import io
import logging
import os
import queue
import re
import signal
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

START_CODE = """
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

get_ipython().system = lambda *args: print('Assume we have this package, ! is disabled!')
{}
"""  # noqa


class TimeoutError(Exception):
    pass


class IPythonInterpreter(BaseAction):
    """A IPython executor that can execute Python scripts in a jupyter manner.

    Args:
        timeout (int): Upper bound of waiting time for Python script execution.
            Defaults to 20.
        user_data_dir (str, optional): Specified the user data directory for files
            loading. If set to `ENV`, use `USER_DATA_DIR` environment variable.
            Defaults to `ENV`.
        work_dir (str, optional): Specify which directory to save output images to.
            Defaults to ``'./work_dir/tmp_dir'``.
        description (dict): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to ``True``.
    """

    _KERNEL_CLIENTS = {}

    def __init__(self,
                 timeout: int = 20,
                 user_data_dir: str = 'ENV',
                 work_dir='./work_dir/tmp_dir',
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        super().__init__(description, parser, enable)

        self.timeout = timeout
        if user_data_dir == 'ENV':
            user_data_dir = os.environ.get('USER_DATA_DIR', '')

        if user_data_dir:
            user_data_dir = os.path.dirname(user_data_dir)
            user_data_dir = f"import os\nos.chdir('{user_data_dir}')"
        self.user_data_dir = user_data_dir
        self._initialized = False
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

    @staticmethod
    def start_kernel():
        from jupyter_client import KernelManager

        # start the kernel and manager
        km = KernelManager()
        km.start_kernel()
        kc = km.client()
        return km, kc

    def initialize(self):
        if self._initialized:
            return
        pid = os.getpid()
        if pid not in self._KERNEL_CLIENTS:
            self._KERNEL_CLIENTS[pid] = self.start_kernel()
        self.kernel_manager, self.kernel_client = self._KERNEL_CLIENTS[pid]
        self._initialized = True
        self._call(START_CODE.format(self.user_data_dir), None)

    def reset(self):
        if not self._initialized:
            self.initialize()
        else:
            code = "get_ipython().run_line_magic('reset', '-f')\n" + \
                START_CODE.format(self.user_data_dir)
            self._call(code, None)

    def _call(self,
              command: str,
              timeout: Optional[int] = None) -> Tuple[str, bool]:
        self.initialize()
        command = extract_code(command)

        # check previous remaining result
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=5)
                msg_type = msg['msg_type']
                if msg_type == 'status':
                    if msg['content'].get('execution_state') == 'idle':
                        break
            except queue.Empty:
                # assume no result
                break

        self.kernel_client.execute(command)

        def _inner_call():
            result = ''
            images = []
            succeed = True
            image_idx = 0

            while True:
                text = ''
                image = ''
                finished = False
                msg_type = 'error'
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=20)
                    msg_type = msg['msg_type']
                    if msg_type == 'status':
                        if msg['content'].get('execution_state') == 'idle':
                            finished = True
                    elif msg_type == 'execute_result':
                        text = msg['content']['data'].get('text/plain', '')
                        if 'image/png' in msg['content']['data']:
                            image_b64 = msg['content']['data']['image/png']
                            image_url = publish_image_to_local(
                                image_b64, self.work_dir)
                            image_idx += 1
                            image = '![fig-%03d](%s)' % (image_idx, image_url)

                    elif msg_type == 'display_data':
                        if 'image/png' in msg['content']['data']:
                            image_b64 = msg['content']['data']['image/png']
                            image_url = publish_image_to_local(
                                image_b64, self.work_dir)
                            image_idx += 1
                            image = '![fig-%03d](%s)' % (image_idx, image_url)

                        else:
                            text = msg['content']['data'].get('text/plain', '')
                    elif msg_type == 'stream':
                        msg_type = msg['content']['name']  # stdout, stderr
                        text = msg['content']['text']
                    elif msg_type == 'error':
                        succeed = False
                        text = escape_ansi('\n'.join(
                            msg['content']['traceback']))
                        if 'M6_CODE_INTERPRETER_TIMEOUT' in text:
                            text = f'Timeout. No response after {timeout} seconds.'  # noqa
                except queue.Empty:
                    # stop current task in case break next input.
                    self.kernel_manager.interrupt_kernel()
                    succeed = False
                    text = f'Timeout. No response after {timeout} seconds.'
                    finished = True
                except Exception:
                    succeed = False
                    msg = ''.join(traceback.format_exception(*sys.exc_info()))
                    # text = 'The code interpreter encountered an unexpected error.'  # noqa
                    text = msg
                    logging.warning(msg)
                    finished = True
                if text:
                    # result += f'\n\n{msg_type}:\n\n```\n{text}\n```'
                    result += f'{text}'

                if image:
                    images.append(image_url)
                if finished:
                    return succeed, dict(text=result, image=images)

        try:
            if timeout:

                def handler(signum, frame):
                    raise TimeoutError()

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)
            succeed, result = _inner_call()
        except TimeoutError:
            succeed = False
            text = 'The code interpreter encountered an unexpected error.'
            result = f'\n\nerror:\n\n```\n{text}\n```'
        finally:
            if timeout:
                signal.alarm(0)

        # result = result.strip('\n')
        return succeed, result

    @tool_api
    def run(self, command: str, timeout: Optional[int] = None) -> ActionReturn:
        r"""When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.

        Args:
            command (:class:`str`): Python code
            timeout (:class:`Optional[int]`): Upper bound of waiting time for Python script execution.
        """
        tool_return = ActionReturn(url=None, args=None, type=self.name)
        tool_return.args = dict(text=command)
        succeed, result = self._call(command, timeout)
        if succeed:
            text = result['text']
            image = result.get('image', [])
            resp = [dict(type='text', content=text)]
            if image:
                resp.extend([dict(type='image', content=im) for im in image])
            tool_return.result = resp
            # tool_return.result = dict(
            #     text=result['text'], image=result.get('image', [])[0])
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = result.get('text', '') if isinstance(
                result, dict) else result
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return


def extract_code(text):
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


def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


def publish_image_to_local(image_base64: str, work_dir='./work_dir/tmp_dir'):
    import PIL.Image
    image_file = str(uuid.uuid4()) + '.png'
    local_image_file = os.path.join(work_dir, image_file)

    png_bytes = base64.b64decode(image_base64)
    assert isinstance(png_bytes, bytes)
    bytes_io = io.BytesIO(png_bytes)
    PIL.Image.open(bytes_io).save(local_image_file, 'png')

    return local_image_file


# local test for code interpreter
def get_multiline_input(hint):
    print(hint)
    print('// Press ENTER to make a new line. Press CTRL-D to end input.')
    lines = []
    while True:
        try:
            line = input()
        except EOFError:  # CTRL-D
            break
        lines.append(line)
    print('// Input received.')
    if lines:
        return '\n'.join(lines)
    else:
        return ''


class BatchIPythonInterpreter(BaseAction):
    """A IPython executor that can execute Python scripts in batches in a jupyter manner."""

    def __init__(
        self,
        python_interpreter: Dict[str, Any],
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ):
        self.python_interpreter_init_args = python_interpreter
        self.index2python_interpreter = {}
        super().__init__(description, parser, enable)

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
        tasks = []
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for idx, command in zip(indexes, commands):
                interpreter = self.index2python_interpreter.setdefault(
                    idx,
                    IPythonInterpreter(**self.python_interpreter_init_args))
                tasks.append(pool.submit(interpreter.run, command))
        wait(tasks)
        results = [task.result() for task in tasks]
        if not is_batch:
            return results[0]
        return results

    def reset(self):
        self.index2python_interpreter.clear()


if __name__ == '__main__':
    code_interpreter = IPythonInterpreter()
    while True:
        print(code_interpreter(get_multiline_input('Enter python code:')))
