# flake8: noqa: E501
import asyncio
import base64
import io
import json
import logging
import os
import queue
import re
import signal
import sys
import tempfile
import traceback
import uuid
from typing import Optional, Tuple, Type

from jupyter_client import AsyncKernelClient, AsyncKernelManager, AsyncMultiKernelManager
from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

logger = logging.getLogger(__name__)

START_CODE = """
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

get_ipython().system = lambda *args: print('Assume we have this package, ! is disabled!')
{}
"""  # noqa


class TimeoutError(Exception):
    pass


class KernelDeath(Exception):
    pass


async def async_run_code(
    km: AsyncKernelManager,
    code,
    *,
    interrupt_after=30,
    iopub_timeout=40,
    wait_for_ready_timeout=60,
    shutdown_kernel=True,
):
    assert iopub_timeout > interrupt_after
    try:

        async def get_iopub_msg_with_death_detection(kc: AsyncKernelClient,
                                                     *,
                                                     timeout=None):
            loop = asyncio.get_running_loop()
            dead_fut = loop.create_future()

            def restarting():
                assert (
                    False
                ), "Restart shouldn't happen because config.KernelRestarter.restart_limit is expected to be set to 0"

            def dead():
                logger.info("Kernel has died, will NOT restart")
                dead_fut.set_result(None)

            msg_task = asyncio.create_task(kc.get_iopub_msg(timeout=timeout))
            km.add_restart_callback(restarting, "restart")
            km.add_restart_callback(dead, "dead")
            try:
                done, _ = await asyncio.wait(
                    [dead_fut, msg_task], return_when=asyncio.FIRST_COMPLETED)
                if dead_fut in done:
                    raise KernelDeath()
                assert msg_task in done
                return await msg_task
            finally:
                msg_task.cancel()
                km.remove_restart_callback(restarting, "restart")
                km.remove_restart_callback(dead, "dead")

        async def send_interrupt():
            await asyncio.sleep(interrupt_after)
            logger.info("Sending interrupt to kernel")
            await km.interrupt_kernel()

        @retry(
            retry=retry_if_result(lambda ret: ret[-1].strip() in [
                'KeyboardInterrupt',
                f"Kernel didn't respond in {wait_for_ready_timeout} seconds",
            ] if isinstance(ret, tuple) else False),
            stop=stop_after_attempt(3),
            wait=wait_fixed(1),
            retry_error_callback=lambda state: state.outcome.result())
        async def run():
            execute_result = None
            error_traceback = None
            stream_text_list = []
            kc = km.client()
            assert isinstance(kc, AsyncKernelClient)
            kc.start_channels()
            try:
                await kc.wait_for_ready(timeout=wait_for_ready_timeout)
                msg_id = kc.execute(code)
                while True:
                    message = await get_iopub_msg_with_death_detection(
                        kc, timeout=iopub_timeout)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            json.dumps(message, indent=2, default=str))
                    assert message["parent_header"]["msg_id"] == msg_id
                    msg_type = message["msg_type"]
                    if msg_type == "status":
                        if message["content"]["execution_state"] == "idle":
                            break
                    elif msg_type == "stream":
                        stream_name = message["content"]["name"]
                        stream_text = message["content"]["text"]
                        stream_text_list.append(stream_text)
                    elif msg_type == "execute_result":
                        execute_result = message["content"]["data"]
                    elif msg_type == "error":
                        error_traceback_lines = message["content"]["traceback"]
                        error_traceback = "\n".join(error_traceback_lines)
                    elif msg_type == "execute_input":
                        pass
                    else:
                        assert False, f"Unknown message_type: {msg_type}"
            finally:
                kc.stop_channels()
            return execute_result, error_traceback, "".join(stream_text_list)

        if interrupt_after:
            run_task = asyncio.create_task(run())
            send_interrupt_task = asyncio.create_task(send_interrupt())
            done, _ = await asyncio.wait([run_task, send_interrupt_task],
                                         return_when=asyncio.FIRST_COMPLETED)
            if run_task in done:
                send_interrupt_task.cancel()
            else:
                assert send_interrupt_task in done
            result = await run_task
        else:
            result = await run()
        return result
    finally:
        if shutdown_kernel:
            await km.shutdown_kernel()


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
    """

    _KERNEL_CLIENTS = {}

    def __init__(
        self,
        timeout: int = 20,
        user_data_dir: str = 'ENV',
        work_dir='./work_dir/tmp_dir',
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)

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


class AsyncIPythonInterpreter(AsyncActionMixin, IPythonInterpreter):
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
    """

    _UNBOUND_KERNEL_CLIENTS = asyncio.Queue()

    def __init__(
        self,
        timeout: int = 20,
        user_data_dir: str = 'ENV',
        work_dir=os.path.join(tempfile.gettempdir(), 'tmp_dir'),
        max_kernels: Optional[int] = None,
        reuse_kernel: bool = True,
        startup_rate: bool = 32,
        connection_dir: str = tempfile.gettempdir(),
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(timeout, user_data_dir, work_dir, description, parser)
        from traitlets.config import Config

        c = Config()
        c.KernelManager.transport = 'ipc'
        self._amkm = AsyncMultiKernelManager(
            config=c, connection_dir=connection_dir)
        self._max_kernels = max_kernels
        self._reuse_kernel = reuse_kernel
        self._sem = asyncio.Semaphore(startup_rate)
        self._lock = asyncio.Lock()

    async def initialize(self, session_id: str):
        session_id = str(session_id)
        while True:
            if session_id in self._KERNEL_CLIENTS:
                return self._KERNEL_CLIENTS[session_id]
            if self._reuse_kernel and not self._UNBOUND_KERNEL_CLIENTS.empty():
                self._KERNEL_CLIENTS[
                    session_id] = await self._UNBOUND_KERNEL_CLIENTS.get()
                return self._KERNEL_CLIENTS[session_id]
            async with self._sem:
                if self._max_kernels is None or len(
                        self._KERNEL_CLIENTS
                ) + self._UNBOUND_KERNEL_CLIENTS.qsize() < self._max_kernels:
                    kernel_id = None
                    try:
                        kernel_id = await self._amkm.start_kernel()
                        kernel = self._amkm.get_kernel(kernel_id)
                        client = kernel.client()
                        _, error_stacktrace, stream_text = await async_run_code(
                            kernel,
                            START_CODE.format(self.user_data_dir),
                            shutdown_kernel=False)
                        # check if the output of START_CODE meets expectations
                        if not (error_stacktrace is None
                                and stream_text == ''):
                            raise RuntimeError
                    except Exception as e:
                        print(f'Starting kernel error: {e}')
                        if kernel_id:
                            await self._amkm.shutdown_kernel(kernel_id)
                            self._amkm.remove_kernel(kernel_id)
                        await asyncio.sleep(1)
                        continue
                    if self._max_kernels is None:
                        self._KERNEL_CLIENTS[session_id] = (kernel_id, kernel,
                                                            client)
                        return kernel_id, kernel, client
                    async with self._lock:
                        if len(self._KERNEL_CLIENTS
                               ) + self._UNBOUND_KERNEL_CLIENTS.qsize(
                               ) < self._max_kernels:
                            self._KERNEL_CLIENTS[session_id] = (kernel_id,
                                                                kernel, client)
                            return kernel_id, kernel, client
                    await self._amkm.shutdown_kernel(kernel_id)
                    self._amkm.remove_kernel(kernel_id)
            await asyncio.sleep(1)

    async def reset(self, session_id: str):
        session_id = str(session_id)
        if session_id not in self._KERNEL_CLIENTS:
            return
        _, kernel, _ = self._KERNEL_CLIENTS[session_id]
        code = "get_ipython().run_line_magic('reset', '-f')\n" + \
            START_CODE.format(self.user_data_dir)
        await async_run_code(kernel, code, shutdown_kernel=False)

    async def shutdown(self, session_id: str):
        session_id = str(session_id)
        if session_id in self._KERNEL_CLIENTS:
            kernel_id, _, _ = self._KERNEL_CLIENTS.get(session_id)
            await self._amkm.shutdown_kernel(kernel_id)
            self._amkm.remove_kernel(kernel_id)
            del self._KERNEL_CLIENTS[session_id]

    async def close_session(self, session_id: str):
        session_id = str(session_id)
        if self._reuse_kernel:
            if session_id in self._KERNEL_CLIENTS:
                await self.reset(session_id)
                await self._UNBOUND_KERNEL_CLIENTS.put(
                    self._KERNEL_CLIENTS.pop(session_id))
        else:
            await self.shutdown(session_id)

    async def _call(self, command, timeout=None, session_id=None):
        _, kernel, _ = await self.initialize(str(session_id))
        result = await async_run_code(
            kernel,
            extract_code(command),
            interrupt_after=timeout or self.timeout,
            shutdown_kernel=False)
        execute_result, error_stacktrace, stream_text = result
        if error_stacktrace is not None:
            ret = re.sub('^-*\n', '', escape_ansi(error_stacktrace))
            if ret.endswith('KeyboardInterrupt: '):
                ret = 'The code interpreter encountered a timeout error.'
            status, ret = False, ret.strip()
        elif execute_result is not None:
            status, ret = True, dict(text=execute_result.get('text/plain', ''))
        else:
            status, ret = True, dict(text=stream_text.strip())
        return status, ret

    @tool_api
    async def run(self,
                  command: str,
                  timeout: Optional[int] = None,
                  session_id: Optional[str] = None) -> ActionReturn:
        r"""When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.

        Args:
            command (:class:`str`): Python code
            timeout (:class:`Optional[int]`): Upper bound of waiting time for Python script execution.
        """
        tool_return = ActionReturn(url=None, args=None, type=self.name)
        tool_return.args = dict(text=command)
        succeed, result = await self._call(command, timeout, session_id)
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


if __name__ == '__main__':
    code_interpreter = IPythonInterpreter()
    while True:
        print(code_interpreter(get_multiline_input('Enter python code:')))
