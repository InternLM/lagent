"""Tmux-backed tools exposed as standard function-call tools.

``TerminalExecute`` takes a batch of keystroke sequences and returns the
resulting pane output in one round.  Batching matters because
``AsyncEnvAgent`` executes tool calls via ``asyncio.gather``; a persistent
tmux pane cannot be driven concurrently, so one call per turn per pane.

``MarkTaskComplete`` emits a sentinel the orchestrator interprets as a
completion signal.  The orchestrator (``Terminus2Agent``) implements the
two-step confirmation handshake — the action itself is stateless.
"""

"""Local tmux session wrapper.

Port of harbor's ``TmuxSession`` that drops the ``BaseEnvironment.exec``
indirection and runs ``tmux`` as a local subprocess.  Intended for use
inside the lagent-daemon sandbox where tmux is already installed.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import time

from __future__ import annotations

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode


_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
_ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
_NEWLINE_CHARS = "\r\n"
_TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
# tmux silently drops commands exceeding its internal buffer (~16 KB since
# tmux 1.9).  Keep a conservative margin below that ceiling.
_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000

_OUTPUT_BYTE_LIMIT = 10_000
_DEFAULT_DURATION_CAP_SEC = 60.0
_TIMEOUT_TEMPLATE = (
    "Command '{command}' timed out after {timeout_sec}s.\n\n{terminal_state}"
)


class TmuxSession:
    """Persistent tmux pane driven by local subprocess calls."""

    def __init__(
        self,
        session_name: str,
        pane_width: int = 160,
        pane_height: int = 40,
        working_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        history_limit: int = 10_000_000,
    ):
        """Create a (not-yet-started) tmux session descriptor.

        Args:
            session_name (str): tmux session name (also used as the pane target).
            pane_width (int): starting pane width, maps to ``tmux -x``.
            pane_height (int): starting pane height, maps to ``tmux -y``.
            working_dir (str | None): cwd for the tmux server process.
            extra_env (dict[str, str] | None): env vars injected into the pane
                via ``tmux new-session -e``.
            history_limit (int): tmux scrollback line count.
        """
        if pane_width <= 0 or pane_height <= 0:
            raise ValueError("pane_width and pane_height must be positive integers.")
        self._session_name = session_name
        self._pane_width = int(pane_width)
        self._pane_height = int(pane_height)
        self._working_dir = working_dir
        self._extra_env = dict(extra_env or {})
        self._history_limit = int(history_limit)
        self._previous_buffer: str | None = None
        self._start()

    def _start(self) -> None:
        env_opts = "".join(
            f"-e {shlex.quote(f'{k}={v}')} " for k, v in self._extra_env.items()
        )
        start_cmd = (
            f"export TERM=xterm-256color && export SHELL=/bin/bash && "
            f"tmux new-session {env_opts}-x {self._pane_width} "
            f"-y {self._pane_height} -d "
            f"-s {shlex.quote(self._session_name)} 'bash --login'"
        )
        result = subprocess.run(
            start_cmd, shell=True, cwd=self._working_dir,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start tmux session: {result.stderr!r}")
        subprocess.run(
            f"tmux set-option -g history-limit {self._history_limit}",
            shell=True, capture_output=True,
        )

    async def is_session_alive(self) -> bool:
        rc, _, _ = await self._exec(
            f"tmux has-session -t {shlex.quote(self._session_name)}",
        )
        return rc == 0

    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Send keystrokes to the pane.

        Args:
            keys (str | list[str]): strings like ``"ls\\n"`` or tmux key names
                like ``"C-c"``.  When a list is passed each element is one
                ``tmux send-keys`` argument.
            block (bool): if true, waits for the command to finish (via
                ``tmux wait``) up to ``max_timeout_sec``.
            min_timeout_sec (float): minimum wall-time to spend when non-blocking.
            max_timeout_sec (float): ceiling for blocking mode.
        """
        prepared, is_blocking = self._prepare_keys(keys, block)
        if is_blocking:
            await self._send_blocking(prepared, max_timeout_sec)
        else:
            await self._send_non_blocking(prepared, min_timeout_sec)

    async def capture_pane(self, capture_entire: bool = False) -> str:
        args = ["tmux", "capture-pane", "-p"]
        if capture_entire:
            args.extend(["-S", "-"])
        args.extend(["-t", self._session_name])
        rc, stdout, _ = await self._exec(" ".join(shlex.quote(a) for a in args))
        if rc != 0:
            return ""
        return stdout or ""

    async def get_incremental_output(self) -> str:
        """Return new pane output since the last call, or the visible screen."""
        current_buffer = await self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            visible = await self.capture_pane(capture_entire=False)
            return f"Current Terminal Screen:\n{visible}"

        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"
        visible = await self.capture_pane(capture_entire=False)
        return f"Current Terminal Screen:\n{visible}"

    async def _exec(self, command: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._working_dir,
        )
        stdout_b, stderr_b = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )

    def _prepare_keys(
        self, keys: str | list[str], block: bool
    ) -> tuple[list[str], bool]:
        if isinstance(keys, str):
            keys = [keys]
        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False
        keys = self._prevent_execution(keys)
        keys.extend([_TMUX_COMPLETION_COMMAND, "Enter"])
        return keys, True

    @staticmethod
    def _is_enter_key(key: str) -> bool:
        return key in _ENTER_KEYS

    @staticmethod
    def _ends_with_newline(key: str) -> bool:
        return re.search(_ENDS_WITH_NEWLINE_PATTERN, key) is not None

    @classmethod
    def _is_executing_command(cls, key: str) -> bool:
        return cls._is_enter_key(key) or cls._ends_with_newline(key)

    @classmethod
    def _prevent_execution(cls, keys: list[str]) -> list[str]:
        keys = keys.copy()
        while keys and cls._is_executing_command(keys[-1]):
            if cls._is_enter_key(keys[-1]):
                keys.pop()
            else:
                stripped = keys[-1].rstrip(_NEWLINE_CHARS)
                if stripped:
                    keys[-1] = stripped
                else:
                    keys.pop()
        return keys

    def _tmux_send_keys(self, keys: list[str]) -> list[str]:
        """Build ``tmux send-keys`` commands, splitting to respect tmux's 16 KB limit."""
        prefix = "tmux send-keys -t " + shlex.quote(self._session_name)
        max_len = _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH

        escaped = [shlex.quote(k) for k in keys]
        single = prefix + " " + " ".join(escaped)
        if len(single) <= max_len:
            return [single]

        commands: list[str] = []
        current: list[str] = []
        current_len = len(prefix)

        def _flush() -> None:
            nonlocal current_len
            if current:
                commands.append(prefix + " " + " ".join(current))
                current.clear()
                current_len = len(prefix)

        for key in keys:
            q = shlex.quote(key)
            addition = 1 + len(q)
            if current_len + addition <= max_len:
                current.append(q)
                current_len += addition
            elif len(prefix) + addition <= max_len:
                _flush()
                current.append(q)
                current_len = len(prefix) + addition
            else:
                _flush()
                max_escaped = max_len - len(prefix) - 1
                for chunk in self._split_key_for_tmux(key, max_escaped):
                    if current_len + 1 + len(chunk) <= max_len:
                        current.append(chunk)
                        current_len += 1 + len(chunk)
                    else:
                        _flush()
                        current.append(chunk)
                        current_len = len(prefix) + 1 + len(chunk)
        _flush()
        return commands

    @staticmethod
    def _split_key_for_tmux(key: str, max_escaped_len: int) -> list[str]:
        chunks: list[str] = []
        remaining = key
        while remaining:
            lo, hi, best = 1, len(remaining), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if len(shlex.quote(remaining[:mid])) <= max_escaped_len:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            chunks.append(shlex.quote(remaining[:best]))
            remaining = remaining[best:]
        return chunks

    async def _send_blocking(self, keys: list[str], max_timeout_sec: float) -> None:
        start = time.time()
        for cmd in self._tmux_send_keys(keys):
            rc, _, stderr = await self._exec(cmd)
            if rc != 0:
                raise RuntimeError(f"tmux send-keys failed: {stderr!r}")
        rc, _, _ = await self._exec(f"timeout {max_timeout_sec}s tmux wait done")
        if rc != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")
        _ = time.time() - start

    async def _send_non_blocking(self, keys: list[str], min_timeout_sec: float) -> None:
        start = time.time()
        for cmd in self._tmux_send_keys(keys):
            rc, _, stderr = await self._exec(cmd)
            if rc != 0:
                raise RuntimeError(f"tmux send-keys failed: {stderr!r}")
        elapsed = time.time() - start
        if elapsed < min_timeout_sec:
            await asyncio.sleep(min_timeout_sec - elapsed)

    def _find_new_content(self, current_buffer: str) -> str | None:
        pb = "" if self._previous_buffer is None else self._previous_buffer.strip()
        if pb in current_buffer:
            idx = current_buffer.index(pb)
            if "\n" in pb:
                idx = pb.rfind("\n")
            return current_buffer[idx:]
        return None


def _limit_output(text: str, max_bytes: int = _OUTPUT_BYTE_LIMIT) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    half = max_bytes // 2
    head = data[:half].decode("utf-8", errors="ignore")
    tail = data[-half:].decode("utf-8", errors="ignore")
    omitted = len(data) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    return (
        f"{head}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{tail}"
    )


class TerminalExecute(AsyncActionMixin, BaseAction):
    """Send a batch of keystroke sequences to a persistent tmux pane.

    The pane survives across tool calls: commands modify cwd, env vars,
    and running processes the next call inherits.  Use one call per turn
    and list every keystroke sequence you want executed in order.
    """

    def __init__(
        self,
        session_name: str = "terminus2",
        pane_width: int = 160,
        pane_height: int = 40,
        working_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        description: dict | None = None,
    ):
        """Create the tool and start the tmux session immediately.

        Args:
            session_name (str): tmux session name.
            pane_width (int): pane width in columns.
            pane_height (int): pane height in rows.
            working_dir (str | None): cwd for the tmux server.
            extra_env (dict[str, str] | None): extra env vars for the pane.
            description (dict | None): override the auto-generated tool schema.
        """
        super().__init__(description=description)
        self._session = TmuxSession(
            session_name=session_name,
            pane_width=pane_width,
            pane_height=pane_height,
            working_dir=working_dir,
            extra_env=extra_env,
        )

    @property
    def session(self) -> TmuxSession:
        return self._session

    @tool_api
    async def run(self, commands: list) -> ActionReturn:
        """Execute a batch of keystroke sequences in the persistent tmux pane.

        Args:
            commands (list): list of command objects.  Each object is a dict
                with the following fields:
                - keystrokes (str): exact characters to send.  Append '\\n' to
                  press Enter and run the command.  Tmux key names are also
                  accepted: 'C-c', 'C-d', 'Enter', 'Tab'.
                - duration (float): seconds to wait after sending before the
                  next command (default 1.0, max 60.0).  Use 0.1 for fast
                  commands (ls, cd, echo), 1.0 for gcc/find, longer for
                  builds.  Prefer polling over long waits.

        Returns:
            str: terminal output produced by the batch (new pane content
                since the last call, or the visible screen if that is not
                determinable).
        """
        if not isinstance(commands, list):
            return ActionReturn(
                type=self.name,
                errmsg="`commands` must be a list of {keystrokes, duration} objects.",
                state=ActionStatusCode.ARGS_ERROR,
            )

        for raw in commands:
            if not isinstance(raw, dict) or "keystrokes" not in raw:
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Invalid command entry: {raw!r}",
                    state=ActionStatusCode.ARGS_ERROR,
                )
            keystrokes = raw["keystrokes"]
            try:
                duration = min(float(raw.get("duration", 1.0)), _DEFAULT_DURATION_CAP_SEC)
            except (TypeError, ValueError):
                duration = 1.0

            try:
                await self._session.send_keys(
                    keystrokes, block=False, min_timeout_sec=duration,
                )
            except TimeoutError:
                terminal_state = _limit_output(await self._session.get_incremental_output())
                return ActionReturn(
                    type=self.name,
                    result=[{
                        "type": "text",
                        "content": _TIMEOUT_TEMPLATE.format(
                            command=keystrokes,
                            timeout_sec=duration,
                            terminal_state=terminal_state,
                        ),
                    }],
                    state=ActionStatusCode.SUCCESS,
                )

        terminal_state = _limit_output(await self._session.get_incremental_output())
        return ActionReturn(
            type=self.name,
            result=[{"type": "text", "content": terminal_state}],
            state=ActionStatusCode.SUCCESS,
        )


class MarkTaskComplete(AsyncActionMixin, BaseAction):
    """Signal that the task is complete.

    Calling this once asks the orchestrator for confirmation; calling it
    again in the next turn finalizes the run.  The handshake is enforced
    by ``Terminus2Agent`` — the tool itself is stateless.
    """

    @tool_api
    async def run(self) -> ActionReturn:
        """Signal task completion.  Call twice in a row to confirm.

        Returns:
            str: placeholder result; the orchestrator rewrites this into the
                 confirmation prompt on the first call.
        """
        return ActionReturn(
            type=self.name,
            result=[{"type": "text", "content": "Task completion requested."}],
            state=ActionStatusCode.SUCCESS,
        )
