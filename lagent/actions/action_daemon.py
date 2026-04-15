"""ActionDaemon — a long-running process inside a sandbox that holds an
ActionExecutor and serves action calls over a Unix socket.

Two components:
  1. **ActionDaemon**: asyncio server, listens on a Unix socket, dispatches
     JSON requests to ``AsyncActionExecutor.forward()``.
  2. **lagent-call**: one-shot CLI that connects to the daemon, sends a
     request, prints the JSON response to stdout, then exits.

Protocol (length-prefixed JSON over Unix stream socket)::

    Request  → 4-byte big-endian length + JSON payload
    Response ← 4-byte big-endian length + JSON payload

Request payload::

    {"name": "shell", "parameters": {"command": "ls"}}
    or
    {"cmd": "list_tools"}    # introspection
    {"cmd": "ping"}          # health check
    {"cmd": "shutdown"}      # graceful stop

Response payload (for action calls)::

    ActionReturn serialised via ``dataclass2dict``.

Usage::

    # Start daemon inside sandbox
    python -m lagent.actions.action_daemon start \\
        --sock /tmp/lagent_action.sock \\
        --actions-config /path/to/actions.json

    # Call from bash
    python -m lagent.actions.action_daemon call \\
        --sock /tmp/lagent_action.sock \\
        '{"name":"shell","parameters":{"command":"ls"}}'

    # List available tools
    python -m lagent.actions.action_daemon call \\
        --sock /tmp/lagent_action.sock \\
        '{"cmd":"list_tools"}'
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lagent.actions.action_executor import ActionExecutor, AsyncActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode, dataclass2dict
from lagent.utils import create_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wire protocol helpers
# ---------------------------------------------------------------------------

_HEADER_FMT = "!I"  # 4-byte unsigned big-endian
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MSG_SIZE = 64 * 1024 * 1024  # 64 MiB safety cap


async def _send_msg(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(struct.pack(_HEADER_FMT, len(data)))
    writer.write(data)
    await writer.drain()


async def _recv_msg(reader: asyncio.StreamReader) -> Optional[bytes]:
    header = await reader.readexactly(_HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FMT, header)
    if length > _MAX_MSG_SIZE:
        raise ValueError(f"Message too large: {length} bytes")
    return await reader.readexactly(length)


# ---------------------------------------------------------------------------
# ActionDaemon
# ---------------------------------------------------------------------------


class ActionDaemon:
    """Asyncio Unix-socket server that wraps an ``AsyncActionExecutor``.

    Parameters
    ----------
    actions : list
        Action instances or dicts (passed to ``create_object``).
    sock_path : str
        Path for the Unix domain socket.
    """

    def __init__(
        self,
        actions: Union[List[BaseAction], List[Dict]],
        sock_path: str = "/tmp/lagent_action.sock",
    ):
        for i, action in enumerate(actions):
            actions[i] = create_object(action)
        self.executor = AsyncActionExecutor(actions)
        self.sock_path = sock_path
        self._server: Optional[asyncio.AbstractServer] = None

    # -- public API --

    async def start(self) -> None:
        """Start listening. Removes stale socket file if present."""
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.sock_path
        )
        # Make socket world-accessible (inside sandbox this is fine)
        os.chmod(self.sock_path, 0o777)
        logger.info("ActionDaemon listening on %s", self.sock_path)
        await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        logger.info("ActionDaemon stopped")

    # -- internals --

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await _recv_msg(reader)
            request = json.loads(raw)
            response = await self._dispatch(request)
            await _send_msg(writer, json.dumps(response, ensure_ascii=False).encode())
        except asyncio.IncompleteReadError:
            pass  # client disconnected
        except Exception as e:
            logger.exception("Error handling client request")
            try:
                err = {"error": str(e)}
                await _send_msg(writer, json.dumps(err).encode())
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, request: dict) -> dict:
        # Control commands
        cmd = request.get("cmd")
        if cmd == "ping":
            return {"status": "ok"}
        if cmd == "list_tools":
            return {"tools": self.executor.description()}
        if cmd == "shutdown":
            # Schedule server close after responding
            async def _delayed_close():
                await asyncio.sleep(0.1)
                if self._server:
                    self._server.close()
            asyncio.create_task(_delayed_close())
            return {"status": "shutting_down"}

        # Action call
        name = request.get("name")
        parameters = request.get("parameters", {})
        if not name:
            return dataclass2dict(ActionReturn(
                errmsg="Missing 'name' in request",
                state=ActionStatusCode.ARGS_ERROR,
            ))

        try:
            action_return = await self.executor.forward(name, parameters)
        except Exception as e:
            logger.exception("Action %s failed", name)
            action_return = ActionReturn(
                args=parameters,
                type=name,
                errmsg=str(e),
                state=ActionStatusCode.API_ERROR,
            )

        return dataclass2dict(action_return)


# ---------------------------------------------------------------------------
# lagent-call: one-shot CLI client
# ---------------------------------------------------------------------------


def lagent_call(sock_path: str, request_json: str) -> str:
    """Send a single request to the daemon and return the response JSON.

    This is a synchronous function suitable for CLI usage.
    """
    return asyncio.run(async_lagent_call(sock_path, request_json))


async def async_lagent_call(sock_path: str, request_json: str) -> str:
    """Async variant of :func:`lagent_call`."""
    reader, writer = await asyncio.open_unix_connection(sock_path)
    try:
        await _send_msg(writer, request_json.encode())
        raw = await _recv_msg(reader)
        return raw.decode()
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_actions_from_config(config_path: str) -> List[Dict]:
    """Load action definitions from a JSON config file.

    Expected format::

        [
            {"type": "lagent.actions.shell.ShellAction", "working_dir": "/workspace"},
            {"type": "lagent.actions.ipython_interpreter.AsyncIPythonInterpreter"},
            ...
        ]
    """
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="lagent.actions.action_daemon",
        description="ActionDaemon: serve lagent actions over Unix socket",
    )
    sub = parser.add_subparsers(dest="command")

    # -- start --
    p_start = sub.add_parser("start", help="Start the daemon")
    p_start.add_argument(
        "--sock", default="/tmp/lagent_action.sock",
        help="Unix socket path (default: /tmp/lagent_action.sock)",
    )
    p_start.add_argument(
        "--actions-config", required=True,
        help="Path to JSON file listing action configs",
    )

    # -- call --
    p_call = sub.add_parser("call", help="Send a one-shot request to the daemon")
    p_call.add_argument(
        "--sock", default="/tmp/lagent_action.sock",
        help="Unix socket path",
    )
    p_call.add_argument(
        "request", help="JSON request string",
    )

    args = parser.parse_args()

    if args.command == "start":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        actions = _load_actions_from_config(args.actions_config)
        daemon = ActionDaemon(actions=actions, sock_path=args.sock)
        try:
            asyncio.run(daemon.start())
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Interrupted, shutting down")
    elif args.command == "call":
        result = lagent_call(args.sock, args.request)
        print(result, flush=True)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
