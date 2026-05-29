"""Small CLI client for Lagent sandbox daemons.

This module is intentionally a client, not another runtime abstraction.  It
turns the daemon's JSON socket protocol into rollout-friendly shell commands
so an outer runner can record each phase as an explicit entry.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MSG_SIZE = 64 * 1024 * 1024


class DaemonCallError(RuntimeError):
    pass


def _call_json(sock: str, payload: dict[str, Any]) -> dict[str, Any] | list:
    raw = asyncio.run(_async_call(sock, json.dumps(payload, ensure_ascii=False).encode()))
    try:
        obj = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise DaemonCallError(f"invalid daemon response: {exc}: {raw}") from exc
    if isinstance(obj, dict) and obj.get("error"):
        raise DaemonCallError(str(obj["error"]))
    if not isinstance(obj, (dict, list)):
        raise DaemonCallError(f"daemon response must be a JSON object or array, got {type(obj).__name__}")
    return obj


async def _async_call(sock: str, payload: bytes) -> str:
    reader, writer = await asyncio.open_unix_connection(sock)
    try:
        writer.write(struct.pack(_HEADER_FMT, len(payload)))
        writer.write(payload)
        await writer.drain()

        header = await reader.readexactly(_HEADER_SIZE)
        (length,) = struct.unpack(_HEADER_FMT, header)
        if length > _MAX_MSG_SIZE:
            raise DaemonCallError(f"daemon response too large: {length} bytes")
        raw = await reader.readexactly(length)
        return raw.decode()
    finally:
        writer.close()
        await writer.wait_closed()


def _write_json(path: str, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=4, default=str), encoding="utf-8")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)


def _tail_log(path: str | None, lines: int = 100) -> str:
    if not path:
        return ""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _die(message: str, code: int, *, log: str | None = None) -> int:
    print(message, file=sys.stderr, flush=True)
    tail = _tail_log(log)
    if tail:
        print(tail, file=sys.stderr, flush=True)
    return code


def cmd_wait_ready(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    while time.monotonic() <= deadline:
        pid = _read_pid(args.pid_file)
        if pid is not None and not _pid_alive(pid):
            return _die(f"daemon pid {pid} exited before ready", 4, log=args.log)
        try:
            response = _call_json(args.sock, {"cmd": "ping"})
        except Exception:
            time.sleep(args.poll_interval)
            continue
        _print_json(response)
        return 0
    return _die(f"daemon did not become ready within {args.timeout}s", 4, log=args.log)


def cmd_start_agent_daemon(args: argparse.Namespace) -> int:
    try:
        for path in (args.sock, args.pid_file):
            if path and os.path.exists(path):
                os.unlink(path)
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "w" if args.truncate_log else "a"
        log_file = log_path.open(log_mode, encoding="utf-8", errors="replace")
        cmd = [
            sys.executable,
            "-m",
            "lagent.serving.sandbox.daemon",
            "start",
            "--mode",
            args.mode,
            "--config",
            args.config,
            "--sock",
            args.sock,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        Path(args.pid_file).write_text(str(proc.pid), encoding="utf-8")
        _print_json({"status": "started", "pid": proc.pid, "sock": args.sock, "log": args.log})
        return 0
    except Exception as exc:
        return _die(f"failed to start daemon: {exc}", 4, log=args.log)


def cmd_chat(args: argparse.Namespace) -> int:
    try:
        instruction = Path(args.instruction_file).read_text(encoding="utf-8")
        response = _call_json(args.sock, {"cmd": "chat", "messages": [instruction]})
        content = response.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        Path(args.response_out).write_text(content or "", encoding="utf-8")
        _print_json(response)
        return 0
    except Exception as exc:
        return _die(f"daemon error in chat: {exc}", 5, log=args.log)


def _trajectory_payload(response: dict[str, Any]) -> dict[str, Any]:
    state = response.get("state_dict", response)
    if isinstance(state, dict) and "trajectory" in state:
        return state
    if isinstance(state, dict):
        flat = []
        for value in state.values():
            if isinstance(value, list):
                flat.extend(item for item in value if isinstance(item, dict))
        return {"trajectory": flat, "raw": state}
    return {"trajectory": state if isinstance(state, list) else [], "raw": state}


def cmd_state_dict(args: argparse.Namespace) -> int:
    try:
        response = _call_json(args.sock, {"cmd": "state_dict"})
        payload = _trajectory_payload(response)
        _write_json(args.trajectory_out, payload)
        _print_json({"status": "ok", "trajectory_out": args.trajectory_out})
        return 0
    except Exception as exc:
        return _die(f"daemon error in state_dict: {exc}", 6, log=args.log)


def cmd_get_messages(args: argparse.Namespace) -> int:
    try:
        response = _call_json(args.sock, {"cmd": "get_messages"})
        _write_json(args.message_out, response)
        _print_json({"status": "ok", "message_out": args.message_out})
        return 0
    except Exception as exc:
        return _die(f"daemon error in get_messages: {exc}", 7, log=args.log)


def cmd_shutdown(args: argparse.Namespace) -> int:
    try:
        response = _call_json(args.sock, {"cmd": "shutdown"})
        _print_json(response)
    except Exception:
        pass
    pid = _read_pid(args.pid_file)
    if pid is not None and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lagent sandbox daemon client")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("wait-ready", help="Wait until daemon responds to ping")
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--pid-file")
    p.add_argument("--log")
    p.set_defaults(func=cmd_wait_ready)

    p = sub.add_parser("start-agent-daemon", help="Start an AgentDaemon in the background")
    p.add_argument("--mode", choices=["agent"], default="agent")
    p.add_argument("--config", required=True)
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--pid-file", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--truncate-log", action="store_true")
    p.set_defaults(func=cmd_start_agent_daemon)

    p = sub.add_parser("chat", help="Send one instruction file to AgentDaemon.chat")
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--response-out", required=True)
    p.add_argument("--log")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("state-dict", help="Dump AgentDaemon.state_dict as trajectory JSON")
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--trajectory-out", required=True)
    p.add_argument("--log")
    p.set_defaults(func=cmd_state_dict)

    p = sub.add_parser("get-messages", help="Dump AgentDaemon.get_messages JSON")
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--message-out", required=True)
    p.add_argument("--log")
    p.set_defaults(func=cmd_get_messages)

    p = sub.add_parser("shutdown", help="Ask daemon to shutdown, then terminate pid file if needed")
    p.add_argument("--sock", default="/tmp/lagent_agent.sock")
    p.add_argument("--pid-file")
    p.add_argument("--log")
    p.set_defaults(func=cmd_shutdown)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
