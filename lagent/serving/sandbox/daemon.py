"""Lagent Daemon — long-running process inside a sandbox that serves
lagent components over a Unix socket.

Architecture::

    BaseDaemon               ← socket server + protocol + ping/shutdown
      ├── ActionDaemon       ← Level 1: action calls + list_tools
      └── AgentDaemon        ← Level 2: chat + state_dict + reset

Protocol (length-prefixed JSON over Unix stream socket)::

    Request  → 4-byte big-endian length + JSON payload
    Response ← 4-byte big-endian length + JSON payload

Usage::

    # Level 1: actions only
    python -m lagent.actions.action_daemon start \\
        --mode actions --config actions.json --sock /tmp/lagent.sock

    # Level 2: full agent
    python -m lagent.actions.action_daemon start \\
        --mode agent --config agent.json --sock /tmp/lagent.sock

    # Call (same for both)
    python -m lagent.actions.action_daemon call \\
        --sock /tmp/lagent.sock '{"cmd":"ping"}'
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
from typing import Any, Dict, List, Optional, Union

from lagent.actions.action_executor import ActionExecutor, AsyncActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage, dataclass2dict
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
# BaseDaemon — socket server + protocol
# ---------------------------------------------------------------------------


class BaseDaemon:
    """Base class: asyncio Unix-socket server with JSON protocol.

    Subclasses implement ``_dispatch(request)`` to handle domain-specific
    commands.  Common commands (ping, shutdown) are handled here.

    Parameters
    ----------
    sock_path : str
        Path for the Unix domain socket.
    """

    daemon_type: str = "base"

    def __init__(self, sock_path: str = "/tmp/lagent_action.sock"):
        self.sock_path = sock_path
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        """Start listening. Removes stale socket file if present."""
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.sock_path)
        os.chmod(self.sock_path, 0o777)
        logger.info("%s listening on %s", self.__class__.__name__, self.sock_path)
        await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        logger.info("%s stopped", self.__class__.__name__)

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
            pass
        except Exception as e:
            logger.exception("Error handling client request")
            try:
                await _send_msg(writer, json.dumps({"error": str(e)}).encode())
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, request: dict) -> dict:
        """Handle a request. Override in subclasses for domain logic."""
        cmd = request.get("cmd")
        if cmd == "ping":
            return {"status": "ok", "type": self.daemon_type}
        if cmd == "shutdown":

            async def _delayed_close():
                await asyncio.sleep(0.1)
                if self._server:
                    self._server.close()

            asyncio.create_task(_delayed_close())
            return {"status": "shutting_down"}
        return {"error": f"Unknown command: {cmd}"}


# ---------------------------------------------------------------------------
# ActionDaemon — Level 1: action execution
# ---------------------------------------------------------------------------


class ActionDaemon(BaseDaemon):
    """Serves an ``AsyncActionExecutor`` over Unix socket.

    Usage::

        daemon = ActionDaemon(
            actions=[ShellAction(), ReadFileAction(), ...],
        )
        await daemon.start()

    Parameters
    ----------
    actions : list
        Action instances or config dicts.
    sock_path : str
        Unix socket path.
    """

    daemon_type = "action"

    def __init__(
        self,
        actions: Union[List[BaseAction], List[Dict]],
        sock_path: str = "/tmp/lagent_action.sock",
    ):
        super().__init__(sock_path=sock_path)
        for i, action in enumerate(actions):
            actions[i] = create_object(action)
        self.executor = AsyncActionExecutor(actions)

    async def _dispatch(self, request: dict) -> dict:
        cmd = request.get("cmd")

        # Common commands
        if cmd in ("ping", "shutdown"):
            return await super()._dispatch(request)

        if cmd == "list_tools":
            return {"tools": self.executor.description()}

        # Action call
        name = request.get("name")
        parameters = request.get("parameters", {})
        if not name:
            return dataclass2dict(
                ActionReturn(
                    errmsg="Missing 'name' in request",
                    state=ActionStatusCode.ARGS_ERROR,
                )
            )

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
# SkillsDaemon — skills loading
# ---------------------------------------------------------------------------


class SkillsDaemon(BaseDaemon):
    """Serves a ``SkillsLoader`` over Unix socket.

    Usage::

        daemon = SkillsDaemon(
            skills_loader=SkillsLoader(workspace),
        )
        await daemon.start()

    Parameters
    ----------
    skills_loader : SkillsLoader
        Skills loader for the sandbox workspace.
    sock_path : str
        Unix socket path.
    """

    daemon_type = "skills"

    def __init__(self, skills_loader, sock_path: str = "/tmp/lagent_skills.sock"):
        super().__init__(sock_path=sock_path)
        self.skills = skills_loader

    async def _dispatch(self, request: dict) -> dict:
        cmd = request.get("cmd")

        if cmd in ("ping", "shutdown"):
            return await super()._dispatch(request)

        if cmd == "list_skills":
            filter_unavailable = request.get("filter_unavailable", True)
            return {"skills": await self.skills.list_skills(filter_unavailable=filter_unavailable)}

        if cmd == "skills_summary":
            return {"summary": await self.skills.build_skills_summary()}

        if cmd == "load_skill":
            content = await self.skills.load_skill(request.get("name", ""))
            return {"content": content}

        if cmd == "load_skills_for_context":
            content = await self.skills.load_skills_for_context(request.get("names", []))
            return {"content": content}

        if cmd == "get_always_skills":
            return {"skills": await self.skills.get_always_skills()}

        return {"error": f"Unknown command: {cmd}"}


# ---------------------------------------------------------------------------
# AgentDaemon — Level 2: full agent
# ---------------------------------------------------------------------------


class AgentDaemon(BaseDaemon):
    """Serves a full ``Agent`` over Unix socket.

    The agent runs entirely inside the sandbox — LLM calls, action
    execution, skills, memory — everything is local to the daemon.

    Usage::

        daemon = AgentDaemon(
            agent=InternClawAgent(
                policy_agent=...,
                env_agent=AsyncEnvAgent(actions=[ShellAction(), ...]),
            ),
        )
        await daemon.start()

    Parameters
    ----------
    agent : Agent or dict
        Agent instance or config dict (passed to ``create_object``).
    sock_path : str
        Unix socket path.
    """

    daemon_type = "agent"

    def __init__(self, agent, sock_path: str = "/tmp/lagent_action.sock"):
        super().__init__(sock_path=sock_path)
        self.agent = create_object(agent)

    async def _dispatch(self, request: dict) -> dict:
        cmd = request.get("cmd")

        # Common commands
        if cmd in ("ping", "shutdown"):
            return await super()._dispatch(request)

        # Tool introspection (via EnvAgent if available)
        if cmd == "list_tools":
            env = getattr(self.agent, 'env_agent', None)
            executor = getattr(env, 'actions', None) if env else None
            if executor:
                return {"tools": executor.description()}
            return {"tools": []}

        # Agent commands
        if cmd == "chat":
            messages = request.get("messages", [])
            try:
                response = await self.agent(*messages)
                return self._serialize_agent_message(response)
            except Exception as e:
                logger.exception("Agent chat failed")
                return {"error": str(e)}

        if cmd == "state_dict":
            try:
                return {"state_dict": self.agent.state_dict()}
            except Exception as e:
                return {"error": str(e)}

        if cmd == "load_state_dict":
            try:
                self.agent.load_state_dict(request["state_dict"])
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}

        if cmd == "reset":
            try:
                self.agent.reset(recursive=request.get("recursive", True))
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}

        if cmd == 'get_messages':
            try:
                return self.agent.get_messages()
            except Exception as e:
                return {"error": str(e)}

        return {"error": f"Unknown command: {cmd}"}

    @staticmethod
    def _serialize_agent_message(msg: AgentMessage) -> dict:
        data = msg.model_dump()
        if isinstance(msg.content, ActionReturn):
            data["content"] = dataclass2dict(msg.content)
        return data


# ---------------------------------------------------------------------------
# lagent-call: one-shot CLI client
# ---------------------------------------------------------------------------


def lagent_call(sock_path: str, request_json: str) -> str:
    """Send a single request to the daemon and return the response JSON.

    Synchronous — suitable for CLI and subprocess usage.
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


def _load_config(config_path: str) -> Union[List[Dict], Dict]:
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="lagent.actions.action_daemon",
        description="Lagent Daemon: serve actions or agents over Unix socket",
    )
    sub = parser.add_subparsers(dest="command")

    # -- start --
    p_start = sub.add_parser("start", help="Start the daemon")
    p_start.add_argument(
        "--sock",
        default="/tmp/lagent_action.sock",
        help="Unix socket path",
    )
    p_start.add_argument(
        "--mode",
        choices=["actions", "agent"],
        default="actions",
        help="'actions' for Level 1 (ActionDaemon), 'agent' for Level 2 (AgentDaemon)",
    )
    p_start.add_argument(
        "--config",
        help="Path to JSON config (action list for Level 1, agent dict for Level 2)",
    )
    # Backward compat
    p_start.add_argument("--actions-config", help=argparse.SUPPRESS)
    p_start.add_argument("--agent-config", help=argparse.SUPPRESS)

    # -- call --
    p_call = sub.add_parser("call", help="Send a one-shot request")
    p_call.add_argument("--sock", default="/tmp/lagent_action.sock")
    p_call.add_argument("request", help="JSON request string")

    args = parser.parse_args()

    if args.command == "start":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

        # Handle backward compat flags
        mode = args.mode
        config_path = args.config
        if args.actions_config:
            mode = "actions"
            config_path = args.actions_config
        elif args.agent_config:
            mode = "agent"
            config_path = args.agent_config

        if not config_path:
            parser.error("Must provide --config (or --actions-config / --agent-config)")

        config = _load_config(config_path)

        if mode == "agent":
            daemon = AgentDaemon(agent=config, sock_path=args.sock)
        else:
            daemon = ActionDaemon(actions=config, sock_path=args.sock)

        logger.info("Starting %s", daemon.__class__.__name__)
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
