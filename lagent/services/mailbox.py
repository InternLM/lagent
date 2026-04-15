"""Mailbox -- shared message service for multi-agent communication.

A centralised message store with per-agent queues.  Agents send
:class:`~lagent.schema.AgentMessage` objects to named recipients;
recipients drain their queue at natural checkpoints in their own
execution loop (typically inside ``EnvAgent.get_env_info()``).

Design
------
* **Shared resource** -- one Mailbox instance shared across all agents
  in a team, like :class:`TaskBoard`.
* **Centralised storage, decentralised routing** -- sender writes
  directly into receiver's queue, no middleman.
* **asyncio.Event notification** -- receivers can ``await
  wait_for_message()`` instead of polling with ``asyncio.sleep()``.
* **state_dict / load_state_dict** -- serialisable for persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lagent.schema import AgentMessage

logger = logging.getLogger("lagent.services.mailbox")


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── serialisation helpers ────────────────────────────────────────────

def _msg_to_dict(msg: AgentMessage) -> dict:
    """Convert an AgentMessage to a plain dict for persistence."""
    return {
        "sender": msg.sender,
        "receiver": msg.receiver,
        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
        "timestamp": msg.timestamp,
    }


def _msg_from_dict(data: dict) -> AgentMessage:
    """Reconstruct an AgentMessage from a persistence dict."""
    return AgentMessage(
        sender=data.get("sender", "unknown"),
        receiver=data.get("receiver"),
        content=data.get("content", ""),
        timestamp=data.get("timestamp", ""),
    )


# ── persistence ──────────────────────────────────────────────────────

def _load_mailbox(path: Path) -> dict[str, list[AgentMessage]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            return {}
        queues = {}
        for name, msgs in data.get("queues", {}).items():
            queues[name] = [
                _msg_from_dict(m) for m in msgs if isinstance(m, dict)
            ]
        return queues
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load mailbox store %s: %s", path, exc)
        return {}


def _save_mailbox(queues: dict[str, list[AgentMessage]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "queues": {
            name: [_msg_to_dict(m) for m in msgs]
            for name, msgs in queues.items()
        },
    }
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to save mailbox store: %s", exc)
        tmp.unlink(missing_ok=True)


# ── Mailbox ──────────────────────────────────────────────────────────

class Mailbox:
    """Shared message service for multi-agent communication.

    Parameters
    ----------
    store_path : Path or None
        Where to persist mailbox state.  ``None`` means purely
        in-memory (no file I/O).
    """

    def __init__(self, store_path: Path | None = None):
        self._queues: dict[str, list[AgentMessage]] = {}
        self._notify: dict[str, asyncio.Event] = {}
        self._store_path = store_path

        if store_path is not None:
            loaded = _load_mailbox(store_path)
            if loaded:
                self._queues = loaded

    # ── registration ─────────────────────────────────────────────

    def register(self, agent_name: str) -> None:
        """Register an agent, initialising its queue and notification
        event.  Idempotent — safe to call multiple times."""
        self._queues.setdefault(agent_name, [])
        if agent_name not in self._notify:
            self._notify[agent_name] = asyncio.Event()

    @property
    def agents(self) -> list[str]:
        """Names of all registered agents."""
        return list(self._queues.keys())

    # ── send ─────────────────────────────────────────────────────

    def send(self, message: AgentMessage) -> None:
        """Send a message.  Uses ``message.receiver`` for routing.

        If ``receiver`` is ``"*"``, the message is broadcast to every
        registered agent except the sender.
        """
        receiver = message.receiver
        if not receiver:
            raise ValueError("AgentMessage.receiver must be set")

        if receiver == "*":
            for name in list(self._queues.keys()):
                if name != message.sender:
                    self._queues[name].append(message)
                    self._signal(name)
        else:
            self._queues.setdefault(receiver, []).append(message)
            self._signal(receiver)

        self._persist()

    # ── receive ──────────────────────────────────────────────────

    def drain(self, agent_name: str) -> list[AgentMessage]:
        """Take all pending messages for *agent_name* and clear its
        queue.  Intended to be called from ``EnvAgent.get_env_info()``.
        """
        msgs = self._queues.get(agent_name)
        if not msgs:
            return []
        taken = list(msgs)
        msgs.clear()
        self._persist()
        return taken

    def has_messages(self, agent_name: str) -> bool:
        """Quick non-destructive check."""
        return bool(self._queues.get(agent_name))

    def peek(self, agent_name: str) -> list[AgentMessage]:
        """View pending messages without removing them."""
        return list(self._queues.get(agent_name, []))

    # ── wait (async notification) ────────────────────────────────

    async def wait_for_message(
        self, agent_name: str, timeout: float | None = None,
    ) -> bool:
        """Block until a message arrives for *agent_name*.

        Returns ``True`` if a message arrived, ``False`` on timeout.
        Use this in worker loops to avoid ``asyncio.sleep()`` polling::

            if not board.list_available():
                await mailbox.wait_for_message(name, timeout=5.0)
        """
        if agent_name not in self._notify:
            self.register(agent_name)
        event = self._notify[agent_name]
        event.clear()
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _signal(self, agent_name: str) -> None:
        event = self._notify.get(agent_name)
        if event is not None:
            event.set()

    # ── management ───────────────────────────────────────────────

    def clear(self, agent_name: str) -> None:
        """Discard all pending messages for one agent."""
        self._queues.pop(agent_name, None)
        self._persist()

    def clear_all(self) -> None:
        """Discard all messages for all agents."""
        for q in self._queues.values():
            q.clear()
        self._persist()

    def message_count(self, agent_name: str | None = None) -> int:
        """Total pending messages, optionally filtered by agent."""
        if agent_name is not None:
            return len(self._queues.get(agent_name, []))
        return sum(len(q) for q in self._queues.values())

    # ── serialisation ────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "queues": {
                name: [_msg_to_dict(m) for m in msgs]
                for name, msgs in self._queues.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self._queues = {}
        for name, msgs in state.get("queues", {}).items():
            self._queues[name] = [
                _msg_from_dict(d) for d in msgs if isinstance(d, dict)
            ]
        self._persist()

    # ── persistence ──────────────────────────────────────────────

    def _persist(self) -> None:
        if self._store_path is not None:
            _save_mailbox(self._queues, self._store_path)
