"""AgentService — lifecycle manager for agent instances.

Core interface: ``config_dict + build → Agent``.

Responsibilities
----------------
* **Template registry**: loads :class:`AgentSpec` via :class:`AgentLoader`.
* **Instance creation**: ``AgentSpec.acreate()`` (or custom build).
* **Registry**: tracks running/stopped instances with IDs.
* **Execution**: sync (await) or async (``asyncio.Task``).
* **Persistence**: ``save_all()`` / ``load_all()`` via ``state_dict()``.

Design decisions
----------------
* No fork — all agents are built from config.  State transfer uses
  ``state_dict()`` / ``load_state_dict()`` when needed.
* ID is auto-generated (``uuid.uuid4().hex[:8]``).
* Service is decoupled from channels/bus.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from lagent.agents.agent import Agent

from .agent_loader import AgentLoader, AgentSpec

logger = logging.getLogger("lagent.interclaw.services.agent")


# ── helpers ───────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


# ── data model ────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class AgentEntry:
    """Registry record for a managed agent instance."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_type: str = ""
    label: str = ""
    task: str = ""
    status: str = AgentStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at_ms: int = field(default_factory=_now_ms)
    finished_at_ms: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "label": self.label,
            "task": self.task,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at_ms": self.created_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentEntry:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            agent_type=data.get("agent_type", ""),
            label=data.get("label", ""),
            task=data.get("task", ""),
            status=data.get("status", AgentStatus.PENDING),
            result=data.get("result"),
            error=data.get("error"),
            created_at_ms=data.get("created_at_ms", _now_ms()),
            finished_at_ms=data.get("finished_at_ms"),
        )


# ── AgentService ──────────────────────────────────────────────────────

class AgentService:
    """Manages the full lifecycle of dynamically spawned agents.

    Responsibilities: spec registry, instance creation from specs,
    lifecycle tracking (run/stop/resume), and persistence.

    Agent *construction* logic (LLM injection, tool selection) lives
    in the caller (e.g. :class:`SubAgentAction`), not here.  This
    service only needs a built :class:`Agent` or a registered
    :class:`AgentSpec`.

    Parameters
    ----------
    agent_loader : AgentLoader, optional
        Discovers and parses agent templates from the filesystem.
    max_concurrent : int
        Maximum number of concurrently running async agents.
    on_complete : callable, optional
        ``async (entry: AgentEntry) -> None`` called when an async agent
        finishes.
    """

    def __init__(
        self,
        agent_loader: AgentLoader | None = None,
        max_concurrent: int = 5,
        on_complete: Callable[[AgentEntry], Awaitable[None]] | None = None,
    ):
        self._loader = agent_loader
        self._max_concurrent = max_concurrent
        self._on_complete = on_complete
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Registry
        self._entries: dict[str, AgentEntry] = {}
        self._agents: dict[str, Agent] = {}  # live instances
        self._tasks: dict[str, asyncio.Task] = {}  # async tasks

        # Spec cache
        self._specs: dict[str, AgentSpec] = {}

        # Saved states (for resume)
        self._saved_states: dict[str, dict] = {}

    # ── initialisation ────────────────────────────────────────────

    async def load_specs(self) -> None:
        """Load all agent templates from the AgentLoader."""
        if self._loader is None:
            return
        self._specs = await self._loader.load_all()
        logger.info(
            "Loaded %d agent specs: %s",
            len(self._specs), list(self._specs.keys()),
        )

    def register_spec(self, spec: AgentSpec) -> None:
        """Manually register an AgentSpec."""
        self._specs[spec.name] = spec

    def get_spec(self, agent_type: str) -> AgentSpec | None:
        """Get a registered spec by name."""
        return self._specs.get(agent_type)

    @property
    def available_types(self) -> list[str]:
        """List all registered agent type names."""
        return list(self._specs.keys())

    # ── spawn ─────────────────────────────────────────────────────

    async def spawn(
        self,
        task: str,
        agent_type: str,
        *,
        label: str | None = None,
        mode: str = "sync",
        state: dict | None = None,
        **spec_kwargs,
    ) -> AgentEntry:
        """Create and run a new agent from a registered spec.

        Parameters
        ----------
        task : str
            Task description / instruction.
        agent_type : str
            Registered agent type name.
        label : str, optional
            Human-readable label.
        mode : str
            ``"sync"`` — block until done.
            ``"async"`` / ``"background"`` — run as asyncio.Task.
        state : dict, optional
            If provided, ``load_state_dict(state)`` is called on the
            new agent before running.
        **spec_kwargs
            Extra keyword arguments forwarded to
            ``AgentSpec.acreate()`` (e.g. ``llm=``, ``actions=``).

        Returns
        -------
        AgentEntry
        """
        agent = await self._create_from_spec(agent_type, **spec_kwargs)

        label = label or (task[:40] + ("…" if len(task) > 40 else ""))
        entry = AgentEntry(
            agent_type=agent_type,
            label=label,
            task=task,
            status=AgentStatus.PENDING,
        )
        self._entries[entry.id] = entry

        logger.info(
            "Spawning agent [%s] type=%s mode=%s: %s",
            entry.id, agent_type, mode, label,
        )

        # Optional state transfer
        if state is not None:
            try:
                agent.load_state_dict(state)
            except Exception as exc:
                logger.warning(
                    "Failed to load state for [%s]: %s", entry.id, exc,
                )

        if mode == "sync":
            await self._run_sync(entry, agent, task)
        else:
            self._run_async(entry, agent, task)

        return entry

    async def spawn_agent(
        self,
        agent: Agent,
        task: str,
        *,
        label: str | None = None,
        mode: str = "sync",
        agent_type: str = "_custom",
    ) -> AgentEntry:
        """Run a pre-built Agent instance directly.

        Use this when you have an already-constructed Agent (e.g. from
        a custom factory) and want it managed by the service.

        Parameters
        ----------
        agent : Agent
            A fully constructed agent instance.
        task : str
            Task description.
        label : str, optional
            Human-readable label.
        mode : str
            ``"sync"`` or ``"async"``.
        agent_type : str
            Label for the entry's ``agent_type`` field.
        """
        label = label or (task[:40] + ("…" if len(task) > 40 else ""))
        entry = AgentEntry(
            agent_type=agent_type,
            label=label,
            task=task,
            status=AgentStatus.PENDING,
        )
        self._entries[entry.id] = entry

        logger.info(
            "Spawning pre-built agent [%s] type=%s mode=%s: %s",
            entry.id, agent_type, mode, label,
        )

        if mode == "sync":
            await self._run_sync(entry, agent, task)
        else:
            self._run_async(entry, agent, task)

        return entry

    async def _create_from_spec(self, agent_type: str, **kwargs) -> Agent:
        """Create agent from a registered spec."""
        spec = self._specs.get(agent_type)
        if spec is None and self._loader is not None:
            spec = await self._loader.load(agent_type)
            if spec is not None:
                self._specs[agent_type] = spec
        if spec is None:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Available: {self.available_types}"
            )
        return await spec.acreate(**kwargs)

    async def _run_sync(
        self, entry: AgentEntry, agent: Agent, task: str,
    ) -> None:
        """Run agent synchronously."""
        entry.status = AgentStatus.RUNNING
        try:
            async with self._semaphore:
                self._agents[entry.id] = agent
                response = await agent(task)
                entry.result = (
                    response.content if hasattr(response, "content")
                    else str(response)
                )
                entry.status = AgentStatus.STOPPED
                entry.finished_at_ms = _now_ms()
        except Exception as exc:
            entry.status = AgentStatus.FAILED
            entry.error = str(exc)
            entry.finished_at_ms = _now_ms()
            logger.error("Agent [%s] sync failed: %s", entry.id, exc)
        finally:
            finished_agent = self._agents.pop(entry.id, None)
            if finished_agent is not None:
                try:
                    self._saved_states[entry.id] = finished_agent.state_dict()
                except Exception as exc:
                    logger.warning(
                        "Failed to save state for [%s]: %s", entry.id, exc,
                    )

    def _run_async(
        self, entry: AgentEntry, agent: Agent, task: str,
    ) -> None:
        """Run agent asynchronously — create asyncio.Task."""
        async def _run() -> None:
            entry.status = AgentStatus.RUNNING
            try:
                async with self._semaphore:
                    self._agents[entry.id] = agent
                    response = await agent(task)
                    entry.result = (
                        response.content if hasattr(response, "content")
                        else str(response)
                    )
                    entry.status = AgentStatus.STOPPED
                    entry.finished_at_ms = _now_ms()
            except Exception as exc:
                entry.status = AgentStatus.FAILED
                entry.error = str(exc)
                entry.finished_at_ms = _now_ms()
                logger.error("Agent [%s] async failed: %s", entry.id, exc)
            finally:
                finished_agent = self._agents.pop(entry.id, None)
                if finished_agent is not None:
                    try:
                        self._saved_states[entry.id] = finished_agent.state_dict()
                    except Exception as exc:
                        logger.warning(
                            "Failed to save state for [%s]: %s",
                            entry.id, exc,
                        )
                self._tasks.pop(entry.id, None)
                if self._on_complete:
                    try:
                        await self._on_complete(entry)
                    except Exception as cb_exc:
                        logger.error(
                            "on_complete failed for [%s]: %s",
                            entry.id, cb_exc,
                        )

        task_obj = asyncio.create_task(_run(), name=f"agent-{entry.id}")
        self._tasks[entry.id] = task_obj

    # ── query ─────────────────────────────────────────────────────

    def list(
        self,
        *,
        status: str | None = None,
        agent_type: str | None = None,
    ) -> list[AgentEntry]:
        """List managed agent entries, optionally filtered."""
        entries = list(self._entries.values())
        if status is not None:
            entries = [e for e in entries if e.status == status]
        if agent_type is not None:
            entries = [e for e in entries if e.agent_type == agent_type]
        return entries

    def get(self, agent_id: str) -> AgentEntry | None:
        """Get a single entry by ID."""
        return self._entries.get(agent_id)

    # ── resume ────────────────────────────────────────────────────

    async def resume(self, agent_id: str, message: str) -> AgentEntry:
        """Resume a stopped agent with a new message.

        Recreates the agent from spec, restores saved state, and sends
        the new message.  Reuses the existing AgentEntry rather than
        creating a new one.
        """
        entry = self._entries.get(agent_id)
        if entry is None:
            raise ValueError(f"Agent {agent_id!r} not found")
        if entry.status == AgentStatus.RUNNING:
            raise ValueError(f"Agent {agent_id!r} is still running")

        spec = self._specs.get(entry.agent_type)
        if spec is None:
            raise ValueError(
                f"Agent spec {entry.agent_type!r} not found, "
                f"cannot resume agent {agent_id!r}"
            )

        agent = await spec.acreate()

        # Restore saved conversation state if available
        saved_state = self._saved_states.get(agent_id)
        if saved_state is not None:
            try:
                agent.load_state_dict(saved_state)
            except Exception as exc:
                logger.warning(
                    "Failed to restore state for [%s]: %s", agent_id, exc,
                )

        # Reset entry for the new run
        entry.result = None
        entry.error = None
        entry.finished_at_ms = None

        await self._run_sync(entry, agent, message)
        return entry

    # ── stop ──────────────────────────────────────────────────────

    async def stop(self, agent_id: str) -> bool:
        """Stop a running async agent. Returns True if cancelled."""
        task = self._tasks.get(agent_id)
        if task is None:
            return False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        entry = self._entries.get(agent_id)
        if entry and entry.status == AgentStatus.RUNNING:
            entry.status = AgentStatus.STOPPED
            entry.finished_at_ms = _now_ms()

        self._tasks.pop(agent_id, None)
        self._agents.pop(agent_id, None)
        logger.info("Agent [%s] stopped", agent_id)
        return True

    # ── persistence ───────────────────────────────────────────────

    async def save_all(self, path: Path) -> None:
        """Save all entries and live agent states to disk."""
        path.mkdir(parents=True, exist_ok=True)

        entries_data = [e.to_dict() for e in self._entries.values()]
        entries_file = path / "agents.json"
        await asyncio.to_thread(
            entries_file.write_text,
            json.dumps(entries_data, ensure_ascii=False, indent=2),
            "utf-8",
        )

        states_dir = path / "states"
        states_dir.mkdir(exist_ok=True)
        for agent_id, agent in self._agents.items():
            try:
                state = agent.state_dict()
                state_file = states_dir / f"{agent_id}.json"
                await asyncio.to_thread(
                    state_file.write_text,
                    json.dumps(state, ensure_ascii=False, indent=2),
                    "utf-8",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save state for [%s]: %s", agent_id, exc,
                )

        logger.info("Saved %d entries to %s", len(entries_data), path)

    async def load_all(self, path: Path) -> None:
        """Load entries and states from disk."""
        entries_file = path / "agents.json"
        if not entries_file.exists():
            return

        raw = await asyncio.to_thread(entries_file.read_text, "utf-8")
        for data in json.loads(raw):
            entry = AgentEntry.from_dict(data)
            self._entries[entry.id] = entry

        states_dir = path / "states"
        if states_dir.exists():
            for state_file in states_dir.iterdir():
                if state_file.suffix == ".json":
                    agent_id = state_file.stem
                    try:
                        state_raw = await asyncio.to_thread(
                            state_file.read_text, "utf-8",
                        )
                        self._saved_states[agent_id] = json.loads(state_raw)
                    except Exception as exc:
                        logger.warning(
                            "Failed to load state for [%s]: %s",
                            agent_id, exc,
                        )

        logger.info(
            "Loaded %d entries from %s",
            len(self._entries), path,
        )

    # ── cleanup ───────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully stop all running agents."""
        running_ids = list(self._tasks.keys())
        for agent_id in running_ids:
            await self.stop(agent_id)
        logger.info(
            "AgentService shut down, stopped %d agents", len(running_ids),
        )

    def remove(self, agent_id: str) -> bool:
        """Remove a finished entry. Cannot remove running agents."""
        entry = self._entries.get(agent_id)
        if entry is None or entry.status == AgentStatus.RUNNING:
            return False
        self._entries.pop(agent_id, None)
        self._saved_states.pop(agent_id, None)
        return True
