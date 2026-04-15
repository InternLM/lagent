"""TaskBoard -- structured task management with dependency tracking.

A shared resource object that agents use to organise work into trackable
tasks.  Inspired by Claude Code's production task system, adapted for
lagent's PyTorch-style architecture.

Design
------
* **Independent resource** -- not part of Agent; injected into Actions.
* **In-memory by default** -- optional JSON file persistence via *store_path*.
* **state_dict / load_state_dict** -- integrates with lagent's Module tree.
* **Dependency graph** -- bidirectional ``blocks`` / ``blocked_by`` with
  cascading cleanup on delete.
* **High water mark** -- task IDs are monotonically increasing and never
  reused, even after deletion.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("lagent.services.task")

_UNSET = object()
"""Sentinel to distinguish 'not provided' from 'set to None'."""


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── Status constants ─────────────────────────────────────────────────

TASK_STATUSES = ("pending", "in_progress", "completed")


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class Task:
    """A single task on the board."""

    id: str = ""
    subject: str = ""
    description: str = ""
    status: str = "pending"
    active_form: str | None = None
    owner: str | None = None
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    created_at_ms: int = field(default_factory=_now_ms)
    updated_at_ms: int = field(default_factory=_now_ms)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(
            id=data.get("id", ""),
            subject=data.get("subject", ""),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            active_form=data.get("active_form"),
            owner=data.get("owner"),
            blocks=list(data.get("blocks") or []),
            blocked_by=list(data.get("blocked_by") or []),
            metadata=data.get("metadata"),
            created_at_ms=data.get("created_at_ms", _now_ms()),
            updated_at_ms=data.get("updated_at_ms", _now_ms()),
        )


@dataclass
class ClaimResult:
    """Result of a :meth:`TaskBoard.claim` attempt."""

    success: bool
    task: Task | None = None
    reason: str = ""
    blocked_by_tasks: list[str] | None = None
    busy_with_tasks: list[str] | None = None


# ── Persistence helpers ──────────────────────────────────────────────

def _load_store(path: Path) -> tuple[list[Task], int]:
    if not path.exists():
        return [], 1
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            return [], 1
        tasks = [
            Task.from_dict(t)
            for t in data.get("tasks", [])
            if isinstance(t, dict)
        ]
        next_id = data.get("next_id", 1)
        return tasks, next_id
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load task store %s: %s", path, exc)
        return [], 1


def _save_store(tasks: list[Task], next_id: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "next_id": next_id,
        "tasks": [t.to_dict() for t in tasks],
    }
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to save task store: %s", exc)
        tmp.unlink(missing_ok=True)


# ── TaskBoard ────────────────────────────────────────────────────────

class TaskBoard:
    """In-memory task board with optional JSON persistence.

    Parameters
    ----------
    store_path : Path or None
        Where to persist ``tasks.json``.  ``None`` means purely
        in-memory (no file I/O).
    """

    def __init__(self, store_path: Path | None = None):
        self._store_path = store_path
        self._tasks: list[Task] = []
        self._next_id: int = 1

        if store_path is not None:
            loaded, hwm = _load_store(store_path)
            if loaded:
                self._tasks = loaded
                self._next_id = max(hwm, self._next_id)

    # ── helpers ───────────────────────────────────────────────────

    def _find(self, task_id: str) -> Task | None:
        return next((t for t in self._tasks if t.id == task_id), None)

    def _persist(self) -> None:
        if self._store_path is not None:
            _save_store(self._tasks, self._next_id, self._store_path)

    def _ensure_next_id(self) -> None:
        """Safety: make sure _next_id is past any existing task ID."""
        if self._tasks:
            max_existing = max(int(t.id) for t in self._tasks)
            self._next_id = max(self._next_id, max_existing + 1)

    def _add_block(self, from_id: str, to_id: str) -> None:
        """Bidirectional: from_task.blocks += to_id, to_task.blocked_by += from_id."""
        from_task = self._find(from_id)
        to_task = self._find(to_id)
        if from_task is not None and to_id not in from_task.blocks:
            from_task.blocks.append(to_id)
        if to_task is not None and from_id not in to_task.blocked_by:
            to_task.blocked_by.append(from_id)

    # ── CRUD ──────────────────────────────────────────────────────

    def create(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        blocked_by: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task.  Returns the created :class:`Task`."""
        task_id = str(self._next_id)
        self._next_id += 1
        now = _now_ms()

        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            status="pending",
            blocked_by=list(blocked_by or []),
            metadata=metadata,
            created_at_ms=now,
            updated_at_ms=now,
        )

        self._tasks.append(task)

        # Bidirectional: update blockers' "blocks" lists
        for dep_id in task.blocked_by:
            dep = self._find(dep_id)
            if dep is not None and task_id not in dep.blocks:
                dep.blocks.append(task_id)

        self._persist()
        return task

    def update(
        self,
        task_id: str,
        status=_UNSET,
        subject=_UNSET,
        description=_UNSET,
        active_form=_UNSET,
        owner=_UNSET,
        metadata=_UNSET,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> Task | None:
        """Update a task.  Returns the updated :class:`Task`, or None if
        not found.

        Pass ``status="deleted"`` to permanently remove the task
        (equivalent to :meth:`delete`).
        """
        # Handle "deleted" status as hard delete
        if status is not _UNSET and status == "deleted":
            self.delete(task_id)
            return None

        task = self._find(task_id)
        if task is None:
            return None

        if status is not _UNSET:
            task.status = status
        if subject is not _UNSET:
            task.subject = subject
        if description is not _UNSET:
            task.description = description
        if active_form is not _UNSET:
            task.active_form = active_form
        if owner is not _UNSET:
            task.owner = owner
        if metadata is not _UNSET and metadata is not None:
            if task.metadata is None:
                task.metadata = {}
            for k, v in metadata.items():
                if v is None:
                    task.metadata.pop(k, None)
                else:
                    task.metadata[k] = v

        # Dependency additions
        if add_blocks:
            for to_id in add_blocks:
                self._add_block(task_id, to_id)
        if add_blocked_by:
            for from_id in add_blocked_by:
                self._add_block(from_id, task_id)

        task.updated_at_ms = _now_ms()
        self._persist()
        return task

    def delete(self, task_id: str) -> bool:
        """Permanently remove a task and cascade-clean all references."""
        task = self._find(task_id)
        if task is None:
            return False

        # Cascading cleanup
        for other in self._tasks:
            if task_id in other.blocks:
                other.blocks.remove(task_id)
            if task_id in other.blocked_by:
                other.blocked_by.remove(task_id)

        self._tasks = [t for t in self._tasks if t.id != task_id]
        # Do NOT decrement _next_id -- high water mark
        self._persist()
        return True

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._find(task_id)

    def list(self, status: str | None = None) -> list[Task]:
        """List tasks, optionally filtered by status."""
        if status is None:
            return list(self._tasks)
        return [t for t in self._tasks if t.status == status]

    # ── Multi-agent coordination ──────────────────────────────────

    def claim(
        self,
        task_id: str,
        agent_name: str,
        check_busy: bool = True,
    ) -> ClaimResult:
        """Attempt to claim a task for an agent.

        Checks
        ------
        1. Task exists.
        2. Task not already completed.
        3. Task not already claimed by a *different* agent.
        4. All ``blocked_by`` tasks are completed.
        5. (optional) Agent has no other ``in_progress`` tasks.
        """
        task = self._find(task_id)
        if task is None:
            return ClaimResult(False, reason="task_not_found")

        if task.status == "completed":
            return ClaimResult(False, task=task, reason="already_completed")

        if task.owner is not None and task.owner != agent_name:
            return ClaimResult(
                False, task=task,
                reason=f"already_claimed by {task.owner}",
            )

        # Check unresolved blockers
        unresolved = [
            bid for bid in task.blocked_by
            if (b := self._find(bid)) is not None
            and b.status != "completed"
        ]
        if unresolved:
            return ClaimResult(
                False, task=task, reason="blocked",
                blocked_by_tasks=unresolved,
            )

        # Check if agent is already busy
        if check_busy:
            busy = [
                t.id for t in self._tasks
                if t.owner == agent_name
                and t.status == "in_progress"
                and t.id != task_id
            ]
            if busy:
                return ClaimResult(
                    False, task=task, reason="agent_busy",
                    busy_with_tasks=busy,
                )

        task.owner = agent_name
        task.status = "in_progress"
        task.updated_at_ms = _now_ms()
        self._persist()
        return ClaimResult(True, task=task)

    def release_agent(self, agent_name: str) -> list[Task]:
        """Release all non-completed tasks owned by *agent_name*.

        Resets ``owner`` to None and ``status`` to ``"pending"``.
        Used when an agent shuts down or fails.
        """
        released = []
        for task in self._tasks:
            if task.owner == agent_name and task.status != "completed":
                task.owner = None
                task.status = "pending"
                task.updated_at_ms = _now_ms()
                released.append(task)
        if released:
            self._persist()
        return released

    # ── Query helpers ─────────────────────────────────────────────

    def all_completed(self) -> bool:
        """True if every task on the board is completed (or board is empty)."""
        return len(self._tasks) == 0 or all(
            t.status == "completed" for t in self._tasks
        )

    def list_available(self) -> list[Task]:
        """List tasks that can be claimed: pending, no unresolved blockers,
        no owner."""
        available = []
        for task in self._tasks:
            if task.status != "pending" or task.owner is not None:
                continue
            unresolved = [
                bid for bid in task.blocked_by
                if (b := self._find(bid)) is not None
                and b.status != "completed"
            ]
            if not unresolved:
                available.append(task)
        return available

    def get_summary(self) -> str:
        """One-line-per-task summary, suitable for prompt injection.

        Format::

            Summary: 2 pending, 1 in_progress, 3 completed

            #1. [completed] Audit existing code
            #2. [in_progress] Design JWT schema  @coder
            #3. [pending] Implement JWT  ▶ blocked by #2
        """
        if not self._tasks:
            return "No tasks."

        counts: dict[str, int] = {}
        for t in self._tasks:
            counts[t.status] = counts.get(t.status, 0) + 1

        header = "Summary: " + ", ".join(
            f"{v} {k}" for k, v in counts.items()
        )

        completed_ids = {
            t.id for t in self._tasks if t.status == "completed"
        }
        lines = []
        for t in self._tasks:
            # Status icon
            icon = {"completed": "completed", "in_progress": "in_progress",
                    "pending": "pending"}.get(t.status, t.status)
            line = f"#{t.id}. [{icon}] {t.subject}"

            # Owner annotation
            if t.owner:
                line += f"  @{t.owner}"

            # Active blocker annotation (filter out completed)
            active_blockers = [
                bid for bid in t.blocked_by if bid not in completed_ids
            ]
            if active_blockers:
                refs = ", ".join(f"#{bid}" for bid in active_blockers)
                line += f"  ▶ blocked by {refs}"

            lines.append(line)

        return header + "\n\n" + "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks],
        }

    def load_state_dict(self, state: dict) -> None:
        self._tasks = [
            Task.from_dict(t)
            for t in state.get("tasks", [])
            if isinstance(t, dict)
        ]
        self._next_id = state.get("next_id", 1)
        self._ensure_next_id()
        self._persist()
