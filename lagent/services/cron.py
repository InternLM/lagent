"""Cron scheduling service.

A self-contained timer-based scheduler that stores jobs as JSON and fires
callbacks when jobs come due.  Completely decoupled from agent / bus — the
caller decides what happens when a job fires via the ``on_job`` callback.

Design heavily inspired by the python-cron-demo reference implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger("lagent.interclaw.services.cron")


# ── helpers ───────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


# ── data model ────────────────────────────────────────────────────────

class ScheduleKind(str, Enum):
    AT = "at"
    EVERY = "every"
    CRON = "cron"


@dataclass
class Schedule:
    """Unified schedule descriptor.

    Exactly one of ``at``, ``every_seconds``, or ``expr`` should be set.
    """
    kind: Literal["at", "every", "cron"] = "at"
    at: str | None = None           # ISO-8601 datetime for one-shot
    every_seconds: float | None = None  # interval in seconds
    expr: str | None = None         # 5-field cron expression
    tz: str | None = None           # IANA timezone (cron only)


@dataclass
class JobState:
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: str | None = None
    last_error: str | None = None
    consecutive_errors: int = 0


@dataclass
class CronJob:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    enabled: bool = True
    schedule: Schedule = field(default_factory=Schedule)
    payload: dict[str, Any] = field(default_factory=dict)
    state: JobState = field(default_factory=JobState)
    created_at_ms: int = field(default_factory=_now_ms)
    updated_at_ms: int = field(default_factory=_now_ms)
    delete_after_run: bool = False

    # ── serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CronJob:
        sched_data = data.get("schedule", {})
        schedule = Schedule(
            **{k: v for k, v in sched_data.items()
               if k in Schedule.__dataclass_fields__}
        )
        state_data = data.get("state", {})
        state = JobState(
            **{k: v for k, v in state_data.items()
               if k in JobState.__dataclass_fields__}
        )
        return cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            schedule=schedule,
            payload=data.get("payload", {}),
            state=state,
            created_at_ms=data.get("created_at_ms", _now_ms()),
            updated_at_ms=data.get("updated_at_ms", _now_ms()),
            delete_after_run=data.get("delete_after_run", False),
        )


# ── schedule computation ──────────────────────────────────────────────

def compute_next_run(schedule: Schedule, now_ms: int) -> int | None:
    """Return the next fire time in epoch-ms, or *None* if no more fires."""

    if schedule.kind == "at":
        if not schedule.at:
            return None
        try:
            dt = datetime.fromisoformat(schedule.at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            at_ms = int(dt.timestamp() * 1000)
        except (ValueError, OverflowError):
            return None
        return at_ms if at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_seconds or schedule.every_seconds <= 0:
            return None
        return now_ms + int(schedule.every_seconds * 1000)

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo
            from croniter import croniter
            tz = ZoneInfo(schedule.tz) if schedule.tz else timezone.utc
            base_dt = datetime.fromtimestamp(now_ms / 1000, tz=tz)
            nxt = croniter(schedule.expr, base_dt).get_next(datetime)
            return int(nxt.timestamp() * 1000)
        except Exception:
            return None

    return None


# ── persistence ───────────────────────────────────────────────────────

def _load_jobs(path: Path) -> list[CronJob]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            return []
        return [
            CronJob.from_dict(j)
            for j in data.get("jobs", [])
            if isinstance(j, dict)
        ]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load cron store %s: %s", path, exc)
        return []


def _save_jobs(jobs: list[CronJob], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "jobs": [j.to_dict() for j in jobs]}
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to save cron store: %s", exc)
        tmp.unlink(missing_ok=True)


# ── service ───────────────────────────────────────────────────────────

# Backoff schedule (seconds) for consecutive errors
_BACKOFF = [30, 60, 300, 900, 3600]
_MAX_TIMER_DELAY_S = 60.0
_MIN_REFIRE_GAP_S = 2.0


class CronService:
    """Async cron scheduler with JSON persistence.

    Parameters
    ----------
    store_path : Path
        Where to persist ``jobs.json``.
    on_job : callable, optional
        ``async (CronJob) -> None`` called when a job fires.  The caller
        decides what to do (e.g. publish an inbound event on the bus).
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Awaitable[None]] | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self._jobs: list[CronJob] = []
        self._timer: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._jobs = _load_jobs(self.store_path)
        self._recompute_all()
        self._persist()
        self._running = True
        self._arm_timer()
        logger.info(
            "Cron service started (%d jobs, %d enabled)",
            len(self._jobs),
            sum(1 for j in self._jobs if j.enabled),
        )

    def stop(self) -> None:
        self._running = False
        self._cancel_timer()
        logger.info("Cron service stopped")

    # ── public CRUD ───────────────────────────────────────────────

    def add_job(
        self,
        name: str,
        schedule: Schedule,
        message: str,
        channel: str | None = None,
        chat_id: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        # Validate schedule consistency
        if schedule.kind == "at" and not schedule.at:
            raise ValueError("Schedule kind 'at' requires 'at' (ISO-8601 datetime)")
        if schedule.kind == "every" and (
            not schedule.every_seconds or schedule.every_seconds <= 0
        ):
            raise ValueError("Schedule kind 'every' requires positive 'every_seconds'")
        if schedule.kind == "cron" and not schedule.expr:
            raise ValueError("Schedule kind 'cron' requires 'expr' (5-field cron expression)")

        now = _now_ms()
        job = CronJob(
            name=name,
            schedule=schedule,
            payload=dict(message=message, channel=channel, chat_id=chat_id),
            state=JobState(next_run_at_ms=compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )
        self._jobs.append(job)
        self._persist()
        self._arm_timer()
        logger.info("Cron: added job '%s' (%s)", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        before = len(self._jobs)
        self._jobs = [j for j in self._jobs if j.id != job_id]
        removed = len(self._jobs) < before
        if removed:
            self._persist()
            self._arm_timer()
            logger.info("Cron: removed job %s", job_id)
        return removed

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        jobs = (
            self._jobs
            if include_disabled
            else [j for j in self._jobs if j.enabled]
        )
        return sorted(
            jobs, key=lambda j: j.state.next_run_at_ms or float("inf")
        )

    def get_job(self, job_id: str) -> CronJob | None:
        return next((j for j in self._jobs if j.id == job_id), None)

    # ── timer engine ──────────────────────────────────────────────

    def _arm_timer(self) -> None:
        self._cancel_timer()
        if not self._running or not self._loop:
            return
        wake = self._next_wake_ms()
        if wake is None:
            return
        delay_s = max((wake - _now_ms()) / 1000, _MIN_REFIRE_GAP_S)
        delay_s = min(delay_s, _MAX_TIMER_DELAY_S)
        self._timer = self._loop.call_later(
            delay_s, lambda: asyncio.ensure_future(self._on_timer())
        )

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    async def _on_timer(self) -> None:
        if not self._running:
            return
        self._jobs = _load_jobs(self.store_path)  # pick up external edits
        now = _now_ms()
        due = [
            j for j in self._jobs
            if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
        ]
        for job in due:
            await self._execute_job(job)
        self._persist()
        self._arm_timer()

    async def _execute_job(self, job: CronJob) -> None:
        start = _now_ms()
        logger.info("Cron: executing '%s' (%s)", job.name, job.id)
        try:
            if self.on_job:
                await self.on_job(job)
            job.state.last_status = "ok"
            job.state.last_error = None
            job.state.consecutive_errors = 0
        except Exception as exc:
            job.state.last_status = "error"
            job.state.last_error = str(exc)
            job.state.consecutive_errors += 1
            logger.error("Cron: job '%s' failed: %s", job.name, exc)

        job.state.last_run_at_ms = start
        job.updated_at_ms = _now_ms()

        # reschedule or retire
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._jobs = [j for j in self._jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            if job.state.consecutive_errors > 0:
                idx = min(
                    job.state.consecutive_errors - 1, len(_BACKOFF) - 1
                )
                job.state.next_run_at_ms = _now_ms() + _BACKOFF[idx] * 1000
            else:
                job.state.next_run_at_ms = compute_next_run(
                    job.schedule, _now_ms()
                )

    # ── helpers ───────────────────────────────────────────────────

    def _next_wake_ms(self) -> int | None:
        times = [
            j.state.next_run_at_ms
            for j in self._jobs
            if j.enabled and j.state.next_run_at_ms
        ]
        return min(times) if times else None

    def _recompute_all(self) -> None:
        now = _now_ms()
        for job in self._jobs:
            if job.enabled:
                job.state.next_run_at_ms = compute_next_run(job.schedule, now)

    def _persist(self) -> None:
        _save_jobs(self._jobs, self.store_path)

    # ── serialisation ────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Export service state for Module tree integration."""
        return {"version": 1, "jobs": [j.to_dict() for j in self._jobs]}

    def load_state_dict(self, state: dict) -> None:
        """Restore service state from a previous :meth:`state_dict`."""
        self._jobs = [
            CronJob.from_dict(j)
            for j in state.get("jobs", [])
            if isinstance(j, dict)
        ]
        self._recompute_all()
        self._persist()
