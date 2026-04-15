"""CronAction — agent-facing toolkit for managing scheduled tasks.

This is a short-lived Action created per-request by the Dispatcher.
It wraps the long-lived :class:`CronService` singleton and carries the
request-scoped ``channel`` / ``chat_id`` so the agent itself stays
completely channel-unaware.

Usage::

    cron_action = CronAction(
        cron_service=app.cron_service,
        channel="feishu",
        chat_id="oc_xxxx",
    )
    executor = AsyncActionExecutor(actions=[cron_action, ...])
"""

from __future__ import annotations

from typing import Annotated, Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

from ..services.cron import CronService, Schedule


class CronAction(BaseAction):
    """Manage scheduled / recurring tasks.

    The agent can add, list, and remove cron jobs through this toolkit.
    Each job will fire at the scheduled time and deliver a message back
    to the originating channel and chat.
    """

    def __init__(
        self,
        cron_service: CronService,
        channel: str | None = None,
        chat_id: str | None = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self._cron = cron_service
        self._channel = channel
        self._chat_id = chat_id

    # ── tool APIs ─────────────────────────────────────────────────

    @tool_api
    def add(
        self,
        name: Annotated[str, "A short human-readable name for the job"],
        message: Annotated[str, "The message / task to deliver when the job fires"],
        schedule_kind: Annotated[
            str,
            "One of: 'at' (one-shot ISO-8601 datetime), "
            "'every' (interval in seconds), "
            "'cron' (5-field cron expression)",
        ] = "at",
        at: Annotated[
            Optional[str],
            "ISO-8601 datetime for one-shot schedule, e.g. '2025-01-15T09:00:00+08:00'",
        ] = None,
        every_seconds: Annotated[
            Optional[float],
            "Interval in seconds for recurring schedule",
        ] = None,
        cron_expr: Annotated[
            Optional[str],
            "5-field cron expression, e.g. '30 9 * * 1-5' for weekdays at 09:30",
        ] = None,
        timezone: Annotated[
            Optional[str],
            "IANA timezone for cron expression, e.g. 'Asia/Shanghai'",
        ] = None,
    ) -> ActionReturn:
        """Add a new scheduled task.

        Args:
            name: A short human-readable name for the job.
            message: The message / task to deliver when the job fires.
            schedule_kind: Schedule type — 'at', 'every', or 'cron'.
            at: ISO-8601 datetime for one-shot schedule.
            every_seconds: Interval in seconds for recurring schedule.
            cron_expr: 5-field cron expression for cron schedule.
            timezone: IANA timezone for cron expression.

        Returns:
            ActionReturn with the created job summary.
        """
        schedule = Schedule(
            kind=schedule_kind,
            at=at,
            every_seconds=every_seconds,
            expr=cron_expr,
            tz=timezone,
        )
        if schedule_kind not in ("at", "every", "cron"):
            return ActionReturn(
                type=self.name,
                errmsg=f"Invalid schedule_kind: {schedule_kind!r}. "
                       f"Must be 'at', 'every', or 'cron'.",
                state=ActionStatusCode.ARGS_ERROR,
            )
        try:
            job = self._cron.add_job(
                name=name,
                schedule=schedule,
                message=message,
                channel=self._channel,
                chat_id=self._chat_id,
                delete_after_run=(schedule_kind == "at"),
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to add cron job: {exc}",
                state=ActionStatusCode.API_ERROR,
            )
        return ActionReturn(
            type=self.name,
            result=[
                dict(
                    type="text",
                    content=(
                        f"✅ Job created: **{job.name}** (id: `{job.id}`)\n"
                        f"  Schedule: {schedule_kind}"
                        f"{f' at {at}' if at else ''}"
                        f"{f' every {every_seconds}s' if every_seconds else ''}"
                        f"{f' cron {cron_expr}' if cron_expr else ''}"
                    ),
                )
            ],
        )

    @tool_api
    def list(self) -> ActionReturn:
        """List all active scheduled tasks.

        Returns:
            ActionReturn with a summary of all active jobs.
        """
        jobs = self._cron.list_jobs(include_disabled=False)
        if not jobs:
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content="No active scheduled tasks.")],
            )
        lines = []
        for j in jobs:
            sched = j.schedule
            sched_desc = (
                f"at {sched.at}" if sched.kind == "at"
                else f"every {sched.every_seconds}s" if sched.kind == "every"
                else f"cron `{sched.expr}`"
            )
            lines.append(
                f"- **{j.name}** (`{j.id}`) — {sched_desc}\n"
                f"  message: {j.payload.get('message', '?')}"
            )
        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content="\n".join(lines))],
        )

    @tool_api
    def remove(
        self,
        job_id: Annotated[str, "The ID of the job to remove"],
    ) -> ActionReturn:
        """Remove a scheduled task by its ID.

        Args:
            job_id: The ID of the job to remove.

        Returns:
            ActionReturn confirming removal or reporting not found.
        """
        removed = self._cron.remove_job(job_id)
        if removed:
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content=f"✅ Job `{job_id}` removed.")],
            )
        return ActionReturn(
            type=self.name,
            errmsg=f"Job `{job_id}` not found.",
            state=ActionStatusCode.API_ERROR,
        )


class AsyncCronAction(AsyncActionMixin, CronAction):
    """Async version of :class:`CronAction`."""
    pass
