"""TaskAction -- agent-facing toolkit for managing a task board.

Wraps :class:`~lagent.services.task.TaskBoard` to expose task CRUD as
``@tool_api`` methods.  Follows the same pattern as
:class:`~lagent.actions.cron.CronAction`.

Usage::

    board = TaskBoard()
    task_action = TaskAction(board)
    executor = AsyncActionExecutor(actions=[task_action, ...])
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.services.task import TaskBoard

logger = logging.getLogger("lagent.actions.task")


def _parse_id_list(value: str | None) -> list[str]:
    """Parse a comma-separated string of task IDs into a list."""
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def _parse_metadata(value: str | None) -> dict | None:
    """Parse a JSON string into a dict, or return None."""
    if not value:
        return None
    try:
        result = json.loads(value)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _format_task_detail(task) -> str:
    """Format a Task into a detailed multi-line string."""
    lines = [
        f"Task #{task.id}",
        f"  Subject: {task.subject}",
        f"  Status: {task.status}",
        f"  Description: {task.description}",
    ]
    if task.active_form:
        lines.append(f"  Active Form: {task.active_form}")
    if task.owner:
        lines.append(f"  Owner: {task.owner}")
    if task.blocks:
        refs = ", ".join(f"#{bid}" for bid in task.blocks)
        lines.append(f"  Blocks: {refs}")
    if task.blocked_by:
        refs = ", ".join(f"#{bid}" for bid in task.blocked_by)
        lines.append(f"  Blocked By: {refs}")
    if task.metadata:
        lines.append(f"  Metadata: {json.dumps(task.metadata, ensure_ascii=False)}")
    return "\n".join(lines)


class TaskAction(BaseAction):
    """Manage a structured task board with dependency tracking.

    Use this toolkit to create, update, list, and inspect tasks.
    Tasks support status tracking (pending / in_progress / completed),
    ownership, and dependency relationships (blocks / blocked_by).
    """

    def __init__(
        self,
        task_board: TaskBoard,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self._board = task_board

    @tool_api
    def create(
        self,
        subject: Annotated[
            str,
            "A brief, actionable title in imperative form "
            "(e.g. 'Fix authentication bug in login flow')",
        ],
        description: Annotated[
            str,
            "Full details of what needs to be done",
        ],
        active_form: Annotated[
            Optional[str],
            "Present continuous form shown when in_progress "
            "(e.g. 'Fixing authentication bug'). "
            "If omitted, the subject is used instead.",
        ] = None,
        blocked_by: Annotated[
            Optional[str],
            "Comma-separated task IDs that must complete before this "
            "task can start (e.g. '1,3')",
        ] = None,
        metadata: Annotated[
            Optional[str],
            "JSON string of arbitrary key-value metadata",
        ] = None,
    ) -> ActionReturn:
        """Create a new task on the board.

        Use this tool to break down complex work into trackable steps.
        Create tasks when work requires 3 or more distinct steps, when
        the user provides multiple things to do, or after receiving new
        instructions that should be captured immediately.

        Do NOT create tasks for single trivial actions or purely
        conversational/informational exchanges.

        All tasks start with status ``pending``. After creating tasks,
        use ``update`` to set dependencies if needed, and check
        ``list`` first to avoid duplicates.

        Args:
            subject: Brief imperative title.
            description: Full task details.
            active_form: Present continuous label for progress display.
            blocked_by: Comma-separated prerequisite task IDs.
            metadata: JSON metadata string.

        Returns:
            ActionReturn with the created task summary.
        """
        try:
            dep_ids = _parse_id_list(blocked_by)
            meta = _parse_metadata(metadata)

            task = self._board.create(
                subject=subject,
                description=description,
                active_form=active_form,
                blocked_by=dep_ids,
                metadata=meta,
            )

            content = f"Created task #{task.id}: {task.subject}"
            if dep_ids:
                refs = ", ".join(f"#{d}" for d in dep_ids)
                content += f" (blocked by {refs})"

            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content=content)],
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to create task: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

    @tool_api
    def update(
        self,
        task_id: Annotated[str, "The ID of the task to update"],
        status: Annotated[
            Optional[str],
            "New status: 'pending', 'in_progress', 'completed', or "
            "'deleted' (permanently removes the task)",
        ] = None,
        subject: Annotated[
            Optional[str], "New subject for the task"
        ] = None,
        description: Annotated[
            Optional[str], "New description for the task"
        ] = None,
        active_form: Annotated[
            Optional[str],
            "Present continuous form for progress display "
            "(e.g. 'Running tests')",
        ] = None,
        owner: Annotated[
            Optional[str], "New owner (agent name)"
        ] = None,
        metadata: Annotated[
            Optional[str],
            "JSON string of metadata keys to merge "
            "(set a key to null to delete it)",
        ] = None,
        add_blocks: Annotated[
            Optional[str],
            "Comma-separated task IDs that cannot start until this "
            "one completes",
        ] = None,
        add_blocked_by: Annotated[
            Optional[str],
            "Comma-separated task IDs that must complete before this "
            "one can start",
        ] = None,
    ) -> ActionReturn:
        """Update a task's status, details, or dependencies.

        Mark ``in_progress`` BEFORE beginning work on a task, not after.
        Only mark ``completed`` when work is FULLY done -- tests pass,
        no partial implementation, no unresolved errors.

        If blocked, create a new task describing what needs to be
        resolved rather than leaving the current task stuck.

        After completing a task, call ``list`` to find the next
        available work.

        Use ``status='deleted'`` to permanently remove a task.

        Args:
            task_id: ID of the task to update.
            status: New status value.
            subject: New subject.
            description: New description.
            active_form: New progress display label.
            owner: New owner.
            metadata: JSON metadata to merge.
            add_blocks: Task IDs this task blocks.
            add_blocked_by: Task IDs blocking this task.

        Returns:
            ActionReturn with the updated task summary.
        """
        try:
            kwargs: dict = {}
            if status is not None:
                kwargs["status"] = status
            if subject is not None:
                kwargs["subject"] = subject
            if description is not None:
                kwargs["description"] = description
            if active_form is not None:
                kwargs["active_form"] = active_form
            if owner is not None:
                kwargs["owner"] = owner

            meta = _parse_metadata(metadata)
            if meta is not None:
                kwargs["metadata"] = meta

            blocks_list = _parse_id_list(add_blocks)
            if blocks_list:
                kwargs["add_blocks"] = blocks_list

            blocked_by_list = _parse_id_list(add_blocked_by)
            if blocked_by_list:
                kwargs["add_blocked_by"] = blocked_by_list

            if not kwargs:
                return ActionReturn(
                    type=self.name,
                    errmsg="No fields to update were provided.",
                    state=ActionStatusCode.ARGS_ERROR,
                )

            task = self._board.update(task_id, **kwargs)

            # status="deleted" returns None
            if task is None and status == "deleted":
                return ActionReturn(
                    type=self.name,
                    result=[dict(
                        type="text",
                        content=f"Task #{task_id} has been deleted.",
                    )],
                )

            if task is None:
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Task #{task_id} not found.",
                    state=ActionStatusCode.API_ERROR,
                )

            return ActionReturn(
                type=self.name,
                result=[dict(
                    type="text",
                    content=f"Updated task #{task.id}: [{task.status}] {task.subject}",
                )],
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to update task: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

    @tool_api
    def get(
        self,
        task_id: Annotated[str, "The ID of the task to retrieve"],
    ) -> ActionReturn:
        """Retrieve full details of a specific task.

        Use this before starting work on a task to understand the
        complete requirements.  Also useful for checking dependency
        relationships (what it blocks, what blocks it).

        Read the task's latest state before updating it to avoid
        stale overwrites.

        Args:
            task_id: The task ID.

        Returns:
            ActionReturn with the task's full details including
            dependencies.
        """
        task = self._board.get(task_id)
        if task is None:
            return ActionReturn(
                type=self.name,
                errmsg=f"Task #{task_id} not found.",
                state=ActionStatusCode.API_ERROR,
            )
        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content=_format_task_detail(task))],
        )

    @tool_api
    def list(
        self,
        status: Annotated[
            Optional[str],
            "Filter by status: 'pending', 'in_progress', or "
            "'completed'. Omit to list all tasks.",
        ] = None,
    ) -> ActionReturn:
        """List all tasks with a status summary.

        Use this to see what tasks are available (pending, not blocked,
        no owner), check overall progress, or find newly unblocked
        work after completing a task.

        Prefer working on tasks in ID order (lowest first) when
        multiple tasks are available, as earlier tasks often set up
        context for later ones.

        Args:
            status: Optional status filter.

        Returns:
            ActionReturn with a summary of tasks.
        """
        summary = self._board.get_summary()
        if status is not None:
            tasks = self._board.list(status=status)
            if not tasks:
                return ActionReturn(
                    type=self.name,
                    result=[dict(
                        type="text",
                        content=f"No tasks with status '{status}'.",
                    )],
                )
            # Rebuild summary with filter
            completed_ids = {
                t.id for t in self._board.list(status="completed")
            }
            lines = []
            for t in tasks:
                line = f"#{t.id}. [{t.status}] {t.subject}"
                if t.owner:
                    line += f"  @{t.owner}"
                active_blockers = [
                    bid for bid in t.blocked_by
                    if bid not in completed_ids
                ]
                if active_blockers:
                    refs = ", ".join(f"#{bid}" for bid in active_blockers)
                    line += f"  ▶ blocked by {refs}"
                lines.append(line)
            content = f"Tasks ({status}): {len(tasks)}\n\n" + "\n".join(lines)
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content=content)],
            )

        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content=summary)],
        )


class AsyncTaskAction(AsyncActionMixin, TaskAction):
    """Async version of :class:`TaskAction`."""
    pass
