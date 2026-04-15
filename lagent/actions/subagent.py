"""AgentAction — agent-facing toolkit for managing sub-agents.

Short-lived Action created per-request by the Dispatcher.  Wraps the
long-lived :class:`AgentService` singleton, analogous to how
:class:`CronAction` wraps :class:`CronService`.

The agent can spawn, list, query, and resume sub-agents through this
toolkit.  Each ``@tool_api`` method maps to an ``AgentService`` operation.

Usage::

    agent_action = AgentAction(
        agent_service=app.agent_service,
    )
    executor = AsyncActionExecutor(actions=[agent_action, ...])
"""

from typing import Annotated, Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

from ..services.agent import AgentService, AgentStatus


class AsyncAgentAction(AsyncActionMixin, BaseAction):
    """Manage sub-agents: spawn, list, query, and resume.

    The main agent can delegate tasks to specialised sub-agents through
    this toolkit.  Each sub-agent runs independently with its own tools
    and context, and returns a result when finished.

    Parameters
    ----------
    agent_service : AgentService
        The service managing agent lifecycle.
    parent_actions : list, optional
        Action instances available to the parent agent.  When a
        dynamic sub-agent is spawned, it inherits (a subset of) these.
        If ``None``, dynamic agents get no tools.
    """

    def __init__(
        self,
        agent_service: AgentService,
        parent_actions: list | None = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self._service = agent_service
        self._parent_actions = parent_actions or []

    # ── tool APIs ─────────────────────────────────────────────────

    @tool_api
    async def spawn(
        self,
        task: Annotated[
            str,
            "A clear, self-contained description of the task for the "
            "sub-agent.  Include all necessary context — the sub-agent "
            "has no access to the current conversation history.",
        ],
        agent_type: Annotated[
            Optional[str],
            "The type of agent to spawn.  Must be one of the available "
            "agent types.  Omit to create a dynamic agent with "
            "system_prompt.",
        ] = None,
        system_prompt: Annotated[
            Optional[str],
            "System prompt for a dynamically created agent.  Only used "
            "when agent_type is not specified.",
        ] = None,
        tools: Annotated[
            Optional[str],
            "Comma-separated tool names to give the sub-agent "
            "(e.g. 'Shell,Read,Grep').  Only used for dynamic agents. "
            "Omit to inherit the parent's full tool set.",
        ] = None,
        label: Annotated[
            str,
            "A short human-readable label for the task.  "
            "Defaults to the first 40 characters of the task.",
        ] = "",
        mode: Annotated[
            str,
            "Execution mode: 'sync' blocks until done (default), "
            "'async' runs in the background.",
        ] = "sync",
    ) -> ActionReturn:
        """Spawn a sub-agent to execute a task.

        Two ways to create the sub-agent:

        1. Provide ``agent_type`` to use a pre-registered agent template
           (with its own prompt, tools, and configuration).
        2. Omit ``agent_type`` and optionally provide ``system_prompt``
           and ``tools`` to dynamically create a lightweight agent.

        Args:
            task: Self-contained task description.
            agent_type: Type of pre-registered agent to spawn.
            system_prompt: Custom system prompt for dynamic agents.
            tools: Comma-separated tool name whitelist for dynamic agents.
            label: Optional short label.
            mode: 'sync' (default) or 'async'.

        Returns:
            ActionReturn with the result or acknowledgement.
        """
        # Resolve tool instances from names
        resolved_tools = None
        if tools and not agent_type:
            allowed = {s.strip() for s in tools.split(",") if s.strip()}
            resolved_tools = [
                a for a in self._parent_actions
                if type(a).__name__ in allowed
            ]

        # Build kwargs for spec.acreate()
        spec_kwargs: dict = {}
        if self._default_llm is not None:
            spec_kwargs["llm"] = self._default_llm
        if resolved_tools is not None:
            spec_kwargs["actions"] = resolved_tools
        elif self._parent_actions:
            spec_kwargs["actions"] = list(self._parent_actions)
        if system_prompt:
            spec_kwargs["system_prompt"] = system_prompt

        try:
            entry = await self._service.spawn(
                task=task,
                agent_type=agent_type or "default",
                label=label,
                mode=mode,
                **spec_kwargs,
            )
        except ValueError as exc:
            return ActionReturn(
                type=self.name,
                errmsg=str(exc),
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to spawn agent: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

        if mode == "sync":
            if entry.status == AgentStatus.FAILED:
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Agent failed: {entry.error}",
                    state=ActionStatusCode.API_ERROR,
                )
            return ActionReturn(
                type=self.name,
                result=[dict(
                    type="text",
                    content=(
                        f"**Agent `{entry.agent_type}` (id: `{entry.id}`) completed.**\n\n"
                        f"{entry.result or '(no output)'}"
                    ),
                )],
            )
        else:
            return ActionReturn(
                type=self.name,
                result=[dict(
                    type="text",
                    content=(
                        f"✅ Agent spawned in background:\n"
                        f"  - **type**: `{entry.agent_type}`\n"
                        f"  - **id**: `{entry.id}`\n"
                        f"  - **label**: {entry.label}\n"
                        f"  - **status**: {entry.status}\n\n"
                        f"Use `list_agents` to check progress, or "
                        f"`query_agent` with the ID to get details."
                    ),
                )],
            )

    @tool_api
    async def list_agents(
        self,
        status: Annotated[
            Optional[str],
            "Filter by status: 'running', 'stopped', 'failed', or 'pending'. "
            "Omit to list all.",
        ] = None,
    ) -> ActionReturn:
        """List all managed sub-agents and their status.

        Args:
            status: Optional status filter.

        Returns:
            ActionReturn with a summary of all agents.
        """
        entries = self._service.list(status=status)
        if not entries:
            msg = "No sub-agents"
            if status:
                msg += f" with status '{status}'"
            msg += " found."
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content=msg)],
            )

        lines = []
        for e in entries:
            status_icon = {
                AgentStatus.PENDING: "⏳",
                AgentStatus.RUNNING: "🔄",
                AgentStatus.STOPPED: "✅",
                AgentStatus.FAILED: "❌",
            }.get(e.status, "❓")
            lines.append(
                f"- {status_icon} **{e.label}** (`{e.id}`) — "
                f"type: `{e.agent_type}`, status: {e.status}"
            )
        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content="\n".join(lines))],
        )

    @tool_api
    async def query_agent(
        self,
        agent_id: Annotated[str, "The ID of the agent to query"],
    ) -> ActionReturn:
        """Get detailed information about a specific sub-agent.

        Args:
            agent_id: The agent's ID.

        Returns:
            ActionReturn with the agent's full details.
        """
        entry = self._service.get(agent_id)
        if entry is None:
            return ActionReturn(
                type=self.name,
                errmsg=f"Agent `{agent_id}` not found.",
                state=ActionStatusCode.API_ERROR,
            )

        info = (
            f"**Agent `{entry.id}`**\n"
            f"- **type**: `{entry.agent_type}`\n"
            f"- **label**: {entry.label}\n"
            f"- **status**: {entry.status}\n"
            f"- **task**: {entry.task}\n"
        )
        if entry.result:
            info += f"\n**Result:**\n{entry.result}"
        if entry.error:
            info += f"\n**Error:** {entry.error}"

        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content=info)],
        )

    @tool_api
    async def resume_agent(
        self,
        agent_id: Annotated[str, "The ID of the stopped agent to resume"],
        message: Annotated[
            str,
            "The new message to send to the agent.  The agent will "
            "resume with its full previous context plus this message.",
        ],
    ) -> ActionReturn:
        """Resume a stopped sub-agent with a new message.

        The agent picks up where it left off, retaining its previous
        conversation history.

        Args:
            agent_id: ID of the stopped agent.
            message: New message to send.

        Returns:
            ActionReturn with the agent's new result.
        """
        try:
            entry = await self._service.resume(agent_id, message)
        except ValueError as exc:
            return ActionReturn(
                type=self.name,
                errmsg=str(exc),
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to resume agent: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

        if entry.status == AgentStatus.FAILED:
            return ActionReturn(
                type=self.name,
                errmsg=f"Agent failed after resume: {entry.error}",
                state=ActionStatusCode.API_ERROR,
            )

        return ActionReturn(
            type=self.name,
            result=[dict(
                type="text",
                content=(
                    f"**Agent `{entry.id}` resumed and completed.**\n\n"
                    f"{entry.result or '(no output)'}"
                ),
            )],
        )

    @tool_api
    async def stop_agent(
        self,
        agent_id: Annotated[str, "The ID of the running agent to stop"],
    ) -> ActionReturn:
        """Stop a running background sub-agent.

        Args:
            agent_id: ID of the agent to stop.

        Returns:
            ActionReturn confirming the stop.
        """
        stopped = await self._service.stop(agent_id)
        if stopped:
            return ActionReturn(
                type=self.name,
                result=[dict(
                    type="text",
                    content=f"✅ Agent `{agent_id}` stopped.",
                )],
            )
        return ActionReturn(
            type=self.name,
            errmsg=f"Agent `{agent_id}` not found or not running.",
            state=ActionStatusCode.API_ERROR,
        )
