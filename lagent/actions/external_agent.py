"""ExternalAgentAction — expose external agent adapters as @tool_api.

Follows the same pattern as :class:`AsyncAgentAction` in
``lagent/actions/subagent.py``: a short-lived Action that wraps
a set of registered external agent adapters, allowing a PolicyAgent
to delegate tasks to external frameworks via tool calls.

Usage::

    from lagent.adapters.cli_adapter import CLIAgentAdapter

    claude = CLIAgentAdapter(
        name="claude-code",
        command_template="claude -p '{task}' --output-format text",
    )
    action = ExternalAgentAction(adapters={"claude-code": claude})
    executor = AsyncActionExecutor(actions=[action, ...])
"""

from typing import Annotated, Any, Dict, Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class ExternalAgentAction(AsyncActionMixin, BaseAction):
    """Toolkit that lets a PolicyAgent delegate tasks to external agents.

    The main agent can call external framework agents (CLI or SDK-based)
    through this toolkit. Each external agent is registered by name.

    Args:
        adapters: Dict mapping adapter names to adapter instances.
        description: Optional description override.
        parser: Parser class. Default: JsonParser.
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, Any]] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self._adapters: Dict[str, Any] = adapters or {}

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register an external agent adapter by name."""
        self._adapters[name] = adapter

    @tool_api
    async def run_agent(
        self,
        agent_name: Annotated[
            str,
            "Name of the external agent to invoke. Must be one of the "
            "registered adapter names (use list_agents to see available).",
        ],
        task: Annotated[
            str,
            "A clear, self-contained description of the task. Include "
            "all necessary context — the external agent has no access "
            "to the current conversation.",
        ],
    ) -> ActionReturn:
        """Delegate a task to an external agent framework.

        Runs the specified external agent with the given task and
        returns its output.

        Args:
            agent_name: Registered name of the external agent adapter.
            task: Self-contained task description.

        Returns:
            ActionReturn with the external agent's result.
        """
        adapter = self._adapters.get(agent_name)
        if adapter is None:
            available = list(self._adapters.keys())
            return ActionReturn(
                type=self.name,
                errmsg=f"Unknown agent: {agent_name!r}. Available: {available}",
                state=ActionStatusCode.API_ERROR,
            )

        try:
            # Go through Agent.__call__() so hooks and memory work
            response = await adapter(task)
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"External agent '{agent_name}' failed: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

        return ActionReturn(
            type=self.name,
            result=[dict(
                type='text',
                content=(
                    f"**External agent `{agent_name}` completed.**\n\n"
                    f"{response.content}"
                ),
            )],
        )

    @tool_api
    async def list_agents(self) -> ActionReturn:
        """List all available external agent adapters and their descriptions.

        Returns:
            ActionReturn with a summary of available external agents.
        """
        if not self._adapters:
            return ActionReturn(
                type=self.name,
                result=[dict(type='text', content='No external agents registered.')],
            )
        lines = []
        for name, adapter in self._adapters.items():
            desc = getattr(adapter, 'description', None) or '(no description)'
            lines.append(f"- **{name}**: {desc}")
        return ActionReturn(
            type=self.name,
            result=[dict(type='text', content='\n'.join(lines))],
        )
