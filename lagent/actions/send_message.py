"""SendMessageAction -- agent-facing toolkit for inter-agent messaging.

Wraps :class:`~lagent.services.mailbox.Mailbox` to expose message
sending as ``@tool_api`` methods.

Usage::

    mailbox = Mailbox()
    action = SendMessageAction(mailbox, agent_name="coder")
    executor = AsyncActionExecutor(actions=[action, ...])
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional, Type

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage
from lagent.services.mailbox import Mailbox

logger = logging.getLogger("lagent.actions.send_message")


class SendMessageAction(BaseAction):
    """Send and receive messages to/from other agents.

    The ``allowed_receivers`` parameter controls the communication
    topology:

    * ``None`` — fully distributed, can message any agent.
    * ``["lead"]`` — centralised, can only message the lead.
    * ``["lead", "tester"]`` — partial mesh.
    """

    def __init__(
        self,
        mailbox: Mailbox,
        agent_name: str,
        allowed_receivers: list[str] | None = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        super().__init__(description, parser)
        self._mailbox = mailbox
        self._agent_name = agent_name
        self._allowed = allowed_receivers

        # Ensure this agent is registered in the mailbox
        self._mailbox.register(agent_name)

    @tool_api
    def send(
        self,
        to: Annotated[
            str,
            "Target agent name, or '*' to broadcast to all agents",
        ],
        content: Annotated[
            str,
            "The message content to send",
        ],
    ) -> ActionReturn:
        """Send a message to another agent.

        Use this to ask questions, share findings, request help, or
        coordinate work with teammates.  The recipient will see the
        message in their next execution cycle.

        Args:
            to: Recipient agent name or '*' for broadcast.
            content: Message text.

        Returns:
            ActionReturn confirming delivery.
        """
        # Topology enforcement
        if (
            self._allowed is not None
            and to != "*"
            and to not in self._allowed
        ):
            return ActionReturn(
                type=self.name,
                errmsg=(
                    f"Cannot send to {to!r}. "
                    f"Allowed recipients: {self._allowed}"
                ),
                state=ActionStatusCode.API_ERROR,
            )

        try:
            msg = AgentMessage(
                sender=self._agent_name,
                receiver=to,
                content=content,
            )
            self._mailbox.send(msg)

            if to == "*":
                targets = [
                    n for n in self._mailbox.agents
                    if n != self._agent_name
                ]
                return ActionReturn(
                    type=self.name,
                    result=[dict(
                        type="text",
                        content=f"Broadcast sent to {len(targets)} agents.",
                    )],
                )

            return ActionReturn(
                type=self.name,
                result=[dict(
                    type="text",
                    content=f"Message sent to @{to}.",
                )],
            )
        except Exception as exc:
            return ActionReturn(
                type=self.name,
                errmsg=f"Failed to send message: {exc}",
                state=ActionStatusCode.API_ERROR,
            )

    @tool_api
    def check_inbox(self) -> ActionReturn:
        """Check for pending messages without consuming them.

        Use this to see if any teammate has sent you a message.
        Messages will be fully delivered through the environment
        context automatically; this tool is for a quick preview.

        Returns:
            ActionReturn listing pending messages or 'no messages'.
        """
        msgs = self._mailbox.peek(self._agent_name)
        if not msgs:
            return ActionReturn(
                type=self.name,
                result=[dict(type="text", content="No new messages.")],
            )
        lines = []
        for m in msgs:
            lines.append(f"- From @{m.sender}: {m.content}")
        header = f"{len(msgs)} pending message(s):\n"
        return ActionReturn(
            type=self.name,
            result=[dict(type="text", content=header + "\n".join(lines))],
        )


class AsyncSendMessageAction(AsyncActionMixin, SendMessageAction):
    """Async version of :class:`SendMessageAction`."""
    pass
