"""Simple test agent — no LLM, no network.

A minimal AsyncAgent that echoes back the input.
Used by unit tests that don't need real LLM calls.
"""

from lagent.agents.agent import AsyncAgent
from lagent.schema import AgentMessage


class EchoAgent(AsyncAgent):
    """Agent that echoes input — no LLM needed."""

    async def forward(self, *message, **kwargs):
        text = " ".join(
            m.content if isinstance(m, AgentMessage) else str(m)
            for m in message
        )
        return AgentMessage(
            sender=self.name,
            content=f"echo: {text}",
        )


name = "simple-agent"
description = "Echo agent for unit tests (no LLM)"
background = False

agent_config = dict(
    type=EchoAgent,
    name="simple-agent",
)
