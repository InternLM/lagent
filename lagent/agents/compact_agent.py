"""CompactAgent — context compression as a first-class Agent.

A standard AsyncAgent with a compact-specific template (COMPACT_PROMPT).
The orchestrator passes the policy_message to ``should_compact()`` to
check, then calls the agent normally.  CompactAgent's own ContextBuilder
assembles the summary request.

Usage (in orchestrator)::

    if compact_agent.should_compact(policy_message):
        summary_msg = await compact_agent(policy_message)
        # inject summary into env_info for next turn
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from lagent.agents.agent import AsyncAgent
from lagent.schema import AgentMessage

logger = logging.getLogger("lagent.agents.compact_agent")


# ── Token estimation ──────────────────────────────────────────────

def estimate_token_count(messages: list, tools: Optional[list] = None) -> int:
    """Rough token count estimation (chars / 4).

    Good enough for threshold decisions — we don't need exact counts.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                total_chars += len(str(part))
        total_chars += len(msg.get("reasoning_content") or "")

    if tools:
        total_chars += len(json.dumps(tools, ensure_ascii=False))

    return total_chars // 4


# ── Compact prompt ────────────────────────────────────────────────

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. \
Tool calls will be REJECTED and will waste your only turn.

Your task is to create a detailed summary of the conversation so far. \
This summary will REPLACE the conversation history, so it must contain \
ALL information needed to continue working effectively.

Please provide a thorough summary covering these sections:

## 1. Primary Request and Intent
What is the user's overall goal? What specific outcomes do they want?

## 2. Key Technical Concepts
What technologies, frameworks, APIs, or patterns are involved?

## 3. Files and Code Sections
List ALL files that were read, created, or modified, with a brief note \
on what was done to each. Include key code snippets that would be \
needed to continue the work.

## 4. Errors and Fixes
Document any errors encountered and how they were (or weren't) resolved.

## 5. Problem-Solving Approach
What strategies were tried? What worked, what didn't?

## 6. All User Messages (Chronological)
Reproduce the ESSENCE of every user message in order. Do not skip any.

## 7. Pending Tasks
What remains to be done? What was the user's last request?

## 8. Current Work State
What is the current state of the work? What file is being edited? \
What was the last action taken?

## 9. Optional Next Step
If there's an obvious next action, state it briefly.

Be thorough and specific. The model reading this summary will have \
NO access to the original conversation.
"""


# ── CompactAgent ──────────────────────────────────────────────────

class AsyncCompactAgent(AsyncAgent):
    """Context compression Agent.  Peer of PolicyAgent / EnvAgent.

    Has its own aggregator/contextbuilder that assembles
    COMPACT_PROMPT + the input message into a summarisation request.

    ``should_compact(message)`` inspects the message to decide
    whether compaction is needed — the orchestrator doesn't need to
    know what fields to check.

    Parameters
    ----------
    llm : BaseLLM or dict
        LLM provider for summarisation (can share with policy).
    max_context_tokens : int
        Maximum context window size in tokens.
    threshold_ratio : float
        Trigger compact when usage exceeds this ratio of max tokens.
    """

    def __init__(
        self,
        max_context_tokens: int = 128_000,
        threshold_ratio: float = 0.85,
        **kwargs,
    ):
        # Default template to COMPACT_PROMPT if not provided
        kwargs.setdefault('template', COMPACT_PROMPT)
        # Default aggregator to CompactAggregator
        if 'aggregator' not in kwargs:
            from lagent.agents.aggregator.compact_aggregator import CompactAggregator
            kwargs['aggregator'] = CompactAggregator()
        super().__init__(**kwargs)
        self._max_context_tokens = max_context_tokens
        self._threshold_ratio = threshold_ratio
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3

    @property
    def threshold_tokens(self) -> int:
        return int(self._max_context_tokens * self._threshold_ratio)

    def should_compact(self, message: AgentMessage) -> bool:
        """Check whether compaction should be triggered.

        Inspects the message's extra_info for context_tokens,
        but subclasses can override with different strategies
        (e.g. message count, content length, etc.).

        Parameters
        ----------
        message : AgentMessage
            Typically the policy_message with extra_info.
        """
        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                "Compact circuit breaker open: %d consecutive failures",
                self._consecutive_failures,
            )
            return False
        context_tokens = (message.extra_info or {}).get('context_tokens', 0)
        return context_tokens > self.threshold_tokens

