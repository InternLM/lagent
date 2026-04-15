"""Claude Code SDK adapter — wraps claude-agent-sdk as a lagent Agent.

Uses the Python SDK instead of CLI subprocess, providing:
- Real multi-turn via session_id (no --continue hack)
- Structured message access (TextBlock, ThinkingBlock, ToolUseBlock)
- Runtime hooks (PreToolUse, PostToolUse, etc.)
- Full usage/cost tracking per turn

Usage::

    from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter

    agent = ClaudeCodeSDKAdapter(max_turns=5, timeout=120)
    r1 = await agent("Read main.py")
    r2 = await agent("Now fix the bug")  # real multi-turn, same session
    print(r2.content)

    # All messages captured structurally
    trace = agent.state_dict()['sdk_trace']
"""

import asyncio
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import AsyncExternalAgent


class ClaudeCodeSDKAdapter(AsyncExternalAgent):
    """Wraps claude-agent-sdk as a lagent Agent with real multi-turn.

    Each ``forward()`` call uses the same ``session_id``, so Claude Code
    maintains full conversation history internally. No Proxy needed for
    trace capture — the SDK yields structured messages directly.

    Args:
        max_turns: Max agent turns per call. Default: None (unlimited).
        permission_mode: Permission mode. Default: "default".
        model: Model name override.
        system_prompt: Custom system prompt.
        allowed_tools: List of allowed tool names.
        disallowed_tools: List of disallowed tool names.
        cwd: Working directory for Claude Code.
        effort: Reasoning effort level ("low", "medium", "high", "max").
        thinking: Thinking config dict. Default: adaptive.
        **kwargs: Passed to AsyncExternalAgent (name, timeout, proxy, hooks).
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        permission_mode: str = 'default',
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        effort: Optional[str] = None,
        thinking: Optional[dict] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'claude-code-sdk')
        kwargs.setdefault('description', 'Claude Code SDK agent')
        super().__init__(**kwargs)

        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.model = model
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.cwd = cwd or self.working_dir
        self.effort = effort
        self.thinking = thinking

        self._session_id: Optional[str] = None
        self._sdk_trace: List[dict] = []
        self._call_count = 0

    def setup(self) -> None:
        try:
            import claude_agent_sdk
        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk is required. "
                "Install with: pip install claude-agent-sdk"
            )

    async def run_external_async(self, task: str, **kwargs) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            UserMessage,
            query,
        )

        options = ClaudeAgentOptions(
            permission_mode=self.permission_mode,
            max_turns=self.max_turns,
        )

        if self.model:
            options.model = self.model
        if self.system_prompt:
            options.system_prompt = self.system_prompt
        if self.allowed_tools:
            options.allowed_tools = self.allowed_tools
        if self.disallowed_tools:
            options.disallowed_tools = self.disallowed_tools
        if self.cwd:
            options.cwd = self.cwd
        if self.effort:
            options.effort = self.effort
        if self.thinking:
            options.thinking = self.thinking

        # Multi-turn: resume session on subsequent calls
        if self._session_id:
            options.resume = self._session_id

        # Inject proxy env if present
        if self.proxy:
            session_key = f"sk-proxy-{self.session_id}"
            options.env = {
                'ANTHROPIC_BASE_URL': self.proxy.url,
                'ANTHROPIC_API_KEY': session_key,
            }

        # Collect messages
        messages = []
        result_text = ''
        result_msg = None

        try:
            async for message in query(prompt=task, options=options):
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'type': type(message).__name__,
                    'call_index': self._call_count,
                }

                if isinstance(message, AssistantMessage):
                    blocks = []
                    for block in message.content:
                        block_dict = asdict(block)
                        block_dict['block_type'] = type(block).__name__
                        blocks.append(block_dict)
                    record['content'] = blocks
                    record['model'] = message.model
                    record['usage'] = message.usage
                    record['stop_reason'] = message.stop_reason
                    record['message_id'] = message.message_id

                    # Extract text from content blocks
                    for block in message.content:
                        if hasattr(block, 'text'):
                            result_text = block.text

                    # Try to capture session_id from AssistantMessage
                    if hasattr(message, 'session_id') and message.session_id:
                        self._session_id = message.session_id

                elif isinstance(message, UserMessage):
                    if isinstance(message.content, str):
                        record['content'] = message.content
                    else:
                        record['content'] = [asdict(b) for b in message.content]

                elif isinstance(message, ResultMessage):
                    result_msg = message
                    record['result'] = message.result
                    record['session_id'] = message.session_id
                    record['usage'] = message.usage
                    record['total_cost_usd'] = message.total_cost_usd
                    record['num_turns'] = message.num_turns
                    record['is_error'] = message.is_error
                    record['stop_reason'] = message.stop_reason

                elif isinstance(message, SystemMessage):
                    record['subtype'] = message.subtype
                    record['data'] = message.data

                messages.append(record)
        except Exception as exc:
            # SDK may error after yielding some messages.
            # Log but don't lose what we already captured.
            import logging
            logging.getLogger(__name__).warning(
                f"SDK query error (captured {len(messages)} events): {exc}"
            )

        self._sdk_trace.extend(messages)

        # Capture session_id for multi-turn
        if result_msg and result_msg.session_id:
            self._session_id = result_msg.session_id

        # Fallback: try to get session_id from trace events
        if not self._session_id:
            for evt in reversed(messages):
                sid = evt.get('session_id')
                if sid:
                    self._session_id = sid
                    break

        self._call_count += 1

        # Return the final result
        if result_msg and result_msg.result:
            return result_msg.result
        return result_text or '(no output)'

    def state_dict(self, prefix='', destination=None) -> dict:
        dest = super().state_dict(prefix=prefix, destination=destination)
        dest[prefix + 'sdk_trace'] = list(self._sdk_trace)
        if self._session_id:
            dest[prefix + 'claude_session_id'] = self._session_id
        return dest

    def load_state_dict(self, state_dict: dict):
        filtered = {
            k: v for k, v in state_dict.items()
            if not k.endswith(('sdk_trace', 'claude_session_id'))
        }
        if not any(k.endswith('memory') for k in filtered):
            filtered['memory'] = []
        super().load_state_dict(filtered)

        # Restore session for multi-turn
        for k, v in state_dict.items():
            if k.endswith('claude_session_id'):
                self._session_id = v
            if k.endswith('sdk_trace'):
                self._sdk_trace = v or []
