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

import copy
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Literal, Optional

from lagent.schema import AgentMessage

from .base import AsyncExternalAgent


class ClaudeCodeParseError(RuntimeError):
    """The SDK returned a malformed terminal message."""


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
        tools: List of built-in tools to enable (None = SDK default, [] = disable all).
        allowed_tools: List of allowed tool names.
        disallowed_tools: List of disallowed tool names.
        mcp_servers: Dict of MCP server configs.
        cwd: Working directory for Claude Code.
        effort: Reasoning effort level ("low", "medium", "high", "max").
        thinking: Thinking config dict. Default: adaptive.
        parse_error_retries: Same-session continuation attempts after malformed
            SDK messages. Default: 0 (disabled).
        **kwargs: Passed to AsyncExternalAgent (name, timeout, proxy, hooks).
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        permission_mode: str = 'default',
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        mcp_servers: Optional[Dict[str, dict]] = None,
        cwd: Optional[str] = None,
        setting_sources: Optional[List[Literal["user", "project", "local"]]] = None,
        skills: Optional[List[str]] = None,
        effort: Optional[str] = None,
        thinking: Optional[dict] = None,
        parse_error_retries: int = 0,
        extra_options: Optional[dict] = None,
        **kwargs,
    ):
        kwargs.setdefault('name', 'claude-code-sdk')
        kwargs.setdefault('description', 'Claude Code SDK agent')
        super().__init__(**kwargs)

        if isinstance(parse_error_retries, bool) or not isinstance(parse_error_retries, int) or parse_error_retries < 0:
            raise ValueError('parse_error_retries must be a nonnegative integer')
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.mcp_servers = mcp_servers or {}
        self.cwd = cwd or self.working_dir
        self.setting_sources = setting_sources
        self.skills = skills
        self.effort = effort
        self.thinking = thinking
        self.parse_error_retries = parse_error_retries
        self.extra_options = extra_options or {}
        self._session_id: Optional[str] = None
        self._sdk_trace: List[dict] = []
        self._call_count = 0
        self._last_finish_info: dict = {}

    def setup(self) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            raise RuntimeError("claude-agent-sdk is required. Install with: pip install claude-agent-sdk")

    async def forward(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        """Attach redacted terminal metadata to successes and failures."""
        self._last_finish_info = {}
        try:
            response = await super().forward(*message, **kwargs)
        except Exception as exc:
            self._last_finish_info = {
                'error': {
                    'kind': 'adapter_start_error',
                    'exception_type': type(exc).__name__,
                }
            }
            response = AgentMessage(
                sender=self.name,
                content=f'External agent failed: {exc}',
                extra_info={'error': str(exc), 'adapter': self.__class__.__name__},
            )
        response.finish_info = copy.deepcopy(self._last_finish_info)
        response.finish_reason = (
            'error'
            if response.extra_info.get('error')
            else response.finish_info.get('result', {}).get('stop_reason')
        )
        return response

    async def _query_once(self, task: str) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            SystemMessage,
            UserMessage,
            query,
        )

        options = ClaudeAgentOptions(permission_mode=self.permission_mode, max_turns=self.max_turns)
        if self.model:
            options.model = self.model
        if self.system_prompt:
            options.system_prompt = self.system_prompt
        if self.tools is not None:
            options.tools = self.tools
        if self.allowed_tools:
            options.allowed_tools = self.allowed_tools
        if self.disallowed_tools:
            options.disallowed_tools = self.disallowed_tools
        if self.mcp_servers:
            options.mcp_servers = self.mcp_servers
        if self.cwd:
            options.cwd = self.cwd
        if self.setting_sources is not None:
            options.setting_sources = self.setting_sources
        if self.skills is not None:
            options.skills = self.skills
        if self.effort:
            options.effort = self.effort
        if self.thinking:
            options.thinking = self.thinking
        for key, value in self.extra_options.items():
            setattr(options, key, value)

        # A captured session must win over a stale configured resume.
        if self._session_id:
            options.resume = self._session_id

        if self.proxy:
            session_key = f"sk-proxy-{self.session_id}"
            env = options.env or {}
            env.update({'ANTHROPIC_BASE_URL': self.proxy.anthropic_base_url, 'ANTHROPIC_API_KEY': session_key})
            options.env = env

        messages = []
        result_text = ''
        result_msg = None
        query_error = None
        tool_call_count = 0
        last_assistant = {}

        try:
            async for message in query(prompt=task, options=options):
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'type': type(message).__name__,
                    'call_index': self._call_count,
                }

                if isinstance(message, AssistantMessage):
                    blocks = []
                    text_parts = []
                    tool_names = []
                    for block in message.content:
                        block_dict = asdict(block)
                        block_dict['block_type'] = type(block).__name__
                        blocks.append(block_dict)
                        if isinstance(getattr(block, 'text', None), str):
                            text_parts.append(block.text)
                        if type(block).__name__ in {'ToolUseBlock', 'ServerToolUseBlock'}:
                            tool_names.append(getattr(block, 'name', None))
                    result_text = ''.join(text_parts)
                    tool_call_count += len(tool_names)
                    last_assistant = {
                        'stop_reason': getattr(message, 'stop_reason', None),
                        'error': getattr(message, 'error', None),
                        'block_types': [block['block_type'] for block in blocks],
                        'tool_names': tool_names,
                        'text_chars': len(result_text),
                    }
                    record['content'] = blocks
                    record['model'] = message.model
                    record['usage'] = message.usage
                    record['stop_reason'] = message.stop_reason
                    record['message_id'] = message.message_id
                    record['error'] = getattr(message, 'error', None)
                    if getattr(message, 'session_id', None):
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
                    for field in ('stop_reason', 'subtype', 'terminal_reason', 'api_error_status', 'errors'):
                        record[field] = getattr(message, field, None)

                elif isinstance(message, SystemMessage):
                    record['subtype'] = message.subtype
                    record['data'] = message.data
                    if isinstance(message.data, dict) and message.data.get('session_id'):
                        self._session_id = message.data['session_id']

                messages.append(record)
        except Exception as exc:
            query_error = exc

        self._sdk_trace.extend(messages)
        self._call_count += 1
        if result_msg is not None and getattr(result_msg, 'session_id', None):
            self._session_id = result_msg.session_id

        raw_errors = getattr(result_msg, 'errors', None)
        errors = [raw_errors] if isinstance(raw_errors, str) else raw_errors or []
        result_value = getattr(result_msg, 'result', None)
        result_info = {}
        if result_msg is not None:
            result_info = {
                field: getattr(result_msg, field, None)
                for field in (
                    'subtype',
                    'stop_reason',
                    'terminal_reason',
                    'api_error_status',
                    'usage',
                    'num_turns',
                    'is_error',
                    'total_cost_usd',
                )
            }
            result_info['error_count'] = len(errors)
        self._last_finish_info = {
            'event_count': len(messages),
            'event_types': [message['type'] for message in messages],
            'last_assistant': last_assistant,
            'tool_call_count': tool_call_count,
            'result': result_info,
            'terminal': {
                'has_result_message': result_msg is not None,
                'result_nonempty': isinstance(result_value, str) and bool(result_value.strip()),
                'visible_text_chars': len(result_text),
            },
        }

        exception_type = type(query_error).__name__ if query_error is not None else None
        stop_reason = getattr(result_msg, 'stop_reason', None)
        if exception_type in {'CLIJSONDecodeError', 'MessageParseError'} or str(stop_reason).lower() == 'parse_error':
            self._last_finish_info['error'] = {
                'kind': 'parse_error',
                'exception_type': exception_type,
            }
            raise ClaudeCodeParseError('Claude Code SDK returned malformed output') from query_error

        # The SDK can yield an error result and then raise ResultError when the
        # CLI exits nonzero. Keep the structured result as the primary cause.
        if result_msg is not None and result_msg.is_error:
            self._last_finish_info['error'] = {
                'kind': 'result_error',
                'error_count': len(errors),
                'exception_type': exception_type,
            }
            detail = '; '.join(str(error) for error in errors)
            raise RuntimeError(detail or str(result_value or 'Claude Code reported an error')) from query_error

        if query_error is not None:
            self._last_finish_info['error'] = {
                'kind': 'result_error' if exception_type == 'ResultError' else 'query_error',
                'exception_type': exception_type,
            }
            if exception_type == 'ResultError':
                self._last_finish_info['error'].update({
                    'subtype': getattr(query_error, 'subtype', None),
                    'terminal_reason': getattr(query_error, 'terminal_reason', None),
                    'api_error_status': getattr(query_error, 'api_error_status', None),
                    'exit_code': getattr(query_error, 'exit_code', None),
                })
            raise RuntimeError(f'Claude Code SDK failed: {query_error}') from query_error

        if isinstance(result_value, str) and result_value.strip():
            self._last_finish_info['terminal_kind'] = 'result'
            return result_value
        terminal_reason = getattr(result_msg, 'terminal_reason', None)
        if (
            result_msg is not None
            and result_text.strip()
            and not last_assistant.get('tool_names')
            and terminal_reason in (None, 'completed')
        ):
            self._last_finish_info['terminal_kind'] = 'assistant_text_fallback'
            return result_text

        reason = 'missing_result' if result_msg is None else 'empty_result'
        self._last_finish_info['terminal_kind'] = 'incomplete'
        self._last_finish_info['error'] = {'kind': 'incomplete_terminal', 'reason': reason}
        raise RuntimeError(f'Claude Code SDK returned no final text: {reason}')

    async def run_external_async(self, task: str, **kwargs) -> str:
        parse_failures = []
        current_task = task
        for attempt in range(self.parse_error_retries + 1):
            try:
                result = await self._query_once(current_task)
            except ClaudeCodeParseError:
                parse_failures.append(copy.deepcopy(self._last_finish_info))
                retry_allowed = (
                    attempt < self.parse_error_retries
                    and bool(self._session_id)
                    and self.max_turns is None
                    and self.extra_options.get('max_turns') is None
                )
                if not retry_allowed:
                    self._last_finish_info['parse_error_attempts'] = parse_failures
                    self._last_finish_info['parse_error_retries'] = attempt
                    raise
                current_task = (
                    'Continue the current task in the existing session. '
                    'The previous response could not be parsed.'
                )
            else:
                if parse_failures:
                    self._last_finish_info['parse_error_attempts'] = parse_failures
                self._last_finish_info['parse_error_retries'] = attempt
                return result
        raise AssertionError('unreachable')
