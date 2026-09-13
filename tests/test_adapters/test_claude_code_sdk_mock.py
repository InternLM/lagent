"""Synthetic regression tests for the Claude Code SDK adapter."""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter
from lagent.adapters.proxy import SessionClient


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    model: str = 'synthetic-model'
    usage: dict = field(
        default_factory=lambda: {'input_tokens': 11, 'output_tokens': 7}
    )
    stop_reason: str = 'end_turn'
    message_id: str = 'synthetic-message'
    error: str | None = None
    session_id: str | None = None


@dataclass
class SystemMessage:
    subtype: str = 'init'
    data: dict = field(default_factory=lambda: {'session_id': 'synthetic-session'})


@dataclass
class UserMessage:
    content: object


@dataclass
class ResultMessage:
    result: object = None
    is_error: bool = False
    errors: list = field(default_factory=list)
    session_id: str = 'synthetic-session'
    usage: dict = field(
        default_factory=lambda: {'input_tokens': 13, 'output_tokens': 7}
    )
    total_cost_usd: float = 0.0
    num_turns: int = 1
    stop_reason: str = 'end_turn'
    subtype: str = 'success'
    terminal_reason: str | None = None
    api_error_status: int | None = None


class MessageParseError(Exception):
    pass


class ResultError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.subtype = 'error_max_turns'
        self.terminal_reason = 'max_turns'
        self.api_error_status = None
        self.exit_code = 1


@pytest.fixture
def sdk(monkeypatch):
    module = ModuleType('claude_agent_sdk')
    for cls in (AssistantMessage, SystemMessage, UserMessage, ResultMessage):
        setattr(module, cls.__name__, cls)
    module.ClaudeAgentOptions = lambda **kwargs: SimpleNamespace(env={}, **kwargs)
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', module)
    return module


def install_script(sdk, attempts):
    calls = []

    async def query(*, prompt, options):
        index = len(calls)
        calls.append({
            'prompt': prompt,
            'resume': getattr(options, 'resume', None),
            'effort': getattr(options, 'effort', None),
        })
        for event in attempts[index]:
            if isinstance(event, Exception):
                raise event
            yield event

    sdk.query = query
    return calls


def run(agent, prompt='synthetic task'):
    return asyncio.run(agent(prompt))


def test_success_keeps_configured_effort_and_terminal_metadata(sdk):
    calls = install_script(sdk, [[
        SystemMessage(),
        AssistantMessage([TextBlock('visible')]),
        ResultMessage(result='done', terminal_reason='completed'),
    ]])

    response = run(ClaudeCodeSDKAdapter(effort='medium'))

    assert response.content == 'done'
    assert response.finish_reason == 'end_turn'
    assert calls[0]['effort'] == 'medium'
    assert response.finish_info['terminal_kind'] == 'result'
    assert response.finish_info['terminal']['result_nonempty'] is True
    assert response.finish_info['result']['subtype'] == 'success'
    assert response.finish_info['result']['terminal_reason'] == 'completed'


def test_reasoning_only_is_failure_and_finish_metadata_is_redacted(sdk):
    private_reasoning = 'PRIVATE_SYNTHETIC_REASONING'
    private_prompt = 'PRIVATE_SYNTHETIC_PROMPT'
    install_script(sdk, [[
        SystemMessage(),
        AssistantMessage(
            [ThinkingBlock(private_reasoning)],
            stop_reason='max_tokens',
        ),
        ResultMessage(result=None, stop_reason='max_tokens'),
    ]])

    response = run(ClaudeCodeSDKAdapter(), private_prompt)

    assert response.finish_reason == 'error'
    assert response.finish_info['terminal_kind'] == 'incomplete'
    assert response.finish_info['result']['stop_reason'] == 'max_tokens'
    encoded = json.dumps(response.finish_info)
    assert private_reasoning not in encoded
    assert private_prompt not in encoded


def test_tool_only_terminal_is_not_mistaken_for_success(sdk):
    install_script(sdk, [[
        SystemMessage(),
        AssistantMessage(
            [ToolUseBlock('call-1', 'read_file', {'path': 'private'})],
            stop_reason='tool_use',
        ),
        ResultMessage(result=None, stop_reason='tool_use'),
    ]])

    response = run(ClaudeCodeSDKAdapter())

    assert response.finish_reason == 'error'
    assert response.finish_info['terminal_kind'] == 'incomplete'
    assert response.finish_info['tool_call_count'] == 1
    assert response.finish_info['last_assistant']['tool_names'] == [
        'read_file'
    ]


def test_result_error_is_preserved_when_sdk_then_raises(sdk):
    calls = install_script(sdk, [[
        ResultMessage(
            is_error=True,
            errors=['synthetic SDK failure'],
            subtype='error_max_turns',
            terminal_reason='max_turns',
        ),
        ResultError('synthetic process failure'),
    ]])

    response = run(ClaudeCodeSDKAdapter(parse_error_retries=1))

    assert len(calls) == 1
    assert response.finish_reason == 'error'
    assert 'synthetic SDK failure' in response.extra_info['error']
    assert response.finish_info['error']['kind'] == 'result_error'
    assert response.finish_info['error']['exception_type'] == 'ResultError'
    assert response.finish_info['result']['is_error'] is True
    assert response.finish_info['result']['subtype'] == 'error_max_turns'
    assert response.finish_info['result']['terminal_reason'] == 'max_turns'


def test_whitespace_result_is_not_success(sdk):
    install_script(sdk, [[ResultMessage(result='   ')]])

    response = run(ClaudeCodeSDKAdapter())

    assert response.finish_reason == 'error'
    assert response.finish_info['terminal_kind'] == 'incomplete'
    assert response.finish_info['terminal']['result_nonempty'] is False


def test_intermediate_text_before_tool_terminal_is_not_success(sdk):
    install_script(sdk, [[
        AssistantMessage([TextBlock('intermediate')]),
        AssistantMessage(
            [ToolUseBlock('call-1', 'read_file', {'path': 'private'})],
            stop_reason='tool_use',
        ),
        ResultMessage(result=None, stop_reason='tool_use'),
    ]])

    response = run(ClaudeCodeSDKAdapter())

    assert response.finish_reason == 'error'
    assert response.finish_info['terminal_kind'] == 'incomplete'
    assert response.finish_info['terminal']['visible_text_chars'] == 0


def test_query_exception_records_exception_type(sdk):
    install_script(sdk, [[SystemMessage(), ValueError('synthetic failure')]])

    response = run(ClaudeCodeSDKAdapter())

    assert response.finish_reason == 'error'
    assert response.finish_info['error'] == {
        'kind': 'query_error',
        'exception_type': 'ValueError',
    }
    assert response.finish_info['event_types'] == ['SystemMessage']


def test_parse_error_retry_resumes_same_session_when_enabled(sdk):
    calls = install_script(sdk, [
        [SystemMessage(), MessageParseError('synthetic parse failure')],
        [AssistantMessage([TextBlock('visible')]), ResultMessage(result='done')],
    ])

    response = run(
        ClaudeCodeSDKAdapter(
            parse_error_retries=1,
            extra_options={'resume': 'stale-session'},
        )
    )

    assert response.content == 'done'
    assert calls[0]['resume'] == 'stale-session'
    assert calls[1]['resume'] == 'synthetic-session'
    assert response.finish_info['parse_error_retries'] == 1
    assert response.finish_info['parse_error_attempts'][0]['error'] == {
        'kind': 'parse_error',
        'exception_type': 'MessageParseError',
    }


def test_zero_argument_anthropic_tool_stream_keeps_dict_input():
    parsed = SessionClient._parse_anthropic_stream([
        {
            'type': 'message_start',
            'message': {
                'id': 'msg-1',
                'role': 'assistant',
                'model': 'synthetic-model',
                'usage': {},
            },
        },
        {
            'type': 'content_block_start',
            'content_block': {
                'type': 'tool_use',
                'id': 'tool-1',
                'name': 'noop',
                'input': {},
            },
        },
        {
            'type': 'content_block_delta',
            'delta': {'type': 'input_json_delta', 'partial_json': ''},
        },
        {'type': 'content_block_stop'},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use'},
            'usage': {},
        },
        {'type': 'message_stop'},
    ])

    assert parsed['content'][0]['input'] == {}


def test_zero_argument_tool_stream_without_initial_input_keeps_dict():
    parsed = SessionClient._parse_anthropic_stream([
        {
            'type': 'message_start',
            'message': {
                'id': 'msg-1',
                'role': 'assistant',
                'model': 'synthetic-model',
                'usage': {},
            },
        },
        {
            'type': 'content_block_start',
            'content_block': {
                'type': 'tool_use',
                'id': 'tool-1',
                'name': 'noop',
            },
        },
        {
            'type': 'content_block_delta',
            'delta': {'type': 'input_json_delta', 'partial_json': ''},
        },
        {'type': 'content_block_stop'},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use'},
            'usage': {},
        },
        {'type': 'message_stop'},
    ])

    assert parsed['content'][0]['input'] == {}


def test_client_preserves_failed_agent_message(tmp_path, monkeypatch):
    from lagent.serving.sandbox import client_cli

    response = {
        'sender': 'synthetic-agent',
        'content': 'External agent failed: synthetic failure',
        'extra_info': {'error': 'synthetic failure'},
        'finish_reason': 'error',
        'finish_info': {'terminal_kind': 'incomplete'},
    }

    async def serialized_response(*args):
        return json.dumps(response)

    monkeypatch.setattr(client_cli, '_async_call', serialized_response)
    instruction = tmp_path / 'instruction.md'
    instruction.write_text('Synthetic task', encoding='utf-8')
    response_out = tmp_path / 'response.json'
    args = SimpleNamespace(
        sock='unused',
        instruction_file=str(instruction),
        response_out=str(response_out),
        log=None,
    )

    assert client_cli.cmd_chat(args) == 5
    assert json.loads(response_out.read_text(encoding='utf-8')) == response


def test_client_rejects_malformed_error_payload(tmp_path, monkeypatch):
    from lagent.serving.sandbox import client_cli

    async def serialized_response(*args):
        return json.dumps({
            'sender': 7,
            'content': 'failure',
            'extra_info': {'error': 'invalid sender'},
        })

    monkeypatch.setattr(client_cli, '_async_call', serialized_response)
    instruction = tmp_path / 'instruction.md'
    instruction.write_text('Synthetic task', encoding='utf-8')
    response_out = tmp_path / 'response.json'
    args = SimpleNamespace(
        sock='unused',
        instruction_file=str(instruction),
        response_out=str(response_out),
        log=None,
    )

    assert client_cli.cmd_chat(args) == 5
    assert not response_out.exists()
