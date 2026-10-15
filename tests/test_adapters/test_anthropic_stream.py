"""Anthropic stream reconstruction must preserve malformed model output."""

import pytest

from lagent.adapters.proxy import SessionClient


@pytest.mark.parametrize(
    ('chunks', 'expected'),
    [(['{"x":', '1}'], {'x': 1}), (['{"x":'], '{"x":'), ([''], ''), ([], {})],
)
def test_tool_input_stream_preserves_invalid_json(chunks, expected):
    events = [
        {'type': 'message_start', 'message': {'id': 'message', 'role': 'assistant', 'usage': {}}},
        {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'tool_use', 'id': 'tool', 'name': 'test', 'input': {}},
        },
        *[
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': chunk}}
            for chunk in chunks
        ],
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_stop'},
    ]
    message = SessionClient._parse_anthropic_stream(events)
    assert message['content'] == [{'type': 'tool_use', 'id': 'tool', 'name': 'test', 'input': expected}]
    assert events[1]['content_block']['input'] == {}
    assert SessionClient._parse_anthropic_stream(events[:-1]) is None
