# Copyright (c) OpenMMLab. All rights reserved.
"""Bundled subset of ``lmdeploy.serve.anthropic.{adapter,protocol}``.

Lagent's ``SessionClient._build_anthropic_record`` needs to convert
Anthropic-shape requests to OpenAI-shape traces; the canonical implementation
lives in ``lmdeploy.serve.anthropic`` but importing it pulls torch + the full
inference engine through ``lmdeploy/__init__.py``'s
``from .api import client, pipeline, serve`` chain. That's fine on the host
(trainer already has lmdeploy), but the in-sandbox SessionClient only needs
the tiny pure-Python conversion routines and shouldn't carry the multi-GB
inference dep.

This module copies the strict minimum needed by lagent's proxy
(``SessionClient``):

  - ``MessagesRequest`` + supporting input models (``MessageParam``,
    ``ContentBlockParam``, ``ToolParam``, ``ToolChoiceParam``)
  - ``to_openai_messages`` (Anthropic messages -> OpenAI dicts)
  - ``to_openai_tools`` (Anthropic tools -> OpenAI ``Tool`` objects)
  - ``Tool`` / ``Function`` (OpenAI-shape tool wrappers)

Streaming events, lmdeploy-specific generation config, tokenizer hooks, and
finish-reason mapping are intentionally omitted — they belong to the worker
side, which has the real ``lmdeploy`` package available.

Source of truth: lmdeploy/serve/anthropic/{adapter,protocol}.py and
lmdeploy/serve/openai/protocol.py. Re-sync if those drift.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# Anthropic input protocol (subset)
# ─────────────────────────────────────────────────────────────────────────────


class ContentBlockParam(BaseModel):
    """Permissive Anthropic input content block.

    Claude Code may replay beta conversation history containing blocks such as
    ``tool_use`` and ``tool_result``. The adapter decides how to render each
    block into a flat OpenAI message.
    """

    model_config = ConfigDict(extra='allow')

    type: str
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any | None = None
    tool_use_id: str | None = None
    content: str | list[Any] | dict[str, Any] | None = None
    is_error: bool | None = None


class MessageParam(BaseModel):
    """Anthropic input message."""

    role: Literal['user', 'assistant', 'system']
    content: str | list[ContentBlockParam]


class ToolParam(BaseModel):
    """Anthropic tool definition in request body."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ToolChoiceAutoParam(BaseModel):
    type: Literal['auto'] = 'auto'


class ToolChoiceAnyParam(BaseModel):
    type: Literal['any'] = 'any'


class ToolChoiceToolParam(BaseModel):
    type: Literal['tool'] = 'tool'
    name: str


ToolChoiceParam = ToolChoiceAutoParam | ToolChoiceAnyParam | ToolChoiceToolParam


class MessagesRequest(BaseModel):
    """Request body for ``POST /v1/messages`` (subset used by the conversion).

    Permissive on extra fields so Claude Agent SDK extensions
    (``context_management`` / ``output_config`` / ``provider`` / ...) survive
    ``model_validate`` without raising.
    """

    model_config = ConfigDict(extra='allow')

    model: str
    messages: list[MessageParam] | None = Field(default=None)
    max_tokens: int = Field(gt=0)
    system: str | list[ContentBlockParam] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[ToolParam] | None = None
    tool_choice: ToolChoiceParam | Literal['auto', 'any'] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-shape tool wrappers (subset)
# ─────────────────────────────────────────────────────────────────────────────


class Function(BaseModel):
    description: str | None = Field(default=None, examples=[None])
    name: str
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: str = Field(default='function', examples=['function'])
    function: Function


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helpers (verbatim from lmdeploy.serve.anthropic.adapter)
# ─────────────────────────────────────────────────────────────────────────────


def _block_get(block: ContentBlockParam | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _stringify_block_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if hasattr(value, 'model_dump'):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False)


def _text_from_block_content(content: Any, field_name: str) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_blocks(content, field_name=field_name)
    return _stringify_block_value(content)


def _text_from_blocks(blocks: list[ContentBlockParam | dict[str, Any]], field_name: str) -> str:
    out: list[str] = []
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            block_type = block.get('type')
            text = block.get('text')
            content = block.get('content')
            tool_use_id = block.get('tool_use_id')
            tool_name = block.get('name')
            tool_input = block.get('input')
            thinking = block.get('thinking')
        else:
            block_type = block.type
            text = block.text
            content = block.content
            tool_use_id = block.tool_use_id
            tool_name = block.name
            tool_input = block.input
            thinking = block.thinking
        if block_type == 'text':
            if text is None:
                raise ValueError(f'Missing `text` in `{field_name}` content block at index {idx}.')
            out.append(text)
        elif block_type == 'tool_result':
            result_text = _text_from_block_content(content, field_name=f'{field_name}[{idx}].content')
            out.append(f'\n[tool_result id={tool_use_id or ""}]\n{result_text}\n[/tool_result]\n')
        elif block_type == 'tool_use':
            tool_payload = _stringify_block_value(tool_input)
            out.append(f'\n[tool_use name={tool_name or ""}]\n{tool_payload}\n[/tool_use]\n')
        elif block_type in ('thinking', 'redacted_thinking'):
            if thinking:
                out.append(f'\n[thinking]\n{thinking}\n[/thinking]\n')
        else:
            out.append(f'\n[{block_type}]\n{_stringify_block_value(block)}\n[/{block_type}]\n')
    return ''.join(out)


def _convert_image_source_to_url(source: Any) -> str:
    source_type = _block_get(source, 'type')
    if source_type == 'url':
        return _block_get(source, 'url', '')
    if source_type == 'base64':
        media_type = _block_get(source, 'media_type', 'image/jpeg')
        data = _block_get(source, 'data', '')
        if data:
            return f'data:{media_type};base64,{data}'
    return ''


def _convert_system_blocks_to_text(system: list[ContentBlockParam]) -> str:
    system_prompt = ''
    for block in system:
        if _block_get(block, 'type') != 'text':
            continue
        text = _block_get(block, 'text')
        if not text or text.startswith('x-anthropic-billing-header'):
            continue
        system_prompt += text
    return system_prompt


def _convert_user_tool_result(block: ContentBlockParam | dict[str, Any]) -> list[dict[str, Any]]:
    tool_text = ''
    tool_image_urls: list[str] = []
    content = _block_get(block, 'content')

    if isinstance(content, str):
        tool_text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if _block_get(item, 'type') == 'text':
                text_parts.append(_block_get(item, 'text', ''))
            elif _block_get(item, 'type') == 'image':
                url = _convert_image_source_to_url(_block_get(item, 'source', {}))
                if url:
                    tool_image_urls.append(url)
        tool_text = '\n'.join(text_parts)

    messages = [
        dict(
            role='tool',
            tool_call_id=_block_get(block, 'tool_use_id') or _block_get(block, 'id') or '',
            content=tool_text or '',
        )
    ]
    if tool_image_urls:
        messages.append(
            dict(
                role='user',
                content=[dict(type='image_url', image_url=dict(url=url)) for url in tool_image_urls],
            )
        )
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Public conversion API (used by lagent.adapters.proxy and xtuner SessionServer)
# ─────────────────────────────────────────────────────────────────────────────


def to_openai_tools(tools: list[ToolParam] | None) -> list[Tool] | None:
    """Convert Anthropic tools into OpenAI protocol tool entries."""

    if not tools:
        return None
    return [
        Tool(
            type='function',
            function=Function(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            ),
        )
        for tool in tools
    ]


def to_openai_messages(request: MessagesRequest) -> list[dict[str, Any]]:
    """Convert Anthropic request messages into OpenAI-compatible message dicts."""

    openai_messages: list[dict[str, Any]] = []
    if request.system is not None:
        if isinstance(request.system, str):
            openai_messages.append(dict(role='system', content=request.system))
        else:
            openai_messages.append(dict(role='system', content=_convert_system_blocks_to_text(request.system)))

    for idx, message in enumerate(request.messages or []):
        if isinstance(message.content, str):
            openai_messages.append(dict(role=message.role, content=message.content))
            continue

        openai_message: dict[str, Any] = dict(role=message.role)
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block_idx, block in enumerate(message.content):
            block_type = _block_get(block, 'type')
            if block_type == 'text':
                text = _block_get(block, 'text')
                if text:
                    content_parts.append(dict(type='text', text=text))
                continue

            if block_type == 'image':
                source = _block_get(block, 'source')
                if source:
                    url = _convert_image_source_to_url(source)
                    if url:
                        content_parts.append(dict(type='image_url', image_url=dict(url=url)))
                continue

            if block_type == 'tool_use':
                name = _block_get(block, 'name') or ''
                arguments = json.dumps(_block_get(block, 'input') or {})
                # Anthropic responses always carry a ``toolu_`` id; fall back to a
                # deterministic content hash (never a random uuid) so converting
                # the same block twice yields a stable id and records stay faithful
                # to the upstream tokens.
                tool_id = _block_get(block, 'id')
                if not tool_id:
                    digest = hashlib.sha1(f'{name}:{arguments}'.encode('utf-8')).hexdigest()[:8]
                    tool_id = f'call_{digest}'
                tool_calls.append(
                    dict(
                        id=tool_id,
                        type='function',
                        function=dict(name=name, arguments=arguments),
                    )
                )
                continue

            if block_type == 'tool_result':
                if message.role == 'user':
                    openai_messages.extend(_convert_user_tool_result(block))
                else:
                    result_text = _text_from_block_content(
                        _block_get(block, 'content'),
                        field_name=f'messages[{idx}].content[{block_idx}].content',
                    )
                    content_parts.append(dict(type='text', text=f'Tool result: {result_text}'))
                continue

            if block_type == 'thinking':
                thinking = _block_get(block, 'thinking')
                if thinking is not None:
                    reasoning_parts.append(thinking)
                continue

            if block_type == 'redacted_thinking':
                continue

            content_parts.append(dict(type='text', text=_stringify_block_value(block)))

        if reasoning_parts:
            openai_message['reasoning_content'] = ''.join(reasoning_parts)
        if tool_calls:
            openai_message['tool_calls'] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]['type'] == 'text':
                openai_message['content'] = content_parts[0]['text']
            else:
                openai_message['content'] = content_parts

        if 'content' in openai_message or 'tool_calls' in openai_message or 'reasoning_content' in openai_message:
            openai_messages.append(openai_message)
    return openai_messages


__all__ = [
    'ContentBlockParam',
    'MessageParam',
    'MessagesRequest',
    'ToolParam',
    'ToolChoiceParam',
    'Tool',
    'Function',
    'to_openai_messages',
    'to_openai_tools',
]
