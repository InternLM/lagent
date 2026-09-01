"""LLM Proxy Recorder — a lightweight HTTP proxy (`SessionClient`) for intercepting,
translating, and recording LLM request/response trajectories.

This proxy intercepts API calls from external agents, forwards them
to the actual model backend, and quietly records the full conversation history (trajectories).

Key Features:
- **Direct Passthrough**: Calls OpenAI models using OpenAI format, and Anthropic models using Anthropic format. Returns matching formats identically without forced translation.
- **Trajectory Recording**: Merges and retains conversation turns into `_records`.
  Interrupted or duplicate prefix traces are intelligently filtered when calling
  `get_messages()`.

Usage::

    proxy = SessionClient(
        real_api_key="sk-ant-...",
        real_base_url="https://api.anthropic.com",
        session_id="my-session-id"
    )
    await proxy.start()

    # Configure your agent's LLM client to hit the proxy:
    # OPENAI_BASE_URL = proxy.openai_base_url
    #
    # Retrieve the deduplicated chat paths later:
    trajectories = proxy.get_messages()

    await proxy.stop()
"""

import copy
import json
import os
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

from lagent.utils import create_object, ctx_session_id, get_logger

from .request_processor import ProxyRequestContext, ProxyRequestProcessor

logger = get_logger(__name__, 'info')


# anthropic_beta flags to forward to the (possibly Bedrock-backed) upstream.
# The Claude Code SDK injects flags Bedrock rejects (e.g. ``claude-code-*``,
# ``oauth-*``), which surface as ``ValidationException: invalid beta flag`` on
# InvokeModelWithResponseStream. We forward only this allowlist and drop the
# rest. ``interleaved-thinking-2025-05-14`` is kept so the model keeps emitting
# reasoning between tool calls. Override via the env var (comma-separated).
_FORWARD_ANTHROPIC_BETAS = {
    s.strip()
    for s in os.environ.get(
        "PROXY_ANTHROPIC_BETA_ALLOWLIST",
        "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
    ).split(",")
    if s.strip()
}

# Top-level Anthropic body fields the Claude Code CLI injects that some backends
# (e.g. Bedrock) reject with "Extra inputs are not permitted". Dropped on the
# /messages path; lmdeploy ignores them anyway (MessagesRequest extra='allow').
# ``context_management`` is the CLI's server-side context-editing field.
_DROP_ANTHROPIC_BODY_FIELDS = {
    s.strip()
    for s in os.environ.get(
        "PROXY_ANTHROPIC_DROP_FIELDS",
        "context_management",
    ).split(",")
    if s.strip()
}


def _is_lmdeploy_input_length_error(response_data: dict[str, Any]) -> bool:
    """Detect lmdeploy's INPUT_LENGTH_ERROR sentinel in an Anthropic response."""
    for block in response_data.get('content') or []:
        if not isinstance(block, dict):
            continue
        text = block.get('text') or block.get('thinking') or ''
        if (
            isinstance(text, str)
            and 'internal error happened, status code ResponseType.INPUT_LENGTH_ERROR' in text.strip()
        ):
            return True
    return False


_NON_TOKENIZED_KEYS = ('cache_control',)
_TOKENIZED_MSG_KEYS = ('tool_call_id', 'name', 'reasoning_content', 'function_call', 'refusal')
_OPENAI_REASONING_DELTA_KEYS = ('reasoning_content', 'reasoning', 'thinking')
_OPENCLAW_TOOL_CALL_ID_RE = re.compile(r'^call_?([0-9a-fA-F]{8,})$')
_PREFIX_TRIE_TERMINAL = object()


def _extract_openai_reasoning_delta(delta: dict[str, Any]) -> Optional[str]:
    for key in _OPENAI_REASONING_DELTA_KEYS:
        value = delta.get(key)
        if value:
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return None


def _canonical_content(content: Any) -> Any:
    """Token-equivalence projection of a message ``content`` value."""
    if content is None:
        return None
    if isinstance(content, str):
        # Whitespace-only content is a serialization artifact: a freshly
        # generated assistant turn may carry an empty text block (e.g. '\n\n')
        # alongside its tool_calls, but the client drops it when replaying the
        # turn as history. Treat it as absent so the prefix chain collapses.
        return content if content.strip() else None
    if isinstance(content, list):
        # A list whose blocks are all plain text is token-equivalent to the
        # concatenated string form: a freshly generated string ``content`` is
        # replayed by the client as ``[{"type": "text", "text": ...}]`` history.
        # Collapse to the joined text (and reuse the whitespace-as-absent rule)
        # so the two shapes prefix-match.
        texts = []
        for block in content:
            keys = (block.keys() - set(_NON_TOKENIZED_KEYS)) if isinstance(block, dict) else None
            if keys is not None and block.get('type') == 'text' and keys <= {'type', 'text'}:
                texts.append(block.get('text') or '')
            else:
                break
        else:
            joined = ''.join(texts)
            return joined if joined.strip() else None
        parts = []
        for block in content:
            if isinstance(block, dict):
                block = {k: v for k, v in block.items() if k not in _NON_TOKENIZED_KEYS}
            parts.append(json.dumps(block, sort_keys=True, ensure_ascii=False))
        return tuple(parts) or None
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def _canonical_tool_call_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    # OpenClaw currently replays history with the underscore in OpenAI-style
    # tool call ids stripped (`call_<hex>` -> `call<hex>`). Normalize only that
    # narrow generated-id shape for prefix matching; do not drop the id.
    match = _OPENCLAW_TOOL_CALL_ID_RE.fullmatch(value)
    if match:
        return f"call_{match.group(1).lower()}"
    return value


def _canonical_msg(msg: Any) -> tuple:
    """Project a message onto the fields the model actually conditions on.

    Used as the comparison basis for prefix dedup so that volatile, non-tokenized
    metadata (cache breakpoints, string-vs-text-block content shape, etc.) does not
    stop a true prefix from matching. Fields that DO reach the tokenizer (tool-call
    ids, arguments, reasoning) are deliberately retained.
    """
    if not isinstance(msg, dict):
        return ('raw', json.dumps(msg, sort_keys=True, ensure_ascii=False))
    key: list = [('role', msg.get('role')), ('content', _canonical_content(msg.get('content')))]
    tool_calls = msg.get('tool_calls')
    if tool_calls:
        norm = []
        for tc in tool_calls:
            fn = (tc.get('function') or {}) if isinstance(tc, dict) else {}
            args = fn.get('arguments')
            if isinstance(args, (dict, list)):
                args = json.dumps(args, sort_keys=True, ensure_ascii=False)
            # Keep the id: it is serialized into the prompt the model conditions on.
            norm.append(
                (
                    _canonical_tool_call_id(tc.get('id')) if isinstance(tc, dict) else None,
                    fn.get('name'),
                    args,
                )
            )
        key.append(('tool_calls', tuple(norm)))
    for field in _TOKENIZED_MSG_KEYS:
        val = msg.get(field)
        if val:
            if field == 'tool_call_id':
                val = _canonical_tool_call_id(val)
            key.append((field, val if isinstance(val, str) else json.dumps(val, sort_keys=True, ensure_ascii=False)))
    return tuple(key)


def _toolset_key(tools: Any) -> str:
    """Stable key for the exact tool schema rendered into the prompt."""
    return json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _maximal_prefix_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only maximal strict-prefix leaves, grouped by exact tool schema."""
    roots: dict[str, dict[Any, Any]] = {}
    for index, record in enumerate(records):
        node = roots.setdefault(_toolset_key(record.get('tools')), {})
        for message in record.get('messages', []):
            node = node.setdefault(_canonical_msg(message), {})
        # Canonical duplicates share a terminal; the latest record wins.
        node[_PREFIX_TRIE_TERMINAL] = index

    retained: set[int] = set()

    for root in roots.values():
        subtree_has_terminal: dict[int, bool] = {}
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            children = [child for key, child in node.items() if key is not _PREFIX_TRIE_TERMINAL]
            if not expanded:
                stack.append((node, True))
                stack.extend((child, False) for child in children)
                continue

            descendant_has_terminal = any(subtree_has_terminal[id(child)] for child in children)
            terminal = node.get(_PREFIX_TRIE_TERMINAL)
            if terminal is not None and not descendant_has_terminal:
                retained.add(terminal)
            subtree_has_terminal[id(node)] = terminal is not None or descendant_has_terminal
    return [records[index] for index in sorted(retained)]


def _anthropic_response_to_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = response.get('content') or []

    normalized: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = block.get('type')

        if block_type == 'text':
            text = block.get('text', '')
            normalized.append({'type': 'text', 'text': text})

        elif block_type == 'thinking':
            # Anthropic API 要求原样回传 thinking block（含 signature）
            normalized_block: dict[str, Any] = {'type': 'thinking', 'thinking': block.get('thinking', '')}
            # 若 API 返回了 signature 字段（真实 Anthropic 云端会附带），保留它
            if 'signature' in block:
                normalized_block['signature'] = block['signature']
            normalized.append(normalized_block)

        elif block_type == 'tool_use':
            # 必须原样回传，id/name/input 缺一不可
            tool_id = block.get('id')
            name = block.get('name')
            input_ = block.get('input', {})

            if not tool_id:
                raise ValueError(f"tool_use block is missing 'id': {block}")
            if not name:
                raise ValueError(f"tool_use block is missing 'name': {block}")

            normalized.append({'type': 'tool_use', 'id': tool_id, 'name': name, 'input': input_})
        else:
            logger.warning(f"Skipping unmodeled response content block type '{block_type}'")
            continue

    return {'role': 'assistant', 'content': normalized}


def _maybe_json_loads(value):
    """Return ``json.loads(value)`` for str input, falling back to the original on error.

    Used to enforce the convention that tool-call ``arguments`` are stored as a
    dict (driving ``get_messages()`` prefix dedup). Non-string values pass through.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_tool_call_arguments(messages) -> None:
    """In-place: parse ``tool_calls[].function.arguments`` strings into dicts.

    lmdeploy serializes them as JSON strings; the standard OpenAI path stores
    them as dicts. Normalize so both paths share one shape.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get('tool_calls') or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function')
            if isinstance(fn, dict) and 'arguments' in fn:
                fn['arguments'] = _maybe_json_loads(fn['arguments'])


def responses_tools_to_openai(tools) -> Optional[list]:
    """Convert OpenAI Responses-API tools to Chat Completions ``tools`` shape.

    Responses uses a flat ``{type: "function", name, description, parameters}``
    for function tools; Chat Completions nests them under ``function``. Non-
    function tool types (``web_search``, ``file_search``, ...) are kept as-is.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get('type') == 'function' and 'function' not in t:
            out.append(
                {
                    'type': 'function',
                    'function': {
                        'name': t.get('name', ''),
                        'description': t.get('description', ''),
                        'parameters': t.get('parameters', {}) or {},
                    },
                }
            )
        else:
            out.append(t)
    return out


class SessionClient:
    """Lightweight HTTP proxy that records LLM request/response pairs.

    Args:
        real_api_key: The actual API key to use when forwarding requests.
        real_base_url: The actual LLM API base URL to forward to.
        port: Port to listen on. 0 means auto-assign.
        request_processor: Optional stateful request/response processor. It is
            disabled by default, preserving direct proxy behavior.
    """

    def __init__(
        self,
        real_api_key: str,
        real_base_url: str,
        port: int = 0,
        session_id: Optional[str] = None,
        extra_body: Optional[dict] = None,
        http_proxy: Optional[str] = None,
        request_processor: Optional[Union[dict, ProxyRequestProcessor]] = None,
    ):
        self.real_api_key = real_api_key
        self.real_base_url = real_base_url.rstrip('/')
        self.port = port
        self.http_proxy = http_proxy
        self.session_id = session_id or ctx_session_id.get() or os.getenv('XTUNER_SESSION_ID') or str(uuid.uuid4().int)
        self.extra_body = extra_body or {}
        self.request_processor = create_object(request_processor)
        self._request_index = 0
        self._request_processor_disabled = False
        self._request_processor_errors = 0
        self._processor_stats_logged = False
        self._records: Dict[str, List[Dict[str, list]]] = defaultdict(list)
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    @property
    def url(self) -> str:
        """The raw proxy endpoint."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """OpenAI-compatible API base URL exposed by this proxy."""
        return f"{self.url}/v1"

    @property
    def anthropic_base_url(self) -> str:
        """Anthropic-compatible API base URL exposed by this proxy."""
        return self.url

    @property
    def is_running(self) -> bool:
        return self._site is not None

    async def start(self):
        """Start the proxy HTTP server."""
        if self.is_running:
            return
        self._app = web.Application()
        # Catch-all route to proxy any path
        self._app.router.add_route('*', '/{path:.*}', self._handle_request)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, '127.0.0.1', self.port)
        await self._site.start()
        # Update port if auto-assigned
        if self.port == 0:
            self.port = self._site._server.sockets[0].getsockname()[1]
        logger.info(f"LLM proxy started on port {self.port}")

    async def stop(self):
        """Stop the proxy HTTP server."""
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        logger.info('LLM proxy stopped')

    def _apply_request_processor(
        self,
        request_data: dict[str, Any],
        context: ProxyRequestContext,
    ) -> dict[str, Any]:
        if self.request_processor is None or self._request_processor_disabled:
            return request_data
        fallback = copy.deepcopy(request_data)
        try:
            processed = self.request_processor.before_forward(copy.deepcopy(request_data), context)
            if not isinstance(processed, dict):
                raise TypeError('request processor must return a dict')
            return processed
        except Exception as exc:
            self._request_processor_errors += 1
            self._request_processor_disabled = True
            logger.warning(
                'Disabling request processor %s for session %s after before_forward failed: %s',
                type(self.request_processor).__name__,
                self.session_id,
                exc,
            )
            return fallback

    def _observe_request_processor_response(
        self,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
        context: ProxyRequestContext,
    ) -> None:
        if self.request_processor is None or self._request_processor_disabled:
            return
        try:
            self.request_processor.after_response(
                copy.deepcopy(request_data),
                copy.deepcopy(response_data),
                context,
            )
        except Exception as exc:
            self._request_processor_errors += 1
            self._request_processor_disabled = True
            logger.warning(
                'Disabling request processor %s for session %s after after_response failed: %s',
                type(self.request_processor).__name__,
                self.session_id,
                exc,
            )

    def get_request_processor_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if self.request_processor is not None:
            try:
                stats.update(self.request_processor.get_stats())
            except Exception:
                stats['stats_unavailable'] = True
        stats['processor_errors'] = self._request_processor_errors
        stats['processor_disabled'] = self._request_processor_disabled
        return stats

    def _log_request_processor_stats(self, records: int, retained: int) -> None:
        if self.request_processor is None or self._processor_stats_logged:
            return
        stats = self.get_request_processor_stats()
        stats.update({'records': records, 'retained_records': retained})
        logger.info('Request processor summary for session %s: %s', self.session_id, json.dumps(stats, sort_keys=True))
        self._processor_stats_logged = True

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler: extract session, forward, record, return."""
        # 1. Read request body
        request_body = await request.read()
        request_data = None
        try:
            request_data = json.loads(request_body) if request_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # Detect provider format by inspecting the requested endpoint
        req_path = request.match_info['path']
        is_anthropic = req_path.endswith('/messages') or '/v1/messages' in req_path
        is_responses = req_path.endswith('/responses') or '/v1/responses' in req_path

        # 2. Inject session_id into request body
        if isinstance(request_data, dict):
            request_data.update(self.extra_body)
            request_data['session_id'] = self.session_id
            if is_anthropic:
                request_data['provider'] = 'anthropic'
                # Drop SDK-injected beta flags the upstream (e.g. Bedrock) rejects;
                # keep only the allowlist (see _FORWARD_ANTHROPIC_BETAS).
                betas = request_data.get('anthropic_beta')
                if isinstance(betas, list):
                    kept = [b for b in betas if b in _FORWARD_ANTHROPIC_BETAS]
                    if kept:
                        request_data['anthropic_beta'] = kept
                    else:
                        request_data.pop('anthropic_beta', None)
                # Drop top-level body fields the upstream rejects (e.g. Bedrock's
                # "context_management: Extra inputs are not permitted").
                for _f in _DROP_ANTHROPIC_BODY_FIELDS:
                    request_data.pop(_f, None)
            if 'messages' in request_data or 'input' in request_data:
                provider = 'anthropic' if is_anthropic else ('responses' if is_responses else 'openai')
                request_context = ProxyRequestContext(
                    session_id=self.session_id,
                    request_index=self._request_index,
                    provider=provider,
                    path=req_path,
                )
                self._request_index += 1
                request_data = self._apply_request_processor(request_data, request_context)
            request_body = json.dumps(request_data).encode('utf-8')

        # By default we assume the incoming request is already in the target format
        provider_request_data = copy.deepcopy(request_data) if request_data else {}

        # 4. Build forwarding headers — replace auth with real key
        forward_headers = dict(request.headers)
        forward_headers.pop('Host', None)
        forward_headers.pop('host', None)
        # CRITICAL: We modified the request body, so the original Content-Length is wrong.
        # We must remove it so aiohttp can calculate the correct length automatically.
        forward_headers.pop('Content-Length', None)
        forward_headers.pop('content-length', None)

        if 'Authorization' in forward_headers:
            forward_headers['Authorization'] = f'Bearer {self.real_api_key}'
        if 'x-api-key' in forward_headers:
            forward_headers['x-api-key'] = self.real_api_key
        if is_anthropic:
            forward_headers['anthropic-version'] = '2023-06-01'
            # Mirror the body filter on the anthropic-beta header (comma-joined).
            for k in [h for h in forward_headers if h.lower() == 'anthropic-beta']:
                kept = [v.strip() for v in forward_headers[k].split(',') if v.strip() in _FORWARD_ANTHROPIC_BETAS]
                if kept:
                    forward_headers[k] = ','.join(kept)
                else:
                    forward_headers.pop(k, None)

        # 5. Forward to real LLM
        # Build target URL, avoiding path duplication
        # e.g. real_base_url="http://api.com/v1", path="/v1/chat/completions"
        # should produce "http://api.com/v1/chat/completions" not "http://api.com/v1/v1/..."
        base_parsed = urlparse(self.real_base_url)
        base_path = base_parsed.path.rstrip('/')
        if req_path.startswith(base_path.lstrip('/')):
            # Path already includes the base path prefix, use as-is
            target_url = f"{base_parsed.scheme}://{base_parsed.netloc}/{req_path}"
        else:
            target_url = f"{self.real_base_url}/{req_path.lstrip('/')}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = provider_request_data.get('stream', False) if provider_request_data else False

        async with aiohttp.ClientSession() as client:
            async with client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                data=request_body,
                proxy=self.http_proxy,
            ) as resp:
                if is_stream:
                    # Stream response: collect chunks, forward as-is.
                    # If the downstream client disconnects mid-stream (e.g. an
                    # AsyncAPIClient bails out on a finish_reason=='error' chunk
                    # after the prompt overflowed the session), keep draining
                    # the upstream so the trace is still recorded in full.
                    response_chunks = []
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ('transfer-encoding', 'content-length', 'content-encoding')
                        },
                    )
                    await response.prepare(request)
                    client_alive = True
                    async for chunk in resp.content.iter_any():
                        response_chunks.append(chunk)
                        if client_alive:
                            try:
                                await response.write(chunk)
                            except (ConnectionError, aiohttp.ClientConnectionResetError):
                                client_alive = False
                    if client_alive:
                        try:
                            await response.write_eof()
                        except (ConnectionError, aiohttp.ClientConnectionResetError):
                            pass
                    raw_response = b''.join(response_chunks)
                else:
                    raw_response = await resp.read()
                    response = web.Response(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ('transfer-encoding', 'content-length', 'content-encoding')
                        },
                        body=raw_response,
                    )

        # 6. Parse response for recording
        response_data = None
        if is_stream:
            # Parse SSE stream to extract final data
            response_data = self._parse_stream_response(raw_response)
        else:
            try:
                response_data = json.loads(raw_response)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Upstream returned an error body — forward it to the client
            # untouched (so the agent sees the real error), but treat
            # response as invalid so we don't record garbage.
            if response_data and response_data.get('type') == 'error':
                logger.warning(f"Anthropic-format error from upstream: {response_data}")
                response_data = None
            elif response_data and isinstance(response_data.get('error'), dict):
                logger.warning(f"OpenAI-format error from upstream: {response_data}")
                response_data = None
            elif (
                response_data
                and isinstance(response_data.get('choices'), list)
                and response_data['choices']
                and response_data['choices'][0].get('finish_reason') == 'error'
            ):
                logger.warning(f"OpenAI finish_reason=error from upstream: {response_data}")
                response_data = None
            elif response_data and _is_lmdeploy_input_length_error(response_data):
                # Anthropic counterpart of finish_reason=error: lmdeploy returns
                # HTTP 200 with the error text in content (stop_reason is remapped
                # to stop_sequence, so we can't key on it).
                logger.warning(f"lmdeploy INPUT_LENGTH_ERROR from upstream: {response_data}")
                response_data = None

        if not response_data:
            return response

        if isinstance(request_data, dict) and ('messages' in request_data or 'input' in request_data):
            self._observe_request_processor_response(request_data, response_data, request_context)

        # 7. Record
        if isinstance(request_data, dict) and ('messages' in request_data or 'input' in request_data):
            request_data = copy.deepcopy(request_data)
            if is_anthropic:
                built = self._build_anthropic_record(request_data, response_data)
            elif is_responses:
                built = self._build_responses_record(request_data, response_data)
            else:
                built = self._build_openai_record(request_data, response_data)

            if built is None:
                return response

            messages, tools = built
            self._records[self.session_id].append({'messages': messages, 'tools': tools})
            logger.debug(
                f"Updated messages for session {self.session_id}: {len(self._records[self.session_id])} traces total"
            )
        return response

    def _build_anthropic_record(self, request_data, response_data):
        try:
            from lmdeploy.serve.anthropic.adapter import to_openai_messages, to_openai_tools
            from lmdeploy.serve.anthropic.protocol import MessagesRequest
        except ImportError:
            from lagent.utils.adapter import MessagesRequest, to_openai_messages, to_openai_tools

        try:
            resp_msg = _anthropic_response_to_assistant_message(response_data)
        except Exception as exc:
            logger.warning(f"Failed to parse Anthropic response: {exc}")
            return None

        # Anthropic server-side built-ins (web_search_*, computer_*, bash_*,
        # text_editor_*) only carry ``type`` + ``name`` and lack the
        # ``input_schema`` that ``ToolParam`` requires. They're handled by
        # Anthropic's own infra so the trace-side conversion can't model them
        # — drop them here (the wire-forwarded body upstream is untouched, so
        # the real provider still gets the original list).
        tools_in = request_data.get('tools')
        if isinstance(tools_in, list):
            kept = [t for t in tools_in if isinstance(t, dict) and 'input_schema' in t]
            if len(kept) != len(tools_in):
                dropped = [t.get('type') or t.get('name') for t in tools_in if t not in kept]
                logger.debug(
                    f"Dropping {len(tools_in) - len(kept)} anthropic server-side tool(s) from trace record: {dropped}"
                )
            if kept:
                request_data['tools'] = kept
            else:
                request_data.pop('tools', None)

        request_data['messages'].append(resp_msg)
        req = MessagesRequest.model_validate(request_data)
        messages = to_openai_messages(req)
        if not messages or messages[-1].get('role') != 'assistant':
            logger.debug(f"Skipping record for session {self.session_id}: assistant turn dropped in conversion")
            return None
        tools = [tool.model_dump() for tool in to_openai_tools(req.tools)] if req.tools else None
        _normalize_tool_call_arguments(messages)
        return messages, tools

    def _build_responses_record(self, request_data, response_data):
        # OpenAI Responses API: the conversation is a flat list of output items
        # (message / function_call / reasoning / ...). Store them flattened so
        # the client's next-turn `input` (which echoes them) prefix-matches
        # under get_messages() dedup. Convention: `function_call.arguments` is
        # stored as a dict, applied uniformly to input- and output-side items.
        if not response_data.get('output'):
            logger.debug(f"Skipping record for session {self.session_id}: no valid response")
            return None

        raw_input = request_data['input']
        if isinstance(raw_input, str):
            messages = [{'role': 'user', 'content': raw_input}]
        elif isinstance(raw_input, list):
            messages = raw_input
        else:
            messages = []
        messages.extend(response_data['output'])
        tools = responses_tools_to_openai(request_data.get('tools'))

        for item in messages:
            if (
                isinstance(item, dict)
                and item.get('type') == 'function_call'
                and isinstance(item.get('arguments'), str)
            ):
                item['arguments'] = _maybe_json_loads(item['arguments'])
        return messages, tools

    def _build_openai_record(self, request_data, response_data):
        # Standard OpenAI chat completions trace format.
        if not (response_data.get('choices') and response_data['choices'][0].get('message')):
            logger.debug(f"Skipping record for session {self.session_id}: no valid response")
            return None

        messages = request_data.get('messages', [])
        choice = response_data['choices'][0]
        raw_msg = choice['message']
        assistant_msg = {'role': raw_msg.get('role', 'assistant')}
        # Only keep pure standard OpenAI fields to prevent contamination.
        allowed_fields = ['content', 'tool_calls', 'function_call', 'refusal']

        for field in allowed_fields:
            if raw_msg.get(field) is None:
                continue
            val = copy.deepcopy(raw_msg[field])
            if field == 'function_call':
                val['arguments'] = _maybe_json_loads(val.get('arguments'))
            assistant_msg[field] = val
        reasoning_content = _extract_openai_reasoning_delta(raw_msg)
        if reasoning_content:
            assistant_msg['reasoning_content'] = reasoning_content
        finish_reason = choice.get('finish_reason')
        if finish_reason is not None:
            assistant_msg['finish_reason'] = finish_reason
        messages.append(assistant_msg)
        _normalize_tool_call_arguments(messages)
        return messages, request_data.get('tools')

    @staticmethod
    def _parse_stream_response(raw: bytes) -> Optional[dict]:
        """Parse SSE stream response to reconstruct the complete message.

        Supports both Anthropic and OpenAI streaming formats. Returns
        ``None`` if the stream is invalid (malformed SSE JSON, no
        events, or format-specific failure modes detected by the
        per-format parsers).
        """
        text = raw.decode('utf-8', errors='replace')
        events: list = []
        saw_done = False

        for line in text.split('\n'):
            line = line.strip()
            if not line.startswith('data: '):
                continue
            if line == 'data: [DONE]':
                saw_done = True
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                # Mirrors model.py: malformed SSE JSON invalidates the
                # whole stream rather than getting silently skipped.
                logger.warning(f"Invalid SSE JSON in stream: {line[6:][:200]}")
                return None

        if not events:
            return None

        # Detect format: Responses API events start with "response.";
        # OpenAI ChatCompletion has "choices"; Anthropic uses message_start / content_block_*
        first = events[0]
        evt_type = first.get('type', '')
        if evt_type.startswith('response.') or first.get('object') == 'response':
            return SessionClient._parse_responses_stream(events)
        if 'choices' in first or first.get('object') == 'chat.completion.chunk':
            return SessionClient._parse_openai_stream(events, saw_done=saw_done)
        return SessionClient._parse_anthropic_stream(events)

    @staticmethod
    def _parse_openai_stream(events: list, saw_done: bool = False) -> Optional[dict]:
        """Reconstruct OpenAI chat completion from stream chunks.

        Validation mirrors ``AsyncAPIClient.chat`` in ``model.py``:
        returns ``None`` if any of these signals an invalid stream:

        - mid-stream error event (``error`` field, ``type=='error'``,
          ``object=='error'``)
        - ``finish_reason == 'error'`` (except input-length errors,
          which carry useful trailing content and are kept)
        - no ``choices`` ever observed (``saw_choice``)
        - no ``data: [DONE]`` terminator (``saw_done``)
        - no terminal ``finish_reason`` on choices[0]
        """
        message: Dict = {
            'choices': [{'message': {'role': 'assistant', 'content': ''}, 'finish_reason': None}],
        }
        content_parts: List[str] = []
        reasoning_content_parts: List[str] = []
        tool_calls_map: Dict[int, dict] = {}
        function_call_data: Optional[dict] = None
        usage: dict = {}
        saw_choice = False

        for event in events:
            # In-stream error event — invalidate the whole trace.
            if event.get('error') is not None or event.get('type') == 'error' or event.get('object') == 'error':
                logger.warning(f"OpenAI stream error event: {event}")
                return None

            if event.get('id') and 'id' not in message:
                message['id'] = event['id']
            if event.get('model'):
                message['model'] = event['model']

            choices = event.get('choices', [])
            if choices:
                saw_choice = True

            for choice in choices:
                delta = choice.get('delta', {})

                if delta.get('content'):
                    content_parts.append(delta['content'])
                reasoning_delta = _extract_openai_reasoning_delta(delta)
                if reasoning_delta:
                    reasoning_content_parts.append(reasoning_delta)

                for tc_delta in delta.get('tool_calls') or []:
                    idx = tc_delta.get('index', 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            'id': tc_delta.get('id', ''),
                            'type': tc_delta.get('type', 'function'),
                            'function': {'name': '', 'arguments': ''},
                        }
                    tc = tool_calls_map[idx]
                    fn = tc_delta.get('function', {})
                    if fn.get('name'):
                        tc['function']['name'] += fn['name']
                    if fn.get('arguments'):
                        tc['function']['arguments'] += fn['arguments']
                    if tc_delta.get('id'):
                        tc['id'] = tc_delta['id']

                fc_delta = delta.get('function_call')
                if fc_delta:
                    if function_call_data is None:
                        function_call_data = {'name': '', 'arguments': ''}
                    if fc_delta.get('name'):
                        function_call_data['name'] += fc_delta['name']
                    if fc_delta.get('arguments'):
                        function_call_data['arguments'] += fc_delta['arguments']

                if choice.get('finish_reason'):
                    message['choices'][0]['finish_reason'] = choice['finish_reason']
                    if choice['finish_reason'] == 'error':
                        logger.warning(f"OpenAI finish_reason=error: {event}")
                        return None

            if event.get('usage'):
                usage = event['usage']

        # Stream completeness checks (mirror model.py).
        if not saw_choice:
            logger.warning('OpenAI stream ended without any choices')
            return None
        if not saw_done:
            logger.warning('OpenAI stream ended without [DONE] terminator')
            return None
        if not message['choices'][0].get('finish_reason'):
            logger.warning('OpenAI stream ended without terminal finish_reason')
            return None

        msg = message['choices'][0]['message']
        msg['content'] = ''.join(content_parts)
        if reasoning_content_parts:
            msg['reasoning_content'] = ''.join(reasoning_content_parts)
        if tool_calls_map:
            # Keep arguments as-string here (matching the non-stream raw
            # API shape). _handle_request does the single point of
            # JSON-deserialization for both stream and non-stream paths.
            msg['tool_calls'] = [tool_calls_map[idx] for idx in sorted(tool_calls_map)]
        if function_call_data:
            msg['function_call'] = function_call_data
        if usage:
            message['usage'] = usage
        return message

    @staticmethod
    def _parse_responses_stream(events: list) -> Optional[dict]:
        """Reconstruct an OpenAI Responses API result from stream events.

        Prefers the terminal ``response.completed`` event (carries the
        full response object). ``response.failed`` / ``response.incomplete``
        and any ``error`` event invalidate the trace — returns ``None``
        so the caller skips recording.
        """
        # Reject explicit failure / incomplete terminals.
        for event in events:
            evt_type = event.get('type', '')
            if evt_type in ('response.failed', 'response.incomplete', 'error'):
                logger.warning(f"Responses stream terminal failure: {evt_type}")
                return None

        # Fast path: completed event embeds the full response object
        for event in reversed(events):
            if event.get('type') == 'response.completed':
                resp = event.get('response')
                if resp is not None:
                    return resp

        # Fallback: rebuild from per-event deltas
        response: Dict = {}
        output_items: Dict[int, dict] = {}

        for event in events:
            evt_type = event.get('type', '')

            if evt_type in ('response.created', 'response.in_progress'):
                r = event.get('response') or {}
                for k, v in r.items():
                    if k != 'output' and k not in response:
                        response[k] = v

            elif evt_type == 'response.output_item.added':
                idx = event.get('output_index', len(output_items))
                item = event.get('item')
                if item is not None:
                    output_items[idx] = copy.deepcopy(item)

            elif evt_type == 'response.output_item.done':
                idx = event.get('output_index', 0)
                item = event.get('item')
                if item is not None:
                    output_items[idx] = copy.deepcopy(item)

            elif evt_type == 'response.content_part.added':
                idx = event.get('output_index', 0)
                content_idx = event.get('content_index', 0)
                part = event.get('part') or {}
                item = output_items.setdefault(idx, {'type': 'message', 'content': []})
                content = item.setdefault('content', [])
                while len(content) <= content_idx:
                    content.append({})
                content[content_idx] = copy.deepcopy(part)

            elif evt_type == 'response.output_text.delta':
                idx = event.get('output_index', 0)
                content_idx = event.get('content_index', 0)
                delta = event.get('delta', '')
                item = output_items.setdefault(idx, {'type': 'message', 'content': []})
                content = item.setdefault('content', [])
                while len(content) <= content_idx:
                    content.append({'type': 'output_text', 'text': ''})
                part = content[content_idx]
                part.setdefault('text', '')
                part['text'] += delta

            elif evt_type == 'response.output_text.done':
                idx = event.get('output_index', 0)
                content_idx = event.get('content_index', 0)
                text = event.get('text')
                if text is not None:
                    item = output_items.setdefault(idx, {'type': 'message', 'content': []})
                    content = item.setdefault('content', [])
                    while len(content) <= content_idx:
                        content.append({'type': 'output_text', 'text': ''})
                    content[content_idx]['text'] = text

            elif evt_type == 'response.refusal.delta':
                idx = event.get('output_index', 0)
                content_idx = event.get('content_index', 0)
                delta = event.get('delta', '')
                item = output_items.setdefault(idx, {'type': 'message', 'content': []})
                content = item.setdefault('content', [])
                while len(content) <= content_idx:
                    content.append({'type': 'refusal', 'refusal': ''})
                part = content[content_idx]
                part.setdefault('refusal', '')
                part['refusal'] += delta

            elif evt_type == 'response.function_call_arguments.delta':
                idx = event.get('output_index', 0)
                delta = event.get('delta', '')
                item = output_items.setdefault(idx, {'type': 'function_call'})
                item.setdefault('arguments', '')
                item['arguments'] += delta

            elif evt_type == 'response.function_call_arguments.done':
                idx = event.get('output_index', 0)
                args = event.get('arguments')
                if args is not None:
                    item = output_items.setdefault(idx, {'type': 'function_call'})
                    item['arguments'] = args

            elif evt_type == 'response.reasoning_summary_part.added':
                idx = event.get('output_index', 0)
                summary_idx = event.get('summary_index', 0)
                part = event.get('part') or {}
                item = output_items.setdefault(idx, {'type': 'reasoning', 'summary': []})
                summary = item.setdefault('summary', [])
                while len(summary) <= summary_idx:
                    summary.append({})
                summary[summary_idx] = copy.deepcopy(part)

            elif evt_type == 'response.reasoning_summary_text.delta':
                idx = event.get('output_index', 0)
                summary_idx = event.get('summary_index', 0)
                delta = event.get('delta', '')
                item = output_items.setdefault(idx, {'type': 'reasoning', 'summary': []})
                summary = item.setdefault('summary', [])
                while len(summary) <= summary_idx:
                    summary.append({'type': 'summary_text', 'text': ''})
                part = summary[summary_idx]
                part.setdefault('text', '')
                part['text'] += delta

        if not response and not output_items:
            return None

        response['output'] = [output_items[i] for i in sorted(output_items)]
        return response

    @staticmethod
    def _parse_anthropic_stream(events: list) -> Optional[dict]:
        """Reconstruct Anthropic message from stream events.

        Returns ``None`` if an ``error`` event is observed mid-stream
        or no ``message_stop`` is seen.
        """
        message: Dict = {}
        content_blocks: List[dict] = []
        current_block: dict = {}
        saw_message_stop = False

        for event in events:
            event_type = event.get('type', '')

            if event_type == 'error':
                logger.warning(f"Anthropic stream error event: {event}")
                return None

            if event_type == 'message_start':
                # Initial message metadata
                msg = event.get('message', {})
                message = {
                    'id': msg.get('id'),
                    'type': 'message',
                    'role': msg.get('role'),
                    'model': msg.get('model'),
                    'usage': msg.get('usage', {}),
                    'content': [],
                }

            elif event_type == 'content_block_start':
                # New content block
                current_block = dict(event.get('content_block', {}))

            elif event_type == 'content_block_delta':
                # Incremental content
                delta = event.get('delta', {})
                delta_type = delta.get('type', '')
                if delta_type == 'text_delta':
                    current_block.setdefault('text', '')
                    current_block['text'] += delta.get('text', '')
                elif delta_type == 'thinking_delta':
                    current_block.setdefault('thinking', '')
                    current_block['thinking'] += delta.get('thinking', '')
                elif delta_type == 'signature_delta':
                    # Sometimes reasoning blocks have signature chunk in anthropic stream
                    current_block.setdefault('signature', '')
                    current_block['signature'] += delta.get('signature', '')
                elif delta_type == 'input_json_delta':
                    current_block.setdefault('partial_json', '')
                    current_block['partial_json'] += delta.get('partial_json', '')

            elif event_type == 'content_block_stop':
                # Finalize current block
                if current_block:
                    # Parse partial_json into input for tool_use blocks
                    if 'partial_json' in current_block:
                        try:
                            current_block['input'] = json.loads(current_block.pop('partial_json'))
                        except json.JSONDecodeError:
                            current_block['input'] = current_block.pop('partial_json')
                    content_blocks.append(current_block)
                    current_block = {}

            elif event_type == 'message_delta':
                # Final metadata (stop_reason, usage delta)
                delta = event.get('delta', {})
                message['stop_reason'] = delta.get('stop_reason')
                # Merge usage delta
                usage_delta = event.get('usage', {})
                if usage_delta:
                    for k, v in usage_delta.items():
                        if isinstance(v, (int, float)):
                            message['usage'][k] = message['usage'].get(k, 0) + v
                        else:
                            message['usage'][k] = v

            elif event_type == 'message_stop':
                saw_message_stop = True

        if not saw_message_stop:
            logger.warning('Anthropic stream ended without message_stop')
            return None

        # Assemble final message
        message['content'] = content_blocks
        if _is_lmdeploy_input_length_error(message):
            # lmdeploy streams its INPUT_LENGTH_ERROR sentinel as ordinary
            # text/thinking deltas with stop_reason=stop_sequence; mirror the
            # non-stream drop.
            logger.warning(f"lmdeploy INPUT_LENGTH_ERROR in Anthropic stream: {message}")
            return None
        return message

    def get_messages(self) -> List[Dict[str, list]]:
        """Get the latest conversation messages for this session.
        If a sequence of messages is a prefix of another sequence and tools match, it will be filtered out.

        Prefix matching compares messages on a token-equivalence projection
        (``_canonical_msg``) rather than raw dicts, so non-tokenized metadata
        (e.g. cache breakpoints that the client moves each turn) does not stop a
        true prefix from matching. ``tools`` is still compared verbatim: it is
        rendered into the prompt, so different tools mean a different trajectory.

        Returns:
            List of message sequences.
        """
        records = self._records.get(self.session_id, [])
        if not records:
            self._log_request_processor_stats(0, 0)
            return []
        filtered = _maximal_prefix_records(records)
        self._log_request_processor_stats(len(records), len(filtered))
        return filtered

    def release_trace(self):
        """Clear recorded data for this session."""
        self._records.pop(self.session_id, None)
        if self.request_processor is not None:
            try:
                self.request_processor.reset()
            except Exception as exc:
                logger.warning('Failed to reset request processor %s: %s', type(self.request_processor).__name__, exc)
        self._request_index = 0
        self._request_processor_disabled = False
        self._request_processor_errors = 0
        self._processor_stats_logged = False
