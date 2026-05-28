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
    # OPENAI_BASE_URL = proxy.url
    #
    # Retrieve the deduplicated chat paths later:
    trajectories = proxy.get_messages()

    await proxy.stop()
"""

import copy
import json
import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

from lagent.utils import ctx_session_id, get_logger

logger = get_logger(__name__, 'info')


class SessionClient:
    """Lightweight HTTP proxy that records LLM request/response pairs.

    Args:
        real_api_key: The actual API key to use when forwarding requests.
        real_base_url: The actual LLM API base URL to forward to.
        port: Port to listen on. 0 means auto-assign.
    """

    def __init__(
        self,
        real_api_key: str,
        real_base_url: str,
        port: int = 0,
        session_id: Optional[str] = None,
        http_proxy: Optional[str] = None,
    ):
        self.real_api_key = real_api_key
        self.real_base_url = real_base_url.rstrip('/')
        self.port = port
        self.http_proxy = http_proxy
        self.session_id = session_id or ctx_session_id.get() or os.getenv('XTUNER_SESSION_ID') or str(uuid.uuid4().int)
        self._records: Dict[str, List[Dict[str, list]]] = defaultdict(list)
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    @property
    def url(self) -> str:
        """The proxy base URL that external agents should use."""
        return f"http://127.0.0.1:{self.port}"

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
        logger.info("LLM proxy stopped")

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler: extract session, forward, record, return."""
        # 1. Read request body
        request_body = await request.read()
        request_data = None
        try:
            request_data = json.loads(request_body) if request_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # 2. Inject session_id into request body
        if isinstance(request_data, dict):
            request_data['session_id'] = self.session_id
            request_body = json.dumps(request_data).encode('utf-8')

        # Detect provider format by inspecting the requested endpoint
        req_path = request.match_info['path']
        is_anthropic = req_path.endswith('/messages') or '/v1/messages' in req_path
        is_responses = req_path.endswith('/responses') or '/v1/responses' in req_path

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
                    # Stream response: collect chunks, forward as-is
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
                    async for chunk in resp.content.iter_any():
                        response_chunks.append(chunk)
                        await response.write(chunk)
                    await response.write_eof()
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

            # Additional fallback for Anthropic HTTP API error responses wrapped in proper JSON
            if response_data and response_data.get('type') == 'error':
                logger.error(f"Received error from provider: {response_data}")
                raise RuntimeError(f"Provider returned error: {response_data}")

            # Additional fallback for OpenAI HTTP API error responses
            if response_data and 'error' in response_data and isinstance(response_data['error'], dict):
                logger.error(f"Received error from provider: {response_data}")
                raise RuntimeError(f"Provider returned error: {response_data}")

        # 7. Record
        has_messages = isinstance(request_data, dict) and 'messages' in request_data
        has_input = isinstance(request_data, dict) and 'input' in request_data
        if has_messages or has_input:
            assistant_msg = None
            response_items: Optional[list] = None  # for Responses API: flat output items

            if is_anthropic:
                # Handle Anthropic trace format
                if response_data:
                    assistant_msg = {"role": response_data.get("role", "assistant")}
                    allowed_anthropic_fields = ["content"]
                    for field in allowed_anthropic_fields:
                        if response_data.get(field) is not None:
                            assistant_msg[field] = response_data[field]
            elif is_responses:
                # OpenAI Responses API: the conversation is a flat list of output
                # items (message / function_call / reasoning / ...). Store them
                # flattened so the client's next-turn `input` (which echoes them)
                # prefix-matches under get_messages() dedup. Convention:
                # `function_call.arguments` is stored as a dict, applied uniformly
                # to input- and output-side items below.
                if response_data and response_data.get('output') is not None:
                    response_items = copy.deepcopy(response_data.get('output', []))
            else:
                # Handle standard OpenAI trace format
                if response_data and response_data.get('choices'):
                    raw_msg = response_data['choices'][0].get('message')
                    if raw_msg:
                        assistant_msg = {"role": raw_msg.get("role", "assistant")}
                        # Only keep pure standard OpenAI fields to prevent contamination
                        allowed_fields = ["content", "tool_calls", "function_call", "refusal"]
                        # Support for o1 and future reasoning models natively
                        if "reasoning_content" in raw_msg:
                            allowed_fields.append("reasoning_content")

                        for field in allowed_fields:
                            if raw_msg.get(field) is not None:
                                val = copy.deepcopy(raw_msg[field])
                                if field == "tool_calls":
                                    for tc in val:
                                        if tc.get("type") == "function" and "function" in tc:
                                            args = tc["function"].get("arguments")
                                            if isinstance(args, str):
                                                try:
                                                    tc["function"]["arguments"] = json.loads(args)
                                                except json.JSONDecodeError:
                                                    pass
                                elif field == "function_call":
                                    args = val.get("arguments")
                                    if isinstance(args, str):
                                        try:
                                            val["arguments"] = json.loads(args)
                                        except json.JSONDecodeError:
                                            pass

                                assistant_msg[field] = val

            # Build the request side message list
            if has_messages:
                messages = list(request_data['messages'])
            else:
                raw_input = request_data['input']
                if isinstance(raw_input, str):
                    messages = [{"role": "user", "content": raw_input}]
                elif isinstance(raw_input, list):
                    messages = list(raw_input)
                else:
                    messages = []

            if response_items:
                messages.extend(response_items)
            elif assistant_msg:
                messages.append(assistant_msg)

            # Convention: function_call.arguments stored as dict (not JSON
            # string). Applied to all items uniformly — including input items
            # echoed from prior turns — so multi-turn prefix dedup in
            # get_messages() still works.
            if is_responses:
                for idx, item in enumerate(messages):
                    if isinstance(item, dict) and item.get('type') == 'function_call':
                        args = item.get('arguments')
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                            except json.JSONDecodeError:
                                continue
                            # Avoid mutating dicts shared with request_data['input']
                            messages[idx] = {**item, 'arguments': parsed}

            record_item = {"messages": messages, "tools": request_data.get("tools")}

            self._records[self.session_id].append(record_item)

            logger.debug(
                f"Updated messages for session {self.session_id}: {len(self._records[self.session_id])} traces total"
            )

        return response

    @staticmethod
    def _parse_stream_response(raw: bytes) -> Optional[dict]:
        """Parse SSE stream response to reconstruct the complete message.

        Supports both Anthropic and OpenAI streaming formats.
        """
        text = raw.decode('utf-8', errors='replace')
        events = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        if not events:
            return None

        # Detect format: Responses API events start with "response.";
        # OpenAI ChatCompletion has "choices"; Anthropic uses message_start / content_block_*
        first = events[0]
        evt_type = first.get('type', '')
        if evt_type.startswith('response.') or first.get('object') == 'response':
            return SessionClient._parse_responses_stream(events)
        if 'choices' in first or first.get('object') == 'chat.completion.chunk':
            return SessionClient._parse_openai_stream(events)
        return SessionClient._parse_anthropic_stream(events)

    @staticmethod
    def _parse_openai_stream(events: list) -> Optional[dict]:
        """Reconstruct OpenAI chat completion from stream chunks."""
        message = {
            'choices': [{'message': {'role': 'assistant', 'content': ''}}],
        }
        content_parts = []
        reasoning_content_parts = []
        tool_calls_map: Dict[int, dict] = {}  # index → {id, type, function}
        usage = {}

        for event in events:
            if event.get('id') and 'id' not in message:
                message['id'] = event['id']
            if event.get('model'):
                message['model'] = event['model']

            choices = event.get('choices', [])
            for choice in choices:
                delta = choice.get('delta', {})

                # Content
                if delta.get('content'):
                    content_parts.append(delta['content'])

                # Reasoning Content
                if delta.get('reasoning_content'):
                    reasoning_content_parts.append(delta['reasoning_content'])

                # Tool calls
                for tc_delta in delta.get('tool_calls') or []:
                    idx = tc_delta.get('index', 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            'id': tc_delta.get('id', ''),
                            'type': tc_delta.get('type', 'function'),
                            'function': {
                                'name': '',
                                'arguments': '',
                            },
                        }
                    tc = tool_calls_map[idx]
                    fn = tc_delta.get('function', {})
                    if fn.get('name'):
                        tc['function']['name'] += fn['name']
                    if fn.get('arguments'):
                        tc['function']['arguments'] += fn['arguments']
                    if tc_delta.get('id'):
                        tc['id'] = tc_delta['id']

                if choice.get('finish_reason'):
                    message['choices'][0]['finish_reason'] = choice['finish_reason']

            if event.get('usage'):
                usage = event['usage']

        msg = message['choices'][0]['message']
        msg['content'] = ''.join(content_parts)
        if reasoning_content_parts:
            msg['reasoning_content'] = ''.join(reasoning_content_parts)
        if tool_calls_map:
            msg['tool_calls'] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        if usage:
            message['usage'] = usage
        return message

    @staticmethod
    def _parse_responses_stream(events: list) -> Optional[dict]:
        """Reconstruct an OpenAI Responses API result from stream events.

        Prefers the terminal ``response.completed`` / ``response.failed`` /
        ``response.incomplete`` event, which carries the full response object.
        Falls back to incremental reconstruction from delta events when no
        terminal event was seen (e.g. truncated stream).
        """
        # Fast path: terminal events embed the full response object
        for event in reversed(events):
            evt_type = event.get('type', '')
            if evt_type in ('response.completed', 'response.failed', 'response.incomplete'):
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
        """Reconstruct Anthropic message from stream events."""
        message = {}
        content_blocks = []
        current_block = {}

        for event in events:
            event_type = event.get('type', '')

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

        # Assemble final message
        message['content'] = content_blocks
        return message

    def get_messages(self) -> List[Dict[str, list]]:
        """Get the latest conversation messages for this session.
        If a sequence of messages is a prefix of another sequence and tools match, it will be filtered out.

        Returns:
            List of message sequences.
        """
        records = self._records.get(self.session_id, [])
        if not records:
            return []

        filtered = []
        for i, rec_i in enumerate(records):
            is_prefix = False
            seq_i = rec_i.get("messages", [])
            tools_i = rec_i.get("tools")

            for j, rec_j in enumerate(records):
                if i == j:
                    continue

                seq_j = rec_j.get("messages", [])
                tools_j = rec_j.get("tools")

                # Check if tools are completely identical
                if tools_i != tools_j:
                    continue

                # If they are exactly identical, keep the one with the higher index
                if len(seq_i) == len(seq_j) and seq_i == seq_j:
                    if i < j:
                        is_prefix = True
                        break
                # If one is a strict prefix of the other, skip it
                elif len(seq_i) < len(seq_j) and seq_j[: len(seq_i)] == seq_i:
                    is_prefix = True
                    break

            if not is_prefix:
                filtered.append(rec_i)

        return filtered

    def release_trace(self):
        """Clear recorded data for this session."""
        self._records.pop(self.session_id, None)
