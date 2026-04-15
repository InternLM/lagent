"""LLM Proxy Recorder — lightweight HTTP proxy that records LLM
request/response pairs for trajectory capture.

The proxy intercepts all LLM API calls from external agents, records the
full request (including messages history) and response (including usage
and logprobs), then forwards the response unchanged.

Session routing is done via the API key: external agents receive a
synthetic key ``sk-proxy-{session_id}`` which the proxy uses to tag
records, then replaces with the real API key before forwarding.

Usage::

    proxy = LLMProxyRecorder(
        real_api_key="sk-ant-...",
        real_base_url="https://api.anthropic.com",
    )
    await proxy.start()
    # set env for external agent:
    #   OPENAI_BASE_URL=http://localhost:{proxy.port}/v1
    #   OPENAI_API_KEY=sk-proxy-{session_id}
    records = proxy.get_records(session_id)
    await proxy.stop()
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from aiohttp import ClientSession, web
except ImportError:
    ClientSession = None
    web = None

logger = logging.getLogger(__name__)

SESSION_KEY_PATTERN = re.compile(r'^sk-proxy-(.+)$')


class LLMProxyRecorder:
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
        http_proxy: Optional[str] = None,
    ):
        if web is None:
            raise ImportError(
                "aiohttp is required for LLMProxyRecorder. "
                "Install it with: pip install aiohttp"
            )
        self.real_api_key = real_api_key
        self.real_base_url = real_base_url.rstrip('/')
        self.port = port
        self.http_proxy = http_proxy
        self._records: Dict[str, List[dict]] = defaultdict(list)
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
        logger.info(f"LLMProxyRecorder started on port {self.port}")

    async def stop(self):
        """Stop the proxy HTTP server."""
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        logger.info("LLMProxyRecorder stopped")

    def _parse_session_id(self, auth_header: str) -> Optional[str]:
        """Extract session_id from Authorization header.

        Expects format: ``Bearer sk-proxy-{session_id}``
        """
        if not auth_header:
            return None
        token = auth_header.removeprefix('Bearer ').strip()
        # Also handle x-api-key style (Anthropic)
        match = SESSION_KEY_PATTERN.match(token)
        return match.group(1) if match else None

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler: extract session, forward, record, return."""
        # 1. Extract session from auth header
        auth = request.headers.get('Authorization', '')
        api_key = request.headers.get('x-api-key', '')
        session_id = self._parse_session_id(auth) or self._parse_session_id(api_key)

        # 2. Read request body
        request_body = await request.read()
        request_data = None
        try:
            request_data = json.loads(request_body) if request_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # 3. Build forwarding headers — replace auth with real key
        forward_headers = dict(request.headers)
        forward_headers.pop('Host', None)
        forward_headers.pop('host', None)
        if 'Authorization' in forward_headers:
            forward_headers['Authorization'] = f'Bearer {self.real_api_key}'
        if 'x-api-key' in forward_headers:
            forward_headers['x-api-key'] = self.real_api_key

        # 4. Forward to real LLM
        # Build target URL, avoiding path duplication
        # e.g. real_base_url="http://api.com/v1", path="/v1/chat/completions"
        # should produce "http://api.com/v1/chat/completions" not "http://api.com/v1/v1/..."
        req_path = request.match_info['path']
        from urllib.parse import urlparse
        base_parsed = urlparse(self.real_base_url)
        base_path = base_parsed.path.rstrip('/')
        if req_path.startswith(base_path.lstrip('/')):
            # Path already includes the base path prefix, use as-is
            target_url = f"{base_parsed.scheme}://{base_parsed.netloc}/{req_path}"
        else:
            target_url = f"{self.real_base_url}/{req_path}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = request_data.get('stream', False) if request_data else False

        async with ClientSession() as client:
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
                            k: v for k, v in resp.headers.items()
                            if k.lower() not in ('transfer-encoding', 'content-length',
                                                'content-encoding')
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
                            k: v for k, v in resp.headers.items()
                            if k.lower() not in ('transfer-encoding', 'content-length',
                                                'content-encoding')
                        },
                        body=raw_response,
                    )

        # 5. Parse response for recording
        response_data = None
        if is_stream:
            # Parse SSE stream to extract final data
            response_data = self._parse_stream_response(raw_response)
        else:
            try:
                response_data = json.loads(raw_response)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # 6. Record
        if session_id and request_data:
            record = {
                'timestamp': datetime.now().isoformat(),
                'request': request_data,
                'response': response_data,
                'path': request.path,
                'method': request.method,
                'stream': is_stream,
            }
            self._records[session_id].append(record)
            logger.debug(
                f"Recorded LLM call for session {session_id}: "
                f"{request.path} ({len(self._records[session_id])} total)"
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

        # Detect format: OpenAI has "choices", Anthropic has "type"
        first = events[0]
        if 'choices' in first or first.get('object') == 'chat.completion.chunk':
            return LLMProxyRecorder._parse_openai_stream(events)
        else:
            return LLMProxyRecorder._parse_anthropic_stream(events)

    @staticmethod
    def _parse_openai_stream(events: list) -> Optional[dict]:
        """Reconstruct OpenAI chat completion from stream chunks."""
        message = {
            'choices': [{'message': {'role': 'assistant', 'content': ''}}],
        }
        content_parts = []
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

                # Tool calls
                for tc_delta in delta.get('tool_calls', []):
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
        if tool_calls_map:
            msg['tool_calls'] = [
                tool_calls_map[i] for i in sorted(tool_calls_map)
            ]
        if usage:
            message['usage'] = usage
        return message

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
                elif delta_type == 'input_json_delta':
                    current_block.setdefault('partial_json', '')
                    current_block['partial_json'] += delta.get('partial_json', '')

            elif event_type == 'content_block_stop':
                # Finalize current block
                if current_block:
                    # Parse partial_json into input for tool_use blocks
                    if 'partial_json' in current_block:
                        try:
                            current_block['input'] = json.loads(
                                current_block.pop('partial_json')
                            )
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

    def get_records(self, session_id: str) -> List[dict]:
        """Get all recorded LLM call records for a session.

        Each record contains:
            - timestamp: ISO 8601 timestamp
            - request: Full request body (messages, tools, etc.)
            - response: Full response body (choices, usage, etc.)
            - path: API path
            - method: HTTP method
            - stream: Whether streaming was used

        Args:
            session_id: The session identifier.

        Returns:
            List of record dicts, ordered by timestamp.
        """
        return list(self._records.get(session_id, []))

    # ── Standardization ────────────────────────────────────────

    @staticmethod
    def normalize_record(record: dict) -> dict:
        """Normalize a raw proxy record into a standard training format.

        Handles both Anthropic and OpenAI API formats, strips billing
        headers and other noise, and produces a uniform structure::

            {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "...",
                     "reasoning_content": "...", "extra_info": {...}},
                    ...
                ],
                "tools": [...],
                "meta": {
                    "model": "...",
                    "usage": {...},
                    "stop_reason": "...",
                    "timestamp": "...",
                },
                "response": {
                    "role": "assistant",
                    "content": "...",
                    "reasoning_content": "...",
                    "extra_info": {"usage": {...}, "model": "...", ...},
                },
            }
        """
        req = record.get('request', {})
        resp = record.get('response') or {}

        # ── Normalize system prompt ──
        system = req.get('system')
        system_text = None
        if system:
            if isinstance(system, list):
                # Anthropic: list of content blocks, skip billing headers
                parts = []
                for block in system:
                    if not isinstance(block, dict):
                        continue
                    text = block.get('text', '')
                    # Skip billing/tracking headers
                    if text.startswith('x-anthropic-billing-header'):
                        continue
                    parts.append(text)
                system_text = '\n'.join(parts) if parts else None
            elif isinstance(system, str):
                system_text = system

        # ── Normalize messages ──
        messages = []
        if system_text:
            messages.append({'role': 'system', 'content': system_text})

        for msg in req.get('messages', []):
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Flatten content blocks to text
            if isinstance(content, list):
                text_parts = []
                reasoning_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        text_parts.append(str(block))
                        continue
                    btype = block.get('type', '')
                    if btype == 'text':
                        text_parts.append(block.get('text', ''))
                    elif btype in ('thinking', 'reasoning'):
                        reasoning_parts.append(block.get('thinking', block.get('text', '')))
                    elif btype == 'tool_use':
                        text_parts.append(f"[tool_use: {block.get('name', '')}]")
                    elif btype == 'tool_result':
                        text_parts.append(f"[tool_result: {str(block.get('content', ''))[:200]}]")
                    else:
                        text_parts.append(block.get('text', str(block)))

                norm_msg = {'role': role, 'content': '\n'.join(text_parts)}
                if reasoning_parts:
                    norm_msg['reasoning_content'] = '\n'.join(reasoning_parts)
            else:
                norm_msg = {'role': role, 'content': str(content)}

            messages.append(norm_msg)

        # ── Normalize tools ──
        tools = req.get('tools')

        # ── Normalize response ──
        resp_content = ''
        resp_reasoning = ''
        resp_extra = {}

        # Anthropic format
        resp_blocks = resp.get('content', [])
        if isinstance(resp_blocks, list):
            text_parts = []
            reasoning_parts = []
            for block in resp_blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get('type', '')
                if btype == 'text':
                    text_parts.append(block.get('text', ''))
                elif btype in ('thinking', 'reasoning'):
                    reasoning_parts.append(block.get('thinking', block.get('text', '')))
                elif btype == 'tool_use':
                    text_parts.append(f"[tool_use: {block.get('name', '')}]")
            resp_content = '\n'.join(text_parts)
            resp_reasoning = '\n'.join(reasoning_parts)

        # OpenAI format
        choices = resp.get('choices', [])
        if choices and not resp_content:
            choice = choices[0]
            msg = choice.get('message', {})
            resp_content = msg.get('content', '') or ''
            # Handle tool_calls
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                tc_strs = []
                for tc in tool_calls:
                    fn = tc.get('function', {})
                    tc_strs.append(f"[tool_call: {fn.get('name', '')}({fn.get('arguments', '')[:100]})]")
                if not resp_content:
                    resp_content = '\n'.join(tc_strs)
                resp_extra['tool_calls'] = tool_calls
            resp_extra['finish_reason'] = choice.get('finish_reason')

        # Usage (both formats)
        usage = resp.get('usage', {})
        if usage:
            resp_extra['usage'] = usage
        if resp.get('model'):
            resp_extra['model'] = resp['model']
        if resp.get('stop_reason'):
            resp_extra['stop_reason'] = resp['stop_reason']

        # Meta
        model = req.get('model') or resp.get('model')
        meta = {
            'model': model,
            'usage': usage,
            'timestamp': record.get('timestamp'),
        }
        if resp.get('stop_reason'):
            meta['stop_reason'] = resp['stop_reason']
        if choices and choices[0].get('finish_reason'):
            meta['stop_reason'] = choices[0]['finish_reason']

        response_msg = {'role': 'assistant', 'content': resp_content}
        if resp_reasoning:
            response_msg['reasoning_content'] = resp_reasoning
        if resp_extra:
            response_msg['extra_info'] = resp_extra

        return {
            'messages': messages,
            'tools': tools,
            'meta': meta,
            'response': response_msg,
        }

    def get_normalized_records(self, session_id: str) -> List[dict]:
        """Get all records in normalized format."""
        return [self.normalize_record(r) for r in self.get_records(session_id)]

    # ── Chain Rebuilding ──────────────────────────────────────

    def rebuild_chains(self, session_id: str) -> List[List[dict]]:
        """Rebuild conversation chains from normalized records.

        Two consecutive records belong to the same chain if:
        1. Messages count grew (history is appending)
        2. The previous response text appears in current messages

        Args:
            session_id: The session identifier.

        Returns:
            List of chains. Each chain is a list of normalized records.
        """
        records = self.get_normalized_records(session_id)
        if not records:
            return []

        chains: List[List[dict]] = []
        current_chain: List[dict] = [records[0]]

        for prev, curr in zip(records, records[1:]):
            prev_msgs = prev['messages']
            curr_msgs = curr['messages']

            msgs_grew = len(curr_msgs) > len(prev_msgs)

            prev_response_text = prev['response'].get('content', '')[:200]
            has_prev_response = (
                prev_response_text
                and any(
                    m.get('role') == 'assistant'
                    and prev_response_text in m.get('content', '')
                    for m in curr_msgs
                )
            )

            if msgs_grew and has_prev_response:
                current_chain.append(curr)
            else:
                chains.append(current_chain)
                current_chain = [curr]

        chains.append(current_chain)
        return chains

    # ── Training Sample Export ────────────────────────────────

    def to_training_samples(self, session_id: str) -> List[dict]:
        """Convert recorded LLM calls into SFT/RL training samples.

        Each chain produces one sample::

            {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "...",
                     "reasoning_content": "...",
                     "extra_info": {"usage": {...}, "model": "..."}},
                    ...
                ],
                "tools": [...],
                "meta": {
                    "num_calls": 3,
                    "model": "...",
                    "total_usage": {...},
                },
            }
        """
        chains = self.rebuild_chains(session_id)
        samples = []

        for chain in chains:
            if not chain:
                continue

            last = chain[-1]

            # Take the last record's messages (most complete history)
            # + append the last response
            messages = list(last['messages'])

            # Attach extra_info to assistant messages by matching response text
            response_extra_map = {}
            for rec in chain:
                resp = rec['response']
                text = resp.get('content', '')[:200]
                if text:
                    extra = dict(resp.get('extra_info', {}))
                    response_extra_map[text] = extra

            for msg in messages:
                if msg.get('role') == 'assistant':
                    text = msg.get('content', '')[:200]
                    if text in response_extra_map:
                        msg['extra_info'] = response_extra_map[text]

            # Append final response as the last assistant message
            last_resp = dict(last['response'])
            messages.append(last_resp)

            # Aggregate usage
            total_usage = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
            }
            for rec in chain:
                u = rec['meta'].get('usage', {})
                total_usage['total_input_tokens'] += u.get(
                    'input_tokens', u.get('prompt_tokens', 0))
                total_usage['total_output_tokens'] += u.get(
                    'output_tokens', u.get('completion_tokens', 0))

            sample = {
                'messages': messages,
                'tools': last.get('tools'),
                'meta': {
                    'num_calls': len(chain),
                    'model': last['meta'].get('model'),
                    'total_usage': total_usage,
                },
            }
            samples.append(sample)

        return samples

    def clear(self, session_id: Optional[str] = None):
        """Clear recorded data.

        Args:
            session_id: Clear only this session. If None, clear all.
        """
        if session_id:
            self._records.pop(session_id, None)
        else:
            self._records.clear()
