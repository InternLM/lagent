"""LLM Proxy Recorder — a lightweight HTTP proxy (`SessionClient`) for intercepting,
translating, and recording LLM request/response trajectories.

This proxy intercepts OpenAI-schema API calls from external agents, forwards them
to the actual model backend (with real-time schema translation if necessary),
and quietly records the full conversation history (trajectories).

Key Features:
- **Format Translation**: Dynamically converts standard OpenAI requests into
  Anthropic format (if the endpoint indicates Anthropic, e.g., `/v1/messages`),
  and translates the Anthropic API responses/streams back to OpenAI format.
- **Reasoning/Thinking Support**: Fully supports capturing and preserving Claude's
  extended `thinking` blocks, securely managing internal signature alignment across
  multi-turn chats so reasoning context is not lost.
- **Trajectory Recording**: Merges and retains conversation turns into `_records`.
  Interrupted or duplicate prefix traces are intelligently filtered when calling
  `get_messages()`, returning pure OpenAI message schemas.

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

import asyncio
import copy
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


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
        self.session_id = session_id or os.getenv('XTUNER_SESSION_ID') or uuid.uuid4().hex
        self._records: Dict[str, List[List[dict]]] = defaultdict(list)
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

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler: extract session, forward, record, return."""
        # 1. Read request body
        request_body = await request.read()
        request_data = None
        try:
            request_data = json.loads(request_body) if request_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # === 3. Convert OpenAI request to Provider Request ===
        # Detect if we should use Anthropic format by checking the requested endpoint
        req_path = request.match_info['path']
        is_anthropic = req_path.endswith('/messages') or '/v1/messages' in req_path

        # By default we assume the incoming request is standard OpenAI
        provider_request_data = copy.deepcopy(request_data) if request_data else {}

        if is_anthropic and provider_request_data:
            # We need to map OpenAI request -> Anthropic request
            provider_request_data = self._convert_openai_to_anthropic_req(provider_request_data)
            request_body = json.dumps(provider_request_data).encode('utf-8')

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

        # === Convert Provider Response to OpenAI Response ===
        if is_anthropic and response_data:
            response_data = self._convert_anthropic_to_openai_resp(response_data)

        # 7. Record
        if request_data and 'messages' in request_data:
            # At this point, response_data is guaranteed to be in OpenAI format
            assistant_msg = None
            if response_data and 'choices' in response_data and response_data['choices']:
                raw_msg = response_data['choices'][0].get('message')
                if raw_msg:
                    # Only keep pure standard OpenAI fields to prevent contamination
                    # standard fields: role, content, tool_calls, function_call, refusal, reasoning_content
                    assistant_msg = {"role": raw_msg.get("role", "assistant")}
                    if "content" in raw_msg and raw_msg["content"] is not None:
                        assistant_msg["content"] = raw_msg["content"]
                    if "reasoning_content" in raw_msg and raw_msg["reasoning_content"] is not None:
                        assistant_msg["reasoning_content"] = raw_msg["reasoning_content"]
                    if "reasoning_signature" in raw_msg and raw_msg["reasoning_signature"] is not None:
                        assistant_msg["reasoning_signature"] = raw_msg["reasoning_signature"]
                    if "tool_calls" in raw_msg and raw_msg["tool_calls"] is not None:
                        assistant_msg["tool_calls"] = raw_msg["tool_calls"]
                    if "function_call" in raw_msg and raw_msg["function_call"] is not None:
                        assistant_msg["function_call"] = raw_msg["function_call"]
                    if "refusal" in raw_msg and raw_msg["refusal"] is not None:
                        assistant_msg["refusal"] = raw_msg["refusal"]

            # Keep the latest conversation history
            messages = list(request_data['messages'])
            if assistant_msg:
                messages.append(assistant_msg)
            self._records[self.session_id].append(messages)

            logger.debug(
                f"Updated messages for session {self.session_id}: {len(self._records[self.session_id])} traces total"
            )

        return response

    @staticmethod
    def _convert_openai_to_anthropic_req(openai_req: dict) -> dict:
        """Convert standard OpenAI request format to Anthropic format."""
        anthropic_req = {
            "model": openai_req.get("model", ""),
            "messages": [],
        }

        if "max_tokens" in openai_req:
            anthropic_req["max_tokens"] = openai_req["max_tokens"]
        elif "max_completion_tokens" in openai_req:
            anthropic_req["max_tokens"] = openai_req["max_completion_tokens"]
        else:
            anthropic_req["max_tokens"] = 4096  # Anthropic requires max_tokens

        # Pass reasoning capabilities for supported models
        if "thinking" in openai_req:
            anthropic_req["thinking"] = openai_req["thinking"]

        if "temperature" in openai_req:
            anthropic_req["temperature"] = openai_req["temperature"]

        if "top_p" in openai_req:
            anthropic_req["top_p"] = openai_req["top_p"]

        if "top_k" in openai_req:
            anthropic_req["top_k"] = openai_req["top_k"]

        if "stop" in openai_req:
            if isinstance(openai_req["stop"], str):
                anthropic_req["stop_sequences"] = [openai_req["stop"]]
            elif isinstance(openai_req["stop"], list):
                anthropic_req["stop_sequences"] = openai_req["stop"]

        if "stream" in openai_req:
            anthropic_req["stream"] = openai_req["stream"]

        # Convert Messages
        system_prompts = []
        for msg in openai_req.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_prompts.append(content)
                continue

            anthropic_msg = {"role": role, "content": []}

            # Handle thinking / reasoning blocks for assistant
            if role == "assistant" and msg.get("reasoning_content"):
                if msg.get("reasoning_signature"):
                    anthropic_msg["content"].append(
                        {
                            "type": "thinking",
                            "thinking": msg.get("reasoning_content"),
                            "signature": msg.get("reasoning_signature"),
                        }
                    )
                # If there's reasoning_content but no signature, Anthropic API will reject it if passed as 'thinking' block.
                # In that case, we can't safely inject it natively to Claude API without breaking the call.

            # Content could be a string or list (vision, etc). Here we handle common cases.
            if isinstance(content, str):
                if content:
                    anthropic_msg["content"].append({"type": "text", "text": content})
            else:
                # If it's already a list (like OpenAI vision), map accordingly.
                # Avoid overwriting the reasoning block we just appended
                if isinstance(content, list):
                    anthropic_msg["content"].extend(content)
                else:
                    anthropic_msg["content"] = content

            # Handle OpenAI tool calls -> Anthropic tool_use
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    anthropic_msg["content"].append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        }
                    )

            # Handle OpenAI tool response -> Anthropic tool_result
            if role == "tool":
                anthropic_msg["role"] = "user"  # Anthropic uses 'user' for tool results
                anthropic_msg["content"].append(
                    {"type": "tool_result", "tool_use_id": msg.get("tool_call_id"), "content": msg.get("content", "")}
                )

            anthropic_req["messages"].append(anthropic_msg)

        if system_prompts:
            anthropic_req["system"] = "\n".join(system_prompts)

        # Convert Tools
        if "tools" in openai_req:
            anthropic_req["tools"] = []
            for tool in openai_req["tools"]:
                if tool.get("type") == "function":
                    fn = tool.get("function", {})
                    anthropic_req["tools"].append(
                        {
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {}),
                        }
                    )

        # Convert tool_choice
        if "tool_choice" in openai_req:
            tc = openai_req["tool_choice"]
            if tc == "auto":
                anthropic_req["tool_choice"] = {"type": "auto"}
            elif tc == "required":
                anthropic_req["tool_choice"] = {"type": "any"}
            elif isinstance(tc, dict) and tc.get("type") == "function" and "function" in tc:
                anthropic_req["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}

        # Remove empty content arrays if they got created
        for msg in anthropic_req["messages"]:
            if (
                isinstance(msg["content"], list)
                and len(msg["content"]) == 1
                and msg["content"][0].get("type") == "text"
            ):
                msg["content"] = msg["content"][0]["text"]  # Simplify

        return anthropic_req

    @staticmethod
    def _convert_anthropic_to_openai_resp(anthro_resp: dict) -> dict:
        """Convert Anthropic response format to OpenAI format."""
        openai_resp = {
            "id": anthro_resp.get("id", ""),
            "model": anthro_resp.get("model", ""),
            "choices": [{"index": 0, "message": {"role": "assistant"}}],
            "usage": {},
        }

        message = openai_resp["choices"][0]["message"]
        content_text = ""
        tool_calls = []

        # Handle anthro_resp (which could be the parsed stream output or direct HTTP response)
        content_blocks = anthro_resp.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Ensure input is a dictionary before dumping
                input_data = block.get("input", {})
                if isinstance(input_data, str):
                    try:
                        input_data = json.loads(input_data)
                    except:
                        pass
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(input_data) if isinstance(input_data, dict) else str(input_data),
                        },
                    }
                )
            elif block.get("type") in ("thinking", "reasoning"):
                reasoning_text = block.get("thinking", block.get("text", ""))
                assistant_msg = openai_resp["choices"][0]["message"]
                assistant_msg["reasoning_content"] = assistant_msg.get("reasoning_content", "") + reasoning_text

                if block.get("signature"):
                    assistant_msg["reasoning_signature"] = block.get("signature")

        message["content"] = content_text if content_text else None
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Handle stop reason
        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        anthro_stop = anthro_resp.get("stop_reason")
        openai_resp["choices"][0]["finish_reason"] = stop_reason_map.get(anthro_stop, anthro_stop)

        # Handle Usage
        if "usage" in anthro_resp:
            openai_resp["usage"] = {
                "prompt_tokens": anthro_resp["usage"].get("input_tokens", 0),
                "completion_tokens": anthro_resp["usage"].get("output_tokens", 0),
                "total_tokens": anthro_resp["usage"].get("input_tokens", 0)
                + anthro_resp["usage"].get("output_tokens", 0),
            }

        return openai_resp

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
            return SessionClient._parse_openai_stream(events)
        return SessionClient._parse_anthropic_stream(events)

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
            msg['tool_calls'] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
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

    def get_messages(self) -> List[List[dict]]:
        """Get the latest conversation messages in OpenAI format for this session.
        If a sequence of messages is a prefix of another sequence, it will be filtered out.

        Returns:
            List of message sequences.
        """
        records = self._records.get(self.session_id, [])
        if not records:
            return []

        filtered = []
        for i, seq_i in enumerate(records):
            is_prefix = False
            for j, seq_j in enumerate(records):
                if i == j:
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
                filtered.append(seq_i)

        return filtered

    def release_trace(self):
        """Clear recorded data for this session."""
        self._records.pop(self.session_id, None)


if __name__ == '__main__':

    async def _test_debug_openai():
        logging.basicConfig(level=logging.DEBUG)

        # 1. Start the proxy
        proxy = SessionClient(
            real_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),  # Provide a valid OpenAI key if needed
            real_base_url="http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
            session_id="test_session_123",
        )
        await proxy.start()

        # 2. Simulate an Agent sending an OpenAI-format request to the proxy
        dummy_payload = {
            "model": "agentic_rl_qwen35a3b_service",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."},
            ],
            "stream": False,
        }

        print(f"\n--- Sending request to Proxy at {proxy.url} ---")
        async with aiohttp.ClientSession() as session:
            try:
                # Send to proxy URL, the proxy forwards it to real_base_url
                async with session.post(
                    f"{proxy.url}/chat/completions",
                    json=dummy_payload,
                    headers={"Authorization": "Bearer sk-proxy-test"},
                ) as resp:
                    print(f"Proxy returned status: {resp.status}")
                    result = await resp.json()
                    print("Response received from LLM:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Request failed (is the target server running?): {e}")

        # 3. Check what the proxy recorded
        print("\n--- Recorded Context (OpenAI Format) ---")
        messages = proxy.get_messages()
        print(json.dumps(messages, indent=2, ensure_ascii=False))

        # Clean up
        proxy.release_trace()
        await proxy.stop()

    async def _test_debug_claude():
        logging.basicConfig(level=logging.DEBUG)

        proxy = SessionClient(
            real_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN", "EMPTY"),  # Provide a valid Claude key if needed
            real_base_url=os.getenv('ANTHROPIC_BASE_URL'),  # The Claude v1 proxy target
            session_id="test_claude_123",
            http_proxy=os.getenv("HTTP_PROXY"),
        )
        await proxy.start()

        # Simulate sending OpenAI schema request, but pointing to Anthropic endpoint
        dummy_payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."},
            ],
            "stream": True,  # Let's test streaming capabilities too
            "max_completion_tokens": 10000,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }

        print(f"\n--- Sending request to Proxy Claude at {proxy.url} ---")
        async with aiohttp.ClientSession() as session:
            try:
                # Assuming the external agent posts to /v1/messages since we trigger Anthropic handling by endpoint now
                async with session.post(
                    f"{proxy.url}/v1/messages",
                    json=dummy_payload,
                    headers={"Authorization": "Bearer sk-proxy-test", "x-api-key": "sk-proxy-test"},
                ) as resp:
                    print(f"Proxy returned status: {resp.status}")

                    if dummy_payload["stream"]:
                        print("Stream response chunks:")
                        async for chunk in resp.content.iter_any():
                            print(chunk.decode("utf-8", errors="replace"), end="")
                        print("\nStream finished.")
                    else:
                        result = await resp.json()
                        print("Response received from LLM:")
                        print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Request failed: {e}")

        # Check proxy records
        print("\n--- Recorded Context (OpenAI Format) ---")
        messages = proxy.get_messages()
        print(json.dumps(messages, indent=2, ensure_ascii=False))

        proxy.release_trace()
        await proxy.stop()

    asyncio.run(_test_debug_openai())
    # asyncio.run(_test_debug_claude())
