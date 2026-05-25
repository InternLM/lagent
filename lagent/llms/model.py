import asyncio
import json
import random
import traceback
from typing import Dict, List, Optional, TypedDict

import aiohttp

from lagent.llms.openai import AsyncGPTAPI
from lagent.utils import ctx_session_id, get_logger

logger = get_logger()


class SampleParameters(TypedDict):
    temperature: float
    top_p: float
    top_k: int


class ModelConfig(TypedDict):
    model: str
    base_url: str | List[str]
    api_key: Optional[str]


class AsyncAPIClient(AsyncGPTAPI):
    def __init__(
        self,
        model: ModelConfig,
        sample_params: SampleParameters,
        timeout: int = 600,
        max_retry: int = 1,
        sleep_interval: int = 5,
        extra_body: Optional[dict] = None,
        session_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_urls = [
            url.rstrip('/') + '/chat/completions'
            for url in (model['base_url'] if isinstance(model['base_url'], list) else [model['base_url']])
        ]
        self.api_key = model.get("api_key", "")
        self.proxy = model.get("proxy")
        self.model_name = model["model"]
        self.sample_params = sample_params
        self.max_retry = max_retry
        self.timeout = timeout
        self.sleep_interval = sleep_interval
        self.extra_body = extra_body
        self.session_id = session_id or ctx_session_id.get()

    async def chat(self, messages: List[dict], tools=None, **gen_params) -> dict:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        reasoning_effort = self.sample_params.get("reasoning_effort")
        payload: Dict = dict(
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=self.sample_params.get("temperature", 0.7),
            top_p=self.sample_params.get("top_p", 1.0),
            max_tokens=self.sample_params.get("max_tokens", 64 * 1024),
        )
        if tools is not None:
            payload["tools"] = tools
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if self.extra_body:
            payload.update(self.extra_body)
        if self.session_id is not None:
            payload["session_id"] = self.session_id

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        # def _error_completion(content: str) -> dict:
        #     return {
        #         "id": "error",
        #         "object": "chat.completion",
        #         "created": 0,
        #         "model": self.model_name,
        #         "choices": [
        #             {
        #                 "index": 0,
        #                 "message": {"role": "assistant", "content": content},
        #                 "finish_reason": "failed",
        #             }
        #         ],
        #         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        #     }

        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for attempt in range(self.max_retry):
                url = random.choice(self.base_urls)
                try:
                    message_data = {
                        "id": "",
                        "object": "chat.completion",
                        "created": 0,
                        "model": self.model_name,
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}
                        ],
                    }
                    content_parts = []
                    reasoning_content_parts = []
                    tool_calls_map = {}
                    function_call_data = None
                    usage = {}

                    async with session.post(url, json=payload, headers=headers, proxy=self.proxy) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}: {text}")

                        async for line in resp.content:
                            if line:
                                decoded = line.decode("utf-8").strip()
                                if not decoded:
                                    continue
                                if decoded.startswith("data: "):
                                    data_str = decoded[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        if "id" in data and not message_data["id"]:
                                            message_data["id"] = data["id"]
                                        if "created" in data and not message_data["created"]:
                                            message_data["created"] = data["created"]
                                        if "model" in data:
                                            message_data["model"] = data["model"]

                                        for choice in data.get("choices", []):
                                            delta = choice.get("delta", {})

                                            if "content" in delta and delta["content"]:
                                                content_parts.append(delta["content"])

                                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                                reasoning_content_parts.append(delta["reasoning_content"])

                                            for tc_delta in delta.get("tool_calls") or []:
                                                idx = tc_delta.get("index", 0)
                                                if idx not in tool_calls_map:
                                                    tool_calls_map[idx] = {
                                                        "id": tc_delta.get("id", ""),
                                                        "type": tc_delta.get("type", "function"),
                                                        "function": {"name": "", "arguments": ""},
                                                    }
                                                tc = tool_calls_map[idx]
                                                fn = tc_delta.get("function", {})
                                                if fn.get("name"):
                                                    tc["function"]["name"] += fn["name"]
                                                if fn.get("arguments"):
                                                    tc["function"]["arguments"] += fn["arguments"]
                                                if tc_delta.get("id"):
                                                    tc["id"] = tc_delta["id"]

                                            fc_delta = delta.get("function_call")
                                            if fc_delta:
                                                if function_call_data is None:
                                                    function_call_data = {"name": "", "arguments": ""}
                                                if fc_delta.get("name"):
                                                    function_call_data["name"] += fc_delta["name"]
                                                if fc_delta.get("arguments"):
                                                    function_call_data["arguments"] += fc_delta["arguments"]

                                            if "finish_reason" in choice and choice["finish_reason"]:
                                                message_data["choices"][0]["finish_reason"] = choice["finish_reason"]

                                        if "usage" in data and data["usage"]:
                                            usage = data["usage"]
                                    except json.JSONDecodeError:
                                        pass

                    msg = message_data["choices"][0]["message"]
                    msg["content"] = "".join(content_parts)
                    if reasoning_content_parts:
                        msg["reasoning_content"] = "".join(reasoning_content_parts)

                    if tool_calls_map:
                        tool_calls = []
                        for idx in sorted(tool_calls_map.keys()):
                            tc = tool_calls_map[idx]
                            args = tc["function"].get("arguments")
                            if isinstance(args, str):
                                try:
                                    tc["function"]["arguments"] = json.loads(args)
                                except json.JSONDecodeError:
                                    pass
                            tool_calls.append(tc)
                        msg["tool_calls"] = tool_calls

                    if function_call_data:
                        args = function_call_data.get("arguments")
                        if isinstance(args, str):
                            try:
                                function_call_data["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                pass
                        msg["function_call"] = function_call_data

                    if usage:
                        message_data["usage"] = usage

                    return message_data

                except asyncio.TimeoutError as e:
                    logger.error(f"LLM Call Timeout: {e}")
                    if attempt == self.max_retry - 1:
                        logger.error(f"LLM Call Error: {e}{traceback.format_exc()}")
                        raise e
                    await asyncio.sleep(self.sleep_interval)
                except Exception as e:
                    for val in [
                        "用户额度不足",
                        "剩余额度",
                        "TimeoutError",
                        "litellm.BadRequestError",
                        "litellm.APIError: APIError",
                        "Failed to parse fc related info to json format!",
                        "Error code",
                    ]:
                        if val in str(e):
                            traceback.print_exc()
                            logger.error(f"[Retry] {attempt} LLM Call Error: {e}")
                            if attempt == self.max_retry - 1:
                                logger.error(f"LLM Call Error: {e}{traceback.format_exc()}")
                                raise e
                            await asyncio.sleep(self.sleep_interval)
                            break
                    else:
                        logger.error(f"LLM Call Error: {e}{traceback.format_exc()}")
                        raise e


if __name__ == '__main__':
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_temperature',
                'description': 'Get current temperature at a location.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The location to get the temperature for, in the format \'City, State, Country\'.',
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit'],
                            'description': 'The unit to return the temperature in. Defaults to \'celsius\'.',
                        },
                    },
                    'required': ['location'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_temperature_date',
                'description': 'Get temperature at a location and date.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The location to get the temperature for, in the format \'City, State, Country\'.',
                        },
                        'date': {
                            'type': 'string',
                            'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.',
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit'],
                            'description': 'The unit to return the temperature in. Defaults to \'celsius\'.',
                        },
                    },
                    'required': ['location', 'date'],
                },
            },
        },
    ]

    messages = [
        {
            'role': 'user',
            'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?',
        }
    ]
    messages = [{'role': 'user', 'content': '上海温度'}]
    # model_name = "claude-opus-4-6"
    # api_base = "http://35.220.164.252:3888/v1"
    # api_key = ""
    # proxy = "http://100.100.72.89:8899"
    # extra_body = {}

    extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    model_name = "/mnt/shared-storage-user/llmit1/user/liujiangning/exp/s2_preview/agent_rl/s2-preview-thinker_sft_0228b_rl0312rc1_fix_klmismatch/20260331212858/hf-15"
    api_base = "http://10.102.252.171:23333/v1"
    api_key = "sk-admin"
    proxy = None

    async def main():
        model = AsyncAPIClient(
            model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxy),
            sample_params=SampleParameters(
                temperature=0.7,
            ),
            timeout=600,
            max_retry=5,
            sleep_interval=5,
            extra_body=extra_body,
        )
        response = await model.chat(messages, tools=tools)
        print("Response:", response)

    asyncio.run(main())
