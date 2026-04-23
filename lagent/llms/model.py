import asyncio
import json
import random
import traceback
from logging import getLogger
from typing import Dict, List, Optional, TypedDict, Union

import aiohttp

from lagent.llms.openai import AsyncGPTAPI

logger = getLogger(__name__)


# from pdp_ext.fc_inferencer import ModelConfig, SampleParameters
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
        max_retry: int = 50,
        sleep_interval: int = 5,
        extra_body: Optional[dict] = None,
        max_tool_response_length: Optional[int] = 4096,
        max_tool_calls_per_turn: int = 5,
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
        self.max_tool_response_length = max_tool_response_length
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

    async def chat(self, messages: List[dict], tools=None, **gen_params) -> str:
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
            stream=False,
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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        def _error_completion(content: str) -> dict:
            return {
                "id": "error",
                "object": "chat.completion",
                "created": 0,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for attempt in range(self.max_retry):
                url = random.choice(self.base_urls)
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        proxy=self.proxy,
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}: {text}")
                        data = await resp.json()

                    return data

                except asyncio.TimeoutError as e:
                    logger.error(f"LLM Call Timeout: {e}")
                    if attempt == self.max_retry - 1:
                        return _error_completion(f"LLM Call Timeout: {e}")
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
                                return _error_completion(f"LLM Call Error: {e}")
                            await asyncio.sleep(self.sleep_interval)
                            break
                    else:
                        traceback.print_exc()
                        return _error_completion(f"LLM Call Error: {e}")


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
