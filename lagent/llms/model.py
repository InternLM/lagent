import asyncio
import json
import random
import traceback
from logging import getLogger
from typing import List, Union, Optional, Dict, TypedDict

import aiohttp
from lagent.llms.openai import AsyncGPTAPI

logger = getLogger(__name__)

import httpx
from openai import APITimeoutError, AsyncOpenAI, NOT_GIVEN
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
        http_client = httpx.AsyncClient(proxy=model.get('proxy'), timeout=timeout, trust_env=False) if model.get('proxy') else httpx.AsyncClient(timeout=timeout)
        self.clients = [
            AsyncOpenAI(api_key=model["api_key"], base_url=url, http_client=http_client)
            for url in (model['base_url'] if isinstance(model['base_url'], list) else [model['base_url']])
        ]
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
        for attempt in range(self.max_retry):
            try:
                client = random.choice(self.clients)
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    temperature=self.sample_params.get("temperature", 0.7),
                    top_p=self.sample_params.get("top_p", 1.0),
                    timeout=self.timeout,
                    extra_body=self.extra_body,
                    max_tokens= self.sample_params.get("max_tokens", 64 * 1024),
                    reasoning_effort=self.sample_params.get("reasoning_effort", NOT_GIVEN)
                )
                break
            except (APITimeoutError, TimeoutError) as e:
                logger.error(f"LLM Call Timeout: {e}")
                if attempt == self.max_retry - 1:
                    assistant_msg_dict = {"role": "assistant", "content": f"LLM Call Timeout: {e}"}
                    return assistant_msg_dict
                await asyncio.sleep(self.sleep_interval)
            except Exception as e:
                for val in [
                    "用户额度不足",
                    "剩余额度",
                    "TimeoutError",
                    "litellm.BadRequestError",
                    "litellm.APIError: APIError",
                    "Failed to parse fc related info to json format!",
                    "Error code"
                ]:
                    if val in str(e):
                        import traceback
                        traceback.print_exc()
                        logger.error(f"[Retry] {attempt} LLM Call Error: {e}")
                        if attempt == self.max_retry - 1:
                            assistant_msg_dict = {"role": "assistant", "content": f"LLM Call Error: {e}"}
                            return assistant_msg_dict
                        await asyncio.sleep(self.sleep_interval)
                        break
                else:
                    import traceback
                    traceback.print_exc()
                    assistant_msg_dict = {"role": "assistant", "content": f"LLM Call Error: {e}"}
                    return assistant_msg_dict

        choice = response.choices[0]
        message_data = choice.message
        return message_data.model_dump()

if __name__ == '__main__':
    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_current_temperature',
            'description': 'Get current temperature at a location.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': [
                            'celsius',
                            'fahrenheit'
                        ],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': [
                    'location'
                ]
            }
        }
    }, {
        'type': 'function',
        'function': {
            'name': 'get_temperature_date',
            'description': 'Get temperature at a location and date.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'date': {
                        'type': 'string',
                        'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': [
                            'celsius',
                            'fahrenheit'
                        ],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': [
                    'location',
                    'date'
                ]
            }
        }
    }]



    messages = [
        {'role': 'user', 'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?'}
    ]
    messages = [
        {'role': 'user', 'content': '上海温度'}
    ]
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
            sample_params=SampleParameters(temperature=0.7, ),
            timeout=600,
            max_retry=5,
            sleep_interval=5,
            extra_body=extra_body,
        )
        response = await model.chat(messages, tools=tools)
        print("Response:", response)
    
    asyncio.run(main())