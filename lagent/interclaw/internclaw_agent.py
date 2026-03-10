import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from lagent.agents.agent import Agent, AsyncAgent
from lagent.actions import BaseAction, ActionExecutor, AsyncActionExecutor
from lagent.utils import create_object
from lagent.schema import AgentMessage
import asyncio


def get_tool_prompt(actions: list, exclude_arguments: list = None) -> str:
    from copy import deepcopy
    exclude_arguments = exclude_arguments or ['session_id']

    def _convert_tool_schema(action_description: dict, name_pattern: str = '{}') -> dict:
        properties = {}
        for param in action_description['parameters']:
            param = deepcopy(param)
            param_name, param_type = param.pop('name'), param.pop('type')
            if param_name in exclude_arguments:
                continue
            param_type = [t.lower() for t in param_type] if isinstance(param_type, list) else param_type.lower()
            properties[param_name] = {'type': param_type, **param}
        return {
            'type': 'function',
            'function': {
                'name': name_pattern.format(action_description['name']),
                'description': action_description['description'],
                'parameters': {'type': 'object', 'properties': properties, 'required': action_description['required']},
            },
        }

    tools = []
    for action in actions if isinstance(actions, list) else [actions]:
        action = create_object(action)
        action_desc = action.description
        if action.is_toolkit:
            for api in action_desc['api_list']:
                tools.append(_convert_tool_schema(api, f"{action.name}.{{}}"))
        else:
            tools.append(_convert_tool_schema(action_desc))
    return tools

class AsyncPolicyAgent(AsyncAgent):
        
    async def forward(self, *message, session_id=0, **kwargs):
        formatted_messages, tools = self.aggregator.aggregate(
            self.memory.get(session_id), self.name, self.output_format, self.template
        )
        llm_response = await self.llm.chat(formatted_messages, tools=tools, **kwargs)
        message = AgentMessage(sender=self.name, content=llm_response['content'], tool_calls=llm_response.get('tool_calls', []), reasoning_content=llm_response.get('reasoning_content'))
        return message


class AsyncEnvAgent(AsyncAgent):
    def __init__(self,
                 actions,
                 **kwargs):
        super().__init__(**kwargs)
        self.actions = AsyncActionExecutor(actions)
        self.stateful_tools = set()
        
    async def forward(self, message, session_id=0, **kwargs):
        from lagent.schema import (
            ActionReturn,
            ActionStatusCode,
            ActionValidCode,
            AgentStatusCode,
        )
        from copy import deepcopy
        from dataclasses import asdict
        import json
        from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(2),
            retry=retry_if_result(
                lambda r: r.valid == ActionValidCode.OPEN
                and r.state not in [ActionStatusCode.SUCCESS, ActionStatusCode.ARGS_ERROR]
            ),
            retry_error_callback=lambda retry_state: retry_state.outcome.result(),
        )
        async def _inner_func(tool_call):
            try:
                if tool_call['function']['name'].split('.', 1)[0] not in self.actions:
                    return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f'Tool {tool_call['function']["name"]} Not Found')
                if isinstance(tool_call['function']['arguments'], str):
                    tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
                if tool_call['function']['name'] in self.stateful_tools:
                    tool_call = deepcopy(tool_call)
                    tool_call['function']['arguments']['session_id'] = session_id
            except Exception as e:
                return ActionReturn(valid=ActionValidCode.INVALID, errmsg=str(e))
            tool_response: ActionReturn = (
                await self.actions(
                    AgentMessage(
                        sender='assistant', content=dict(name=tool_call['function']['name'], parameters=tool_call['function']['arguments'])
                    ),
                    session_id=session_id,
                )
            ).content
            return tool_response

        tasks = [_inner_func(tool_call) for tool_call in message.tool_calls]
        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses):
            if resp.valid != ActionValidCode.OPEN:
                return AgentMessage(
                    sender=self.name,
                    content=f'Tool Call Error: {resp.errmsg} in tool call '
                    f'{json.dumps(message.tool_calls[i], ensure_ascii=False)}',
                )
            if resp.state != ActionStatusCode.SUCCESS:
                return AgentMessage(
                    sender=self.name,
                    content=f'Tool Call Error: {resp.errmsg} in tool call '
                    f'{json.dumps(message.tool_calls[i], ensure_ascii=False)}',
                    reward=-1 if resp.state == ActionStatusCode.ARGS_ERROR else 0,
                )
        return_message = AgentMessage(
            sender=self.name, content=[asdict(r) for r in responses], reward=0.0
        )
        return return_message


class InternClawAgent(AsyncAgent):
    def __init__(self,
                 policy_agent: Dict,
                 env_agent: Dict,
                 max_turn: int = 5,
                 finish_condition: Optional[callable] = lambda m: m.tool_calls is None or len(m.tool_calls) == 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.policy_agent = create_object(policy_agent)
        self.env_agent = create_object(env_agent)
        self.max_turn = max_turn
        self.finish_condition = finish_condition
    
    async def forward(self, message, session_id=0, **kwargs):
        
        for _ in range(self.max_turn):
            message = await self.policy_agent(message, session_id=session_id, **kwargs)
            if self.finish_condition and self.finish_condition(message):
                return message
            message = await self.env_agent(message, session_id=session_id, **kwargs)
        return message

if __name__ == "__main__":
    # Example usage
    from lagent.interclaw.context import ContextBuilder
    from lagent.interclaw.model import AsyncAPIClient, ModelConfig, SampleParameters
    from lagent.actions.filesystem import ReadFileAction, WriteFileAction, EditFileAction
    from lagent.actions.shell import ShellAction
    model_name = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1_mini_official/interns1_1_mini_sft_based_cpt_bs512_epoch1_maxlr3e-5_minlr1e-6_max16k-hf/20260207101512/hf-4374"
    # model_name = "gpt-4o-2024-08-06"
    api_base = "http://10.102.218.26:23333/v1/"
    # api_base = f"http://35.220.164.252:3888/v1beta/models/{model_name}:generateContent"
    api_key = "sk-blAvnaExZFrQfHVuyIF5VEB3I0GrQ7FNhdAobU3pKpfLvxLb"
    extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    proxies = dict(
        # http='http://100.103.22.82:8888',
        # https='http://100.103.22.82:8888',
    )

    model = AsyncAPIClient(
            model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key),
            sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
            timeout=600,
            max_retry=5,
            sleep_interval=5,
            extra_body=extra_body,
        )
    workspace = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace"
    actions = [ReadFileAction(workspace=workspace), WriteFileAction(workspace=workspace), EditFileAction(workspace=workspace), ShellAction(working_dir=workspace)]
    aggregator = ContextBuilder(Path(workspace), tools=get_tool_prompt(actions))
    policy = AsyncPolicyAgent(
                llm=model,
                aggregator=aggregator)
    env = AsyncEnvAgent(actions=actions)
    agent = InternClawAgent(policy_agent=policy, env_agent=env)
    # Quick test
    async def test():
        response = await agent("帮我看一下当前目录有没有 README.md 文件，如果有的话，读一下里面的内容, 如果没有的话，帮我新建一个，写入‘牛逼’", session_id=0)
        print(response)
    import asyncio
    asyncio.run(test())