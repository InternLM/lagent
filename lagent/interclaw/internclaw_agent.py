import asyncio
import base64
import json
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from lagent.actions import AsyncActionExecutor, BaseAction, ActionExecutor
from lagent.agents.agent import Agent, AsyncAgent
from lagent.interclaw.memory import BaseMemoryBackend, MemoryStore
from lagent.interclaw.skills import BaseSkillsBackend, SkillsLoader
from lagent.schema import AgentMessage, AgentStatusCode
from lagent.utils import create_object


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
        
    async def forward(self, *message, tools=None, session_id=0, **kwargs):
        formatted_messages, tools = self.aggregator.aggregate(
            self.memory.get(session_id), self.name, self.output_format, self.template
        )
        llm_response = await self.llm.chat(formatted_messages, tools=tools, **kwargs)
        message = AgentMessage(sender=self.name, content=llm_response['content'], tool_calls=llm_response.get('tool_calls', []), reasoning_content=llm_response.get('reasoning_content'))
        return message


class AsyncEnvAgent(AsyncAgent):
    def __init__(self,
                 actions,
                 skills: SkillsLoader=None,
                 memory_store: MemoryStore=None,
                 stateful_tools = [],
                 **kwargs):
        super().__init__(**kwargs)
        self.actions = AsyncActionExecutor(actions)
        self.stateful_tools = set(stateful_tools)
        self.skills = skills
        self.memory_store = memory_store

    async def get_env_info(self, session_id=0) -> Dict[str, Any]:
        env_info: Dict[str, Any] = {
            'session_id': session_id,
            'skills': '',
            'active_skills': '',
            'memory': '',
            'tools': []
        }

        if self.skills is not None:
            env_info['skills'] = await self.skills.build_skills_summary()
            always_skills = await self.skills.get_always_skills()
            if always_skills:
                env_info['active_skills'] = await self.skills.load_skills_for_context(always_skills)

        if self.memory_store is not None:
            long_term = await self.memory_store.read_long_term()
            env_info['memory'] = {
                'available': True,
                'long_term': long_term,
            }
        if self.actions:
            env_info['tools'] = get_tool_prompt(list(self.actions.actions.values()))

        return env_info
        
    async def forward(self, message, session_id=0, **kwargs):
        from lagent.schema import (
            ActionReturn,
            ActionStatusCode,
            ActionValidCode,
            AgentStatusCode,
        )
        from copy import deepcopy
        from dataclasses import asdict
        from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

        if isinstance(message, str):
            return AgentMessage(sender=self.name, content=message, env_info=await self.get_env_info(session_id=session_id))

        if not message.tool_calls:
            return AgentMessage(
                sender=self.name,
                content=message.content,
                env_info=await self.get_env_info(session_id=session_id),
                tool_calls=message.tool_calls,
            )

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
                    return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f"Tool {tool_call['function']['name']} Not Found")
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
            sender=self.name,
            content=[asdict(r) for r in responses],
            reward=0.0,
            env_info=await self.get_env_info(session_id=session_id),
        )
        return return_message


class InternClawAgent(AsyncAgent):
    def __init__(self,
                 policy_agent: Dict,
                 env_agent: Dict,
                 max_turn: int = 50,
                 finish_condition: Optional[callable] = lambda m, _: m and not m.tool_calls,
                 **kwargs):
        super().__init__(**kwargs)
        self.policy_agent = create_object(policy_agent)
        self.env_agent = create_object(env_agent)
        self.max_turn = max_turn
        self.finish_condition = finish_condition
    
    async def forward(self, env_message, session_id=0, **kwargs):
        selection_message: AgentMessage = None
        current_turn = 0
        env_message = await self.env_agent(env_message, session_id=session_id, **kwargs)
        
        while (
            self.finish_condition is None
            or not self.finish_condition(selection_message, env_message)
            and (self.max_turn is None or current_turn < self.max_turn)
        ):
            selection_message = await self.policy_agent(env_message, session_id=session_id, **kwargs)
            env_message = await self.env_agent(selection_message, session_id=session_id)
            current_turn += 1
        return AgentMessage(role="env", content="Finished", finish_reason='stop')

if __name__ == "__main__":
    # Example usage
    from lagent.interclaw.context import ContextBuilder
    from lagent.interclaw.memory import StatefulActionMemoryBackend
    from lagent.interclaw.model import AsyncAPIClient, ModelConfig, SampleParameters
    from lagent.interclaw.skills import StatefulActionSkillsBackend
    from lagent.actions.filesystem import ReadFileAction, WriteFileAction, EditFileAction
    from lagent.actions.mcp_client import AsyncMCPClientStatefulAction
    from lagent.actions.shell import ShellAction
    from lagent.hooks.logger import MessageLogger
    import os
    # model_name = "gemini-3.1-pro-preview-thinking"
    # api_base = "http://35.220.164.252:3888/v1"
    # api_key = ""
    # proxy = "http://100.100.72.89:8899"
    # extra_body = {}
    model_name = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1_mini_official/interns1_1_mini_sft_based_cpt_bs512_epoch1_maxlr3e-5_minlr1e-6_max16k-hf/20260207101512/hf-4374"
    api_base = "http://10.102.218.28:23333/v1"
    
    api_key='YOUR KEY'
    extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    proxy = None
    
    model = AsyncAPIClient(
            model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxy),
            sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
            timeout=600,
            max_retry=5,
            sleep_interval=5,
            extra_body=extra_body,
        )
    init_dir = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace"
    # actions = [ReadFileAction(workspace=workspace), WriteFileAction(workspace=workspace), EditFileAction(workspace=workspace), ShellAction(working_dir=workspace)]
    shell_action = AsyncMCPClientStatefulAction('http',url='http://simple-shell.ailab.ailab.ai/mcp', init_dir=init_dir)
    # print(asyncio.run(shell_action(dict(command='ls la', session_id="demo"))))
    async def test():
        home_path = await shell_action.run(command='ls -la', session_id='demo')
        workspace = os.path.join(json.loads(home_path.result[0]['content'])['cwd'], 'workspace')
        actions = [shell_action]
        aggregator = ContextBuilder(Path(workspace), tools=None)
        skills = SkillsLoader(Path(workspace))
        memory = MemoryStore(Path(workspace), use_default_backend=False)
        skills_backend = StatefulActionSkillsBackend(shell_action, workspace_root=workspace, session_id='demo')
        memory_backend = StatefulActionMemoryBackend(shell_action, workspace_root=workspace, session_id='demo')
        skills.bind_backend(skills_backend)
        memory.bind_backend(memory_backend)
        policy = AsyncPolicyAgent(
                    llm=model,
                    aggregator=aggregator,
                    hooks=[MessageLogger()])
        env = AsyncEnvAgent(
            actions=actions,
            skills=skills,
            memory_store=memory,
            # skills_backend=skills_backend,
            # memory_backend=memory_backend,
            stateful_tools=[shell_action.name])
        agent = InternClawAgent(policy_agent=policy, env_agent=env)
        # Quick test
        session_id="demo"
        env_state = await env.get_env_info(session_id=session_id)
        print('env_state:', env_state)
        response = await agent("你能帮我分析一下lagent的源码设计么", session_id=session_id)
        # print(response)
        await shell_action.close_session(session_id=session_id)
    import asyncio
    asyncio.run(test())