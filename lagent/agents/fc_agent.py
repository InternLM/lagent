import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Literal, Optional, Union

from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from lagent.actions import AsyncActionExecutor
from lagent.hooks import Hook
from lagent.schema import ActionReturn, ActionStatusCode, ActionValidCode, AgentMessage, AgentStatusCode
from lagent.utils import create_object, truncate_text
from .agent import AsyncAgent

DEFAULT_TOOL_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def get_tool_prompt(actions: list, exclude_arguments: list = None, template: str = DEFAULT_TOOL_TEMPLATE) -> str:
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
    return template.format(tools='\n'.join([json.dumps(tool, ensure_ascii=False) for tool in tools]))


class FunctionCallAgent(AsyncAgent):
    def __init__(
        self,
        select_agent: Union[Dict, AsyncAgent],
        env_agent: Union[Dict, AsyncAgent],
        finish_condition: callable = lambda x, _: x and not x.tool_calls,
        max_turn: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.select_agent = create_object(select_agent)
        self.env_agent = create_object(env_agent)
        self.finish_condition = finish_condition
        self.max_turn = max_turn

    async def forward(self, env_message: AgentMessage, session_id: str | int, **kwargs):
        selection_message: AgentMessage = None
        current_turn = 0
        while (self.finish_condition is None or not self.finish_condition(selection_message, env_message)) and (
            self.max_turn is None or current_turn < self.max_turn
        ):
            selection_message = await self.select_agent(env_message, session_id=session_id, **kwargs)
            if selection_message.stream_state == AgentStatusCode.SERVER_ERR:
                raise ValueError("Rollout response error: state is neither completed nor aborted!")
            if selection_message.stream_state == AgentStatusCode.SESSION_OUT_OF_LIMIT:
                for _ in range(2):  # remove the last two messages
                    self.select_agent.memory.get(session_id).delete(-1)
                return AgentMessage(
                    sender=self.name,
                    content='Exceeded context length limit',
                    finish_reason=selection_message.finish_reason,
                )
            if selection_message.finish_reason == 'abort':
                return AgentMessage(sender=self.name, content='Aborted request', finish_reason='abort')
            env_message = await self.env_agent(selection_message, session_id=session_id)
            current_turn += 1
        return AgentMessage(sender=self.name, content="Finished", finish_reason='stop')


class EnvAgent(AsyncAgent):
    def __init__(
        self,
        actions: list,
        stateful_tools: List[str] = None,
        max_tool_response_length: int = None,
        tool_response_truncate_side: Literal['left', 'right', 'middle'] = 'middle',
        action_hooks: List[Union[dict, Hook]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.actions = AsyncActionExecutor(actions, hooks=action_hooks)
        self.stateful_tools = stateful_tools or []
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncate_side = tool_response_truncate_side
        self._retry_mechanism = retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(2),
            retry=retry_if_result(
                lambda r: r.valid == ActionValidCode.OPEN
                and r.state not in [ActionStatusCode.SUCCESS, ActionStatusCode.ARGS_ERROR]
            ),
            retry_error_callback=lambda retry_state: retry_state.outcome.result(),
        )

    async def forward(self, selection_message: AgentMessage, session_id: str | int, **kwargs):
        if not selection_message.tool_calls:
            return AgentMessage(sender=self.name, content='No tool call')

        tool_responses = await asyncio.gather(
            *[
                self._retry_mechanism(self.execute_tool)(tool_call, session_id)
                for tool_call in selection_message.tool_calls
            ]
        )
        for tool_call_id, tool_response in zip(selection_message.tool_calls_ids, tool_responses):
            tool_response.tool_call_id = tool_call_id
            res = tool_response.format_result()
            if self.max_tool_response_length is not None and len(res) > self.max_tool_response_length:
                res = truncate_text(res, max_num=self.max_tool_response_length, side=self.tool_response_truncate_side)
                tool_response.result = [{'type': 'text', 'content': res}]
        return AgentMessage(sender=self.name, content=[asdict(resp) for resp in tool_responses])

    async def execute_tool(self, tool_call: dict, session_id: str | int) -> ActionReturn:
        try:
            if tool_call['name'].split('.', 1)[0] not in self.actions:
                return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f'Tool {tool_call["name"]} Not Found')
            if isinstance(tool_call['arguments'], str):
                tool_call['arguments'] = json.loads(tool_call['arguments'])
            if tool_call['name'] in self.stateful_tools:
                tool_call = deepcopy(tool_call)
                tool_call['arguments']['session_id'] = session_id
        except Exception as e:
            return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f'Invalid tool call format: {str(e)}')
        tool_response: ActionReturn = (
            await self.actions(
                AgentMessage(
                    sender='assistant', content=dict(name=tool_call['name'], parameters=tool_call['arguments'])
                ),
                session_id=session_id,
            )
        ).content
        return tool_response
