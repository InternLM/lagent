import asyncio
import json
import logging
import platform
from copy import deepcopy
from dataclasses import asdict
from functools import reduce
from operator import add
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from lagent.schema import ActionReturn, ActionStatusCode, ActionValidCode, AgentMessage, AgentStatusCode
from lagent.skills.skills import SkillsLoader
from lagent.utils import create_object, load_class_from_string, truncate_text
from .agent import AsyncAgent

logger = logging.getLogger("lagent.agents.fc_agent")


class FunctionCallAgent(AsyncAgent):
    def __init__(
        self,
        policy_agent: Union[Dict, AsyncAgent],
        env_agent: Union[Dict, AsyncAgent],
        compact_agent: Optional[Dict] = None,
        consolidate_agent: Optional[Dict] = None,
        finish_condition: callable = lambda m, _: m and not m.tool_calls,
        max_turn: Optional[int] = None,
        initialize_input: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.policy_agent = create_object(policy_agent)
        self.env_agent = create_object(env_agent)
        self.compact_agent = create_object(compact_agent)
        self.consolidate_agent = create_object(consolidate_agent)
        if isinstance(finish_condition, str):
            finish_condition = load_class_from_string(finish_condition)
        self.finish_condition = finish_condition
        self.max_turn = max_turn
        self.initialize_input = initialize_input

    async def forward(self, env_message: AgentMessage, **kwargs):
        policy_message: AgentMessage = None
        current_turn = 0
        if self.initialize_input:
            env_message = await self.env_agent(env_message, **kwargs)

        while (self.finish_condition is None or not self.finish_condition(policy_message, env_message)) and (
            self.max_turn is None or current_turn < self.max_turn
        ):
            policy_message = await self.policy_agent(env_message, **kwargs)
            if policy_message.stream_state == AgentStatusCode.SERVER_ERR:
                raise ValueError("Rollout response error: state is neither completed nor aborted!")
            if policy_message.stream_state == AgentStatusCode.SESSION_OUT_OF_LIMIT:
                for _ in range(2):  # remove the last two messages
                    self.policy_agent.memory.delete(-1)
                return AgentMessage(
                    sender=self.name,
                    content=policy_message.content,
                    finish_reason=policy_message.finish_reason,
                )
            if policy_message.finish_reason == 'abort':
                return AgentMessage(sender=self.name, content=policy_message.content, finish_reason='abort')

            # Orchestrator manages memory
            await self._maybe_manage_memory(policy_message, env_message)

            env_message = await self.env_agent(policy_message)
            current_turn += 1
        if policy_message is not None:
            return AgentMessage(sender=self.name, content=policy_message.content, finish_reason='stop')
        return AgentMessage(sender=self.name, content="Finished", finish_reason='stop')

    async def _maybe_manage_memory(self, policy_message: AgentMessage, env_message: AgentMessage) -> None:
        """Orchestrate compact and consolidate.

        Orchestrator calls policy's aggregator to get formatted_messages,
        checks should_compact, and if triggered:
          1. Runs consolidate_agent (optional)
          2. Runs compact_agent to produce summary
          3. Injects summary + boundary into env_message
        ContextBuilder reads these on the next turn.
        """
        if not self.compact_agent:
            return

        from lagent.agents.compact_agent import estimate_token_count

        state = self.get_messages()
        formatted_messages, tools = state['policy_agent.messages'], state['policy_agent.tools']
        compact_input = AgentMessage(
            sender=self.name,
            content=formatted_messages,
            extra_info={'context_tokens': estimate_token_count(formatted_messages, tools)},
        )
        if not (hasattr(self.compact_agent, 'should_compact') and self.compact_agent.should_compact(compact_input)):
            return

        # 1. Consolidate first (preserve info before compacting)
        if self.consolidate_agent:
            try:
                await self.consolidate_agent(compact_input)
                self.consolidate_agent.reset(recursive=True)
                logger.info("Consolidation completed")
            except Exception:
                logger.exception("Consolidation failed, continuing with compact")
        # 2. Compact — inject summary + boundary into env_message
        try:
            summary_msg = await self.compact_agent(compact_input)
            self.compact_agent.reset(recursive=True)
            if summary_msg and summary_msg.content:
                if env_message.env_info is None:
                    env_message.env_info = {}
                env_message.env_info['conversation_summary'] = summary_msg.content
                env_message.env_info['compact_boundary'] = len(self.policy_agent.memory.memory)
                logger.info("Compact summary injected (%d chars)", len(summary_msg.content))
        except Exception:
            logger.exception("Compact failed")


class MemoryProvider(Protocol):
    async def get_info(self) -> dict:
        """Return long-term memory info for EnvAgent's env_info. The content and format are flexible, but should be concise."""
        ...


class EnvAgent(AsyncAgent):
    def __init__(
        self,
        actions,
        skills: Optional[SkillsLoader] = None,
        long_term_memory: Optional[MemoryProvider] = None,
        max_tool_response_length: int = None,
        tool_response_truncate_side: Optional[Literal['left', 'right', 'middle']] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.actions: dict = {}
        for action in actions:
            action = create_object(action)
            for tool in action.to_openai_format_tools():
                self.actions[tool['function']['name']] = action
        self.skills = create_object(skills)
        self.long_term_memory = create_object(long_term_memory)
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

    async def get_env_info(self) -> Dict[str, Any]:
        env_info = {'skills': '', 'active_skills': '', 'memory': '', 'tools': [], 'runtime': {}}
        if self.skills is not None:
            env_info['skills'] = await self.skills.build_skills_summary()
            always_skills = await self.skills.get_always_skills()
            if always_skills:
                env_info['active_skills'] = await self.skills.load_skills_for_context(always_skills)
        if self.long_term_memory is not None:
            env_info['memory'] = await self.long_term_memory.get_info()
        if self.actions:
            env_info['tools'] = reduce(add, [action.to_openai_format_tools() for action in self.actions.values()])
        for name in ['system', 'machine', 'python_version']:
            env_info['runtime'][name] = getattr(platform, name)()
        return env_info

    async def forward(self, message: AgentMessage, **kwargs):
        if not message.tool_calls:
            return AgentMessage(sender=self.name, content=message.content, env_info=await self.get_env_info())

        tool_responses = await asyncio.gather(
            *[self._retry_mechanism(self.execute_tool)(tool_call) for tool_call in message.tool_calls]
        )
        content = []
        for tool_call_id, tool_response in zip(
            message.tool_calls_ids or [tc.get('id') for tc in message.tool_calls], tool_responses
        ):
            tool_response.tool_call_id = tool_call_id
            content.append(asdict(tool_response))
        return AgentMessage(sender=self.name, content=content, env_info=await self.get_env_info())

    async def execute_tool(self, tool_call: dict) -> ActionReturn:
        tool_call = deepcopy(tool_call)
        try:
            if 'function' in tool_call:
                tool_call = tool_call['function']
            if tool_call['name'].split('.', 1)[0] not in self.actions:
                return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f'Tool {tool_call["name"]} Not Found')
            if isinstance(tool_call['arguments'], str):
                tool_call['arguments'] = json.loads(tool_call['arguments'])
        except Exception as e:
            return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f'Invalid tool call format: {str(e)}')
        action = self.actions[tool_call['name']]
        tool_response: ActionReturn = await action(
            tool_call['arguments'], tool_call['name'].rsplit('.', 1)[-1] if action.is_toolkit else 'run'
        )
        if tool_response.max_tool_response_length is None:
            tool_response.max_tool_response_length = self.max_tool_response_length
        if tool_response.tool_response_truncate_side is None:
            tool_response.tool_response_truncate_side = self.tool_response_truncate_side
        return tool_response
