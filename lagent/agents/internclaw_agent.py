import asyncio
import base64
import json
import mimetypes
import platform
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from lagent.actions import AsyncActionExecutor, BaseAction, ActionExecutor
from lagent.agents.agent import Agent, AsyncAgent
from lagent.skills.skills import BaseSkillsBackend, SkillsLoader
from lagent.schema import (
    ActionReturn,
    ActionStatusCode,
    ActionValidCode,
    AgentMessage,
    AgentStatusCode,
)
from lagent.utils import create_object


def get_tool_prompt(actions: list, exclude_arguments: list = None) -> str:
    exclude_arguments = exclude_arguments or []

    def _convert_tool_schema(action_description: dict, name_pattern: str = '{}') -> dict:
        action_description = deepcopy(action_description)
        properties = {}
        required = list(action_description.get('required', []))
        for param in action_description['parameters']:
            param = deepcopy(param)
            param_name, param_type = param.pop('name'), param.pop('type')
            if param_name in exclude_arguments:
                if param_name in required:
                    required.remove(param_name)
                continue
            param_type = [t.lower() for t in param_type] if isinstance(param_type, list) else param_type.lower()
            properties[param_name] = {'type': param_type, **param}
        return {
            'type': 'function',
            'function': {
                'name': name_pattern.format(action_description['name']),
                'description': action_description['description'],
                'parameters': {'type': 'object', 'properties': properties, 'required': required},
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

    async def forward(self, *message, **kwargs):
        formatted_messages, tools = self.aggregator.aggregate(
            self.memory, self.name, self.output_format, self.template
        )
        llm_response = await self.llm.chat(formatted_messages, tools=tools, **kwargs)
        message = AgentMessage(
            sender=self.name,
            content=llm_response.get('content') or '',
            tool_calls=llm_response.get('tool_calls') or [],
            reasoning_content=llm_response.get('reasoning_content'),
        )
        return message


class AsyncEnvAgent(AsyncAgent):
    def __init__(self,
                 actions,
                 skills: SkillsLoader=None,
                 long_term_memory=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(actions, ActionExecutor) or hasattr(actions, 'forward'):
            self.actions = actions
        else:
            self.actions = AsyncActionExecutor(actions)
        self.skills = create_object(skills)
        self.long_term_memory = create_object(long_term_memory)

    async def get_env_info(self) -> Dict[str, Any]:
        env_info: Dict[str, Any] = {
            'skills': '',
            'active_skills': '',
            'memory': '',
            'tools': [],
            'runtime': {}
        }

        if self.skills is not None:
            env_info['skills'] = await self.skills.build_skills_summary()
            always_skills = await self.skills.get_always_skills()
            if always_skills:
                env_info['active_skills'] = await self.skills.load_skills_for_context(always_skills)

        if self.long_term_memory is not None:
            env_info['memory'] = await self.long_term_memory.get_info()
        if self.actions:
            env_info['tools'] = get_tool_prompt(list(self.actions.actions.values()))

        env_info['runtime'] = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
        }

        return env_info

    async def forward(self, message, **kwargs):
        if isinstance(message, str):
            return AgentMessage(sender=self.name, content=message, env_info=await self.get_env_info())

        if not message.tool_calls:
            return AgentMessage(
                sender=self.name,
                content=message.content,
                env_info=await self.get_env_info(),
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
            tool_call = deepcopy(tool_call)
            try:
                if tool_call['function']['name'].split('.', 1)[0] not in self.actions:
                    return ActionReturn(valid=ActionValidCode.INVALID, errmsg=f"Tool {tool_call['function']['name']} Not Found")
                if isinstance(tool_call['function']['arguments'], str):
                    tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
            except Exception as e:
                return ActionReturn(valid=ActionValidCode.INVALID, errmsg=str(e))
            tool_response: ActionReturn = (
                await self.actions(
                    AgentMessage(
                        sender='assistant', content=dict(name=tool_call['function']['name'], parameters=tool_call['function']['arguments'])
                    ),
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
        # Pair each ActionReturn with its tool_call_id for proper LLM API formatting
        tool_results = []
        for tc, r in zip(message.tool_calls, responses):
            result_dict = asdict(r)
            result_dict['tool_call_id'] = tc.get('id', '')
            tool_results.append(result_dict)
        return_message = AgentMessage(
            sender=self.name,
            content=tool_results,
            reward=0.0,
            env_info=await self.get_env_info(),
        )

        return return_message


class InternClawAgent(AsyncAgent):
    def __init__(self,
                 policy_agent: Dict,
                 env_agent: Dict,
                 compact_agent: Dict = None,
                 consolidate_agent: Dict = None,
                 max_turn: int = 500,
                 finish_condition: Optional[callable] = lambda m, _: m and not m.tool_calls,
                 **kwargs):
        super().__init__(**kwargs)
        self.policy_agent = create_object(policy_agent)
        self.env_agent = create_object(env_agent)
        self.compact_agent = create_object(compact_agent)
        self.consolidate_agent = create_object(consolidate_agent)
        self.max_turn = max_turn
        self.finish_condition = finish_condition

    async def forward(self, env_message, **kwargs):
        selection_message: AgentMessage = None
        current_turn = 0
        env_message = await self.env_agent(env_message, **kwargs)

        while not (
            self.finish_condition is not None
            and self.finish_condition(selection_message, env_message)
        ) and (self.max_turn is None or current_turn < self.max_turn):
            selection_message = await self.policy_agent(env_message, **kwargs)

            # ── Orchestrator manages memory ──
            await self._maybe_manage_memory(selection_message, env_message)

            env_message = await self.env_agent(selection_message)
            current_turn += 1
        if selection_message is not None:
            return AgentMessage(sender=self.name, content=selection_message.content, finish_reason='stop')
        return AgentMessage(sender=self.name, content="Finished", finish_reason='stop')

    async def _maybe_manage_memory(
        self, policy_message: AgentMessage, env_message: AgentMessage,
    ) -> None:
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

        # Get formatted context from policy's aggregator
        formatted_messages, tools = self.policy_agent.aggregator.aggregate(
            self.policy_agent.memory, self.policy_agent.name,
            self.policy_agent.output_format, self.policy_agent.template,
        )

        from lagent.agents.compact_agent import estimate_token_count
        compact_input = AgentMessage(
            sender=self.name,
            content=formatted_messages,
            extra_info={'context_tokens': estimate_token_count(formatted_messages, tools)},
        )

        if not self.compact_agent.should_compact(compact_input):
            return

        import logging
        _logger = logging.getLogger("lagent.agents.internclaw")

        # 1. Consolidate first (preserve info before compacting)
        if self.consolidate_agent:
            try:
                await self.consolidate_agent(compact_input)
                self.consolidate_agent.reset(recursive=True)
                _logger.info("Consolidation completed")
            except Exception:
                _logger.exception("Consolidation failed, continuing with compact")

        # 2. Compact — inject summary + boundary into env_message
        try:
            summary_msg = await self.compact_agent(compact_input)
            self.compact_agent.reset(recursive=True)
            if summary_msg and summary_msg.content:
                if env_message.env_info is None:
                    env_message.env_info = {}
                env_message.env_info['conversation_summary'] = summary_msg.content
                env_message.env_info['compact_boundary'] = len(
                    self.policy_agent.memory.memory
                )
                _logger.info("Compact summary injected (%d chars)", len(summary_msg.content))
        except Exception:
            _logger.exception("Compact failed")

if __name__ == "__main__":
    import asyncio
    import os
    from pathlib import Path

    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.agents.compact_agent import AsyncCompactAgent
    from lagent.actions.filesystem import ReadFileAction, WriteFileAction, EditFileAction
    from lagent.actions.shell import ShellAction
    from lagent.actions.save_memory import AsyncSaveMemoryAction
    from lagent.memory.openclaw_provider import OpenClawMemoryProvider
    from lagent.hooks.logger import MessageLogger
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters
    from lagent.agents.aggregator.compact_aggregator import CompactAggregator
    # ── Model config ──
    model_name = "Pro/moonshotai/Kimi-K2.5"
    api_base = "http://35.220.164.252:3888/v1"
    api_key = "" 
    proxy = "http://100.100.72.89:8899"
    
    model_name = "/mnt/shared-storage-user/llmit1/user/liujiangning/exp/s2_preview/agent_rl/s2-preview-thinker_sft_0228b_rl0312rc1_fix_klmismatch/20260331212858/hf-15"
    api_base = "http://10.102.252.171:23333/v1"
    api_key = "sk-admin"
    proxy = None
    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxy),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=600,
        max_retry=500,
        sleep_interval=5,
        extra_body=dict(spaces_between_special_tokens=False)
    )

    workspace = Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace")

    CONSOLIDATION_PROMPT = (
        "You are a memory consolidation agent. Review the conversation "
        "and call the save_memory tool to persist important information.\n\n"
        "Extract key facts, decisions, user preferences, and project context. "
        "Merge with existing long-term memory. For history_entry, write a "
        "grep-searchable summary starting with [YYYY-MM-DD HH:MM]."
    )

    async def main():
        # ── 1. Actions ──
        base_actions = [
            ReadFileAction(workspace=str(workspace)),
            WriteFileAction(workspace=str(workspace)),
            EditFileAction(workspace=str(workspace)),
            ShellAction(working_dir=str(workspace)),
        ]

        # ── 2. Memory provider (read) + action (write) ──
        memory_provider = OpenClawMemoryProvider(workspace)
        save_action = AsyncSaveMemoryAction(workspace)

        # ── 3. Hooks ──
        logger_hook = MessageLogger()

        # ── 4. Policy agent ──
        aggregator = InternClawContextBuilder(workspace, tools=None)
        policy = AsyncPolicyAgent(
            llm=model,
            aggregator=aggregator,
            hooks=[logger_hook],
        )

        # ── 5. Env agent ──
        env = AsyncEnvAgent(
            actions=base_actions + [save_action],
            skills=SkillsLoader(workspace),
            long_term_memory=memory_provider,
        )

        # ── 6. Compact agent ──
        compact = AsyncCompactAgent(
            name='compact',
            llm=model,
            max_context_tokens=65536,
            threshold_ratio=0.5,
        )

        # ── 7. Consolidate agent (standard InternClawAgent) ──
        consolidate_policy = AsyncPolicyAgent(
            name='consolidate_policy',
            llm=model,
            template=CONSOLIDATION_PROMPT,
            hooks=[logger_hook],
            aggregator=CompactAggregator()
        )
        consolidate_env = AsyncEnvAgent(
            actions=[AsyncSaveMemoryAction(workspace)],
        )
        consolidate = InternClawAgent(
            policy_agent=consolidate_policy,
            env_agent=consolidate_env,
            max_turn=1,
            finish_condition=None,
        )

        # ── 8. Orchestrator ──
        agent = InternClawAgent(
            policy_agent=policy,
            env_agent=env,
            compact_agent=compact,
            consolidate_agent=consolidate,
        )

        # ── Interactive loop ──
        print("=" * 60)
        print("  InternClaw Agent (with Memory System)")
        print("  Commands:")
        print("    quit/exit  — stop")
        print("    memory     — check MEMORY.md")
        print("    history    — check HISTORY.md")
        print("    compact    — force compact (consolidate + compress)")
        print("    consolidate — force consolidate only")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n[You] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit'):
                print("Bye!")
                break
            if user_input.lower() == 'memory':
                info = await memory_provider.get_info()
                print("\n--- MEMORY.md ---")
                print(info.get('long_term', '(empty)'))
                print("--- end ---")
                continue
            if user_input.lower() == 'history':
                history_file = workspace / "memory" / "HISTORY.md"
                if history_file.exists():
                    print("\n--- HISTORY.md ---")
                    print(history_file.read_text())
                    print("--- end ---")
                else:
                    print("(no history yet)")
                continue
            if user_input.lower() in ('compact', 'consolidate'):
                from lagent.agents.compact_agent import estimate_token_count
                # Get formatted context from policy
                formatted_messages, tools = policy.aggregator.aggregate(
                    policy.memory, policy.name, policy.output_format, policy.template,
                )
                token_count = estimate_token_count(formatted_messages, tools)
                print(f"\n  Session: {len(policy.memory.memory)} messages, ~{token_count} tokens")

                if user_input.lower() in ('compact', 'consolidate'):
                    # Force consolidate
                    print("  Running consolidation...")
                    try:
                        compact_input = AgentMessage(
                            sender='user',
                            content=formatted_messages,
                            extra_info={'context_tokens': token_count},
                        )
                        await consolidate(compact_input)
                        print("  Consolidation done. Check 'memory' and 'history'.")
                    except Exception as e:
                        print(f"  Consolidation failed: {e}")

                if user_input.lower() == 'compact':
                    # Also run compact
                    print("  Running compact...")
                    try:
                        compact_input = AgentMessage(
                            sender='user',
                            content=formatted_messages,
                            extra_info={'context_tokens': token_count},
                        )
                        summary_msg = await compact(compact_input)
                        summary_content = summary_msg.content
                        if isinstance(summary_content, dict):
                            summary_content = summary_content.get('content', '')
                        if summary_content:
                            # Inject into last env message in policy memory
                            for msg in reversed(policy.memory.memory):
                                if msg.env_info is not None:
                                    msg.env_info['conversation_summary'] = summary_content
                                    msg.env_info['compact_boundary'] = len(policy.memory.memory)
                                    break
                            print(f"  Compact done. Summary: {len(summary_content)} chars")
                            print(f"  First 200 chars: {summary_content[:200]}...")
                        else:
                            print("  Compact returned empty summary.")
                    except Exception as e:
                        print(f"  Compact failed: {e}")
                continue

            response = await agent(user_input)
            print(f"\n[Agent] {response.content}")

    asyncio.run(main())