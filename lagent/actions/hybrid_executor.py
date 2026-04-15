"""HybridActionExecutor — routes actions to local or sandbox execution.

Some actions (subagent, save_memory) must run locally because they need
access to the LLM, parent agent memory, or external services. Environment
actions (shell, ipython, file ops) run in the sandbox.

Usage::

    sandbox = SandboxActionExecutor(
        sandbox_client=my_client,
        actions_config=[{"type": "..ShellAction"}, {"type": "..IPythonInterpreter"}],
    )
    await sandbox.connect()

    executor = HybridActionExecutor(
        local_actions=[AsyncAgentAction(), AsyncSaveMemoryAction()],
        sandbox_executor=sandbox,
    )

    # forward() automatically routes by name
    await executor.forward("shell", {"command": "ls"})        # → sandbox
    await executor.forward("AgentAction.spawn", {"task": "..."})  # → local
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Union

from lagent.actions.action_executor import AsyncActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.actions.sandbox_executor import SandboxActionExecutor
from lagent.schema import ActionReturn, ActionValidCode, AgentMessage


class HybridActionExecutor(AsyncActionExecutor):
    """Routes action calls to either local execution or a remote sandbox.

    Inherits from ``AsyncActionExecutor`` for local actions and delegates
    sandbox actions to a ``SandboxActionExecutor``.

    Parameters
    ----------
    local_actions : list
        Actions to run locally (subagent, save_memory, etc.).
    sandbox_executor : SandboxActionExecutor
        Executor that routes to the sandbox daemon.
    **kwargs
        Passed to ``AsyncActionExecutor.__init__`` (hooks, finish_action, etc.).
    """

    def __init__(
        self,
        local_actions: Union[BaseAction, List[BaseAction], Dict, List[Dict]] = None,
        sandbox_executor: SandboxActionExecutor = None,
        **kwargs,
    ):
        super().__init__(actions=local_actions or [], **kwargs)
        self.sandbox_executor = sandbox_executor

    @property
    def _sandbox_actions(self) -> Dict[str, object]:
        if self.sandbox_executor is None:
            return {}
        return self.sandbox_executor.actions

    def description(self) -> List[Dict]:
        descs = super().description()
        if self.sandbox_executor is not None:
            descs = descs + self.sandbox_executor.description()
        return descs

    def __contains__(self, name: str) -> bool:
        return super().__contains__(name) or (
            self.sandbox_executor is not None and name in self.sandbox_executor
        )

    def keys(self) -> List[str]:
        k = super().keys()
        if self.sandbox_executor is not None:
            k = k + self.sandbox_executor.keys()
        return k

    async def forward(self, name: str, parameters: dict, **kwargs) -> ActionReturn:
        action_name = name.split(".")[0] if "." in name else name

        # 1. Local actions take priority
        if action_name in self.actions:
            return await super().forward(name, parameters, **kwargs)

        # 2. Sandbox actions
        if self.sandbox_executor is not None and action_name in self.sandbox_executor:
            return await self.sandbox_executor.forward(name, parameters, **kwargs)

        # 3. Built-in fallbacks (NoAction, FinishAction, InvalidAction)
        if name == self.no_action.name:
            return self.no_action(parameters)
        elif name == self.finish_action.name:
            return self.finish_action(parameters)
        else:
            return self.invalid_action(parameters)

    async def connect(self) -> None:
        """Connect the sandbox executor (if present)."""
        if self.sandbox_executor is not None:
            await self.sandbox_executor.connect()

    async def close(self) -> None:
        """Close the sandbox executor (if present)."""
        if self.sandbox_executor is not None:
            await self.sandbox_executor.close()
