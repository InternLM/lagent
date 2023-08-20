from typing import Union

from lagent.actions import ActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel


class BaseAgent:

    def __init__(self, llm: Union[BaseModel, BaseAPIModel],
                 action_executor: ActionExecutor, prompter) -> None:

        self._session_history = []
        self._llm = llm
        self._action_executor = action_executor
        self._prompter = prompter

    def add_action(self, tools: BaseAction) -> None:
        self._action_executor.add_action(tools)

    def del_action(self, name: str) -> None:
        self._action_executor.del_action(name)

    def chat(self, message):
        raise NotImplementedError

    def save_session(self):
        raise NotImplementedError

    def load_session(self):
        raise NotImplementedError

    @property
    def session_history(self):
        return self._session_history
