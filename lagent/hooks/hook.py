from itertools import count
from typing import Tuple

from lagent.schema import AgentMessage


class Hook:

    def before_agent(
        self,
        agent,
        message: Tuple[AgentMessage],
        session_id: int,
    ):
        pass

    def after_agent(
        self,
        agent,
        message: AgentMessage,
        session_id: int,
    ):
        pass

    def before_action(
        self,
        executor,
        message: AgentMessage,
        session_id: int,
    ):
        pass

    def after_action(
        self,
        executor,
        message: AgentMessage,
        session_id: int,
    ):
        pass


class RemovableHandle:
    _id_iter = count(0)

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = next(self._id_iter)

    def remove(self):
        del self.hooks_dict[self.id]
