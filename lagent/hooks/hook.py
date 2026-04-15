from itertools import count
from typing import Tuple

from lagent.schema import AgentMessage


class Hook:

    def before_agent(
        self,
        agent,
        message: Tuple[AgentMessage],
    ):
        pass

    def after_agent(
        self,
        agent,
        message: AgentMessage,
    ):
        pass

    def before_action(
        self,
        executor,
        message: AgentMessage,
    ):
        pass

    def after_action(
        self,
        executor,
        message: AgentMessage,
    ):
        pass


class RemovableHandle:
    _id_iter = count(0)

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = next(self._id_iter)

    def remove(self):
        del self.hooks_dict[self.id]
