from itertools import count

from lagent.registry import HOOK_REGISTRY, AutoRegister


class Hook(metaclass=AutoRegister(HOOK_REGISTRY)):

    def before_agent(
        self,
        agent,
        message,
    ):
        pass

    def after_agent(
        self,
        agent,
        message,
    ):
        pass

    def before_action(
        self,
        agent,
        message,
    ):
        pass

    def after_action(
        self,
        agent,
        message,
    ):
        pass


class RemovableHandle:
    _id_iter = count(0)

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = next(self._id_iter)

    def remove(self):
        del self.hooks_dict[self.id]
