from itertools import count


class Hook:

    def before_agent(
        self,
        agent,
        message,
        session_id,
    ):
        pass

    def after_agent(
        self,
        agent,
        message,
        session_id,
    ):
        pass

    def before_action(
        self,
        executor,
        message,
        session_id,
    ):
        pass

    def after_action(
        self,
        executor,
        message,
        session_id,
    ):
        pass


class RemovableHandle:
    _id_iter = count(0)

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = next(self._id_iter)

    def remove(self):
        del self.hooks_dict[self.id]
