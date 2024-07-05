from itertools import count


class Hook:

    def forward_pre_hook(
        self,
        agent,
        message,
    ):
        pass

    def forward_post_hook(
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
