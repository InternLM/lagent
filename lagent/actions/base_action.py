from typing import Optional

from lagent.schema import ActionReturn


class BaseAction:
    """Base class for all actions.

    Args:
        description (str, optional): The description of the action. Defaults to
            None.
        name (str, optional): The name of the action. If None, the name will
            be class name. Defaults to None.
        enable (bool, optional): Whether the action is enabled. Defaults to
            True.
        disable_description (str, optional): The description of the action when
            it is disabled. Defaults to None.
    """

    def __init__(self,
                 description: Optional[str] = None,
                 name: Optional[str] = None,
                 enable: bool = True,
                 disable_description: Optional[str] = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self._description = description
        self._disable_description = disable_description
        self._enable = enable

    def __call__(self, *args, **kwargs) -> ActionReturn:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.name}:{self.description}'

    def __str__(self):
        return self.__repr__()

    def run(self, *args, **kwargs) -> ActionReturn:
        return self.__call__(*args, **kwargs)

    @property
    def enable(self):
        return self._enable

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        if self.enable:
            return self._description
        else:
            return self._disable_description
