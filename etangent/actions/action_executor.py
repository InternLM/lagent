from typing import List, Union

from etangent.schema import ActionReturn, ActionValidCode
from .base_action import BaseAction
from .finish_action import FinishAction
from .invalid_action import InvalidAction
from .no_action import NoAction


class ActionExecutor:

    def __init__(self,
                 actions: Union[BaseAction, List[BaseAction]],
                 invalid_action=InvalidAction(),
                 no_action=NoAction(),
                 finish_action=FinishAction()):
        if isinstance(actions, BaseAction):
            actions = [actions]

        for action in actions:
            assert isinstance(action, BaseAction), \
                f'action must be BaseAction, but got {type(action)}'
        self.actions = {action.name: action for action in actions}
        self.invalid_action = invalid_action
        self.no_action = no_action
        self.finish_action = finish_action

    def get_actions_info(self, only_enable: bool = True):
        if only_enable:
            return {
                k: v.description
                for k, v in self.actions.items() if v.enable
            }
        else:
            return {k: v.description for k, v in self.actions.items()}

    def is_valid(self, name: str):
        return name in self.actions and self.actions[name].enable

    def action_names(self, only_enable: bool = True):
        if only_enable:
            return [k for k, v in self.actions.items() if v.enable]
        else:
            return list(self.actions.keys())

    def add_action(self, action: BaseAction):
        assert isinstance(action, BaseAction), \
            f'action must be BaseAction, but got {type(action)}'
        self.actions[action.name] = action

    def del_action(self, name: str):
        if name in self.actions:
            del self.actions[name]

    def __call__(self, name, command) -> ActionReturn:
        if not self.is_valid(name):
            if name == self.no_action.name:
                action_return = self.no_action.run(command)
            elif name == self.finish_action.name:
                action_return = self.finish_action.run(command)
            else:
                action_return = self.invalid_action(command)
        else:
            action_return = self.actions[name].run(command)
            action_return.valid = ActionValidCode.OPEN
        return action_return
