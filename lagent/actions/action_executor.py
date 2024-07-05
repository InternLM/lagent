from typing import Dict, List, Union

from lagent.actions.base_action import BaseAction
from lagent.registry import TOOL_REGISTRY, ObjectFactory
from lagent.schema import ActionReturn, ActionValidCode, AgentMessage, FunctionCall


@TOOL_REGISTRY.register
class ActionExecutor:
    """The action executor class.

    Args:
        actions (Union[BaseAction, List[BaseAction]]): The action or actions.
        invalid_action (BaseAction, optional): The invalid action. Defaults to
            InvalidAction().
        no_action (BaseAction, optional): The no action.
            Defaults to NoAction().
        finish_action (BaseAction, optional): The finish action. Defaults to
            FinishAction().
        finish_in_action (bool, optional): Whether the finish action is in the
            action list. Defaults to False.
    """

    def __init__(
        self,
        actions: Union[BaseAction, List[BaseAction], Dict, List[Dict]],
        invalid_action: BaseAction = dict(type='InvalidAction'),
        no_action: BaseAction = dict(type='NoAction'),
        finish_action: BaseAction = dict(type='FinishAction'),
        finish_in_action: bool = False,
        hooks: List[Dict] = None,
    ):

        if not isinstance(actions, list):
            actions = [actions]
        finish_action = ObjectFactory.create(finish_action, TOOL_REGISTRY)
        if finish_in_action:
            actions.append(finish_action)
        for i, action in enumerate(actions):
            actions[i] = ObjectFactory.create(action, TOOL_REGISTRY)
        self.actions = {action.name: action for action in actions}

        self.invalid_action = ObjectFactory.create(invalid_action,
                                                   TOOL_REGISTRY)
        self.no_action = ObjectFactory.create(no_action, TOOL_REGISTRY)
        self.finish_action = finish_action
        self._hook = hooks

    def get_actions_info(self) -> List[Dict]:
        actions = []
        for action_name, action in self.actions.items():
            if not action.enable:
                continue
            if action.is_toolkit:
                for api in action.description['api_list']:
                    api_desc = api.copy()
                    api_desc['name'] = f"{action_name}.{api_desc['name']}"
                    actions.append(api_desc)
            else:
                action_desc = action.description.copy()
                actions.append(action_desc)
        return actions

    def is_valid(self, name: str):
        return name in self.actions and self.actions[name].enable

    def action_names(self, only_enable: bool = True):
        if only_enable:
            return [k for k, v in self.actions.items() if v.enable]
        else:
            return list(self.actions.keys())

    def add_action(self, action: Union[BaseAction, Dict]):
        action = ObjectFactory.create(action, TOOL_REGISTRY)
        self.actions[action.name] = action

    def del_action(self, name: str):
        if name in self.actions:
            del self.actions[name]

    def forward(self, name, parameters, **kwargs) -> ActionReturn:

        action_name, api_name = (
            name.split('.') if '.' in name else (name, 'run'))
        action_return: ActionReturn = ActionReturn()
        if not self.is_valid(action_name):
            if name == self.no_action.name:
                action_return = self.no_action(parameters)
            elif name == self.finish_action.name:
                action_return = self.finish_action(parameters)
            else:
                action_return = self.invalid_action(parameters)
        else:
            action_return = self.actions[action_name](parameters, api_name)
            action_return.valid = ActionValidCode.OPEN
        return action_return

    def __call__(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        # message.receiver = self.name
        for hook in self._hooks.values():
            result = hook.forward_pre_hook(message)
            if result:
                message = result

        assert isinstance(message.content, FunctionCall) or (
            isinstance(message.content, dict) and 'name' in message.content
            and 'parameters' in message.content)
        if isinstance(message.content, dict):
            name = message.content['name']
            parameters = message.content['parameters']
        else:
            name = message.content.name
            parameters = message.content.parameters

        response_message = self.forward(
            name=name, parameters=parameters, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
            )

        for hook in self._hooks.values():
            result = hook.forward_post_hook(message)
            if result:
                message = result
        return response_message
