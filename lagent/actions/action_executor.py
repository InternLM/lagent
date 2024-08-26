import inspect
from collections import OrderedDict
from typing import Callable, Dict, List, Union

from lagent.actions.base_action import BaseAction
from lagent.actions.builtin_actions import FinishAction, InvalidAction, NoAction
from lagent.hooks import Hook, RemovableHandle
from lagent.schema import ActionReturn, ActionValidCode, AgentMessage, FunctionCall
from lagent.utils import create_object


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
        invalid_action: BaseAction = dict(type=InvalidAction),
        no_action: BaseAction = dict(type=NoAction),
        finish_action: BaseAction = dict(type=FinishAction),
        finish_in_action: bool = False,
        hooks: List[Dict] = None,
    ):

        if not isinstance(actions, list):
            actions = [actions]
        finish_action = create_object(finish_action)
        if finish_in_action:
            actions.append(finish_action)
        for i, action in enumerate(actions):
            actions[i] = create_object(action)
        self.actions = {action.name: action for action in actions}

        self.invalid_action = create_object(invalid_action)
        self.no_action = create_object(no_action)
        self.finish_action = finish_action
        self._hooks: Dict[int, Hook] = OrderedDict()
        if hooks:
            for hook in hooks:
                hook = create_object(hook)
                self.register_hook(hook)

    def description(self) -> List[Dict]:
        actions = []
        for action_name, action in self.actions.items():
            if action.is_toolkit:
                for api in action.description['api_list']:
                    api_desc = api.copy()
                    api_desc['name'] = f"{action_name}.{api_desc['name']}"
                    actions.append(api_desc)
            else:
                action_desc = action.description.copy()
                actions.append(action_desc)
        return actions

    def __contains__(self, name: str):
        return name in self.actions

    def keys(self):
        return list(self.actions.keys())

    def __setitem__(self, name: str, action: Union[BaseAction, Dict]):
        action = create_object(action)
        self.actions[action.name] = action

    def __delitem__(self, name: str):
        del self.actions[name]

    def forward(self, name, parameters, **kwargs) -> ActionReturn:
        action_name, api_name = (
            name.split('.') if '.' in name else (name, 'run'))
        action_return: ActionReturn = ActionReturn()
        if action_name not in self:
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

    def __call__(self,
                 message: AgentMessage,
                 session_id=0,
                 **kwargs) -> AgentMessage:
        # message.receiver = self.name
        for hook in self._hooks.values():
            result = hook.before_action(self, message, session_id)
            if result:
                message = result

        assert isinstance(message.content, FunctionCall) or (
            isinstance(message.content, dict) and 'name' in message.content
            and 'parameters' in message.content)
        if isinstance(message.content, dict):
            name = message.content.get('name')
            parameters = message.content.get('parameters')
        else:
            name = message.content.name
            parameters = message.content.parameters

        response_message = self.forward(
            name=name, parameters=parameters, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.__class__.__name__,
                content=response_message,
            )

        for hook in self._hooks.values():
            result = hook.after_action(self, response_message, session_id)
            if result:
                response_message = result
        return response_message

    def register_hook(self, hook: Callable):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle


class AsyncActionExecutor(ActionExecutor):

    async def forward(self, name, parameters, **kwargs) -> ActionReturn:
        action_name, api_name = (
            name.split('.') if '.' in name else (name, 'run'))
        action_return: ActionReturn = ActionReturn()
        if action_name not in self:
            if name == self.no_action.name:
                action_return = self.no_action(parameters)
            elif name == self.finish_action.name:
                action_return = self.finish_action(parameters)
            else:
                action_return = self.invalid_action(parameters)
        else:
            action = self.actions[action_name]
            if inspect.iscoroutinefunction(action.__call__):
                action_return = await action(parameters, api_name)
            else:
                action_return = action(parameters, api_name)
            action_return.valid = ActionValidCode.OPEN
        return action_return

    async def __call__(self,
                       message: AgentMessage,
                       session_id=0,
                       **kwargs) -> AgentMessage:
        # message.receiver = self.name
        for hook in self._hooks.values():
            if inspect.iscoroutinefunction(hook.before_action):
                result = await hook.before_action(self, message, session_id)
            else:
                result = hook.before_action(self, message, session_id)
            if result:
                message = result

        assert isinstance(message.content, FunctionCall) or (
            isinstance(message.content, dict) and 'name' in message.content
            and 'parameters' in message.content)
        if isinstance(message.content, dict):
            name = message.content.get('name')
            parameters = message.content.get('parameters')
        else:
            name = message.content.name
            parameters = message.content.parameters

        response_message = await self.forward(
            name=name, parameters=parameters, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.__class__.__name__,
                content=response_message,
            )

        for hook in self._hooks.values():
            if inspect.iscoroutinefunction(hook.after_action):
                result = await hook.after_action(self, response_message,
                                                 session_id)
            else:
                result = hook.after_action(self, response_message, session_id)
            if result:
                response_message = result
        return response_message
