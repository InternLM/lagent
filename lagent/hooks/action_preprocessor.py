from copy import deepcopy

from lagent.schema import ActionReturn, ActionStatusCode, FunctionCall
from .hook import Hook


class ActionPreprocessor(Hook):
    """The ActionPreprocessor is a hook that preprocesses the action message
    and postprocesses the action return message.

    """

    def before_action(self, executor, message, session_id):
        assert isinstance(message.formatted, FunctionCall) or (
            isinstance(message.formatted, dict) and 'name' in message.content
            and 'parameters' in message.formatted) or (
                'action' in message.formatted
                and 'parameters' in message.formatted['action']
                and 'name' in message.formatted['action'])
        if isinstance(message.formatted, dict):
            name = message.formatted.get('name',
                                         message.formatted['action']['name'])
            parameters = message.formatted.get(
                'parameters', message.formatted['action']['parameters'])
        else:
            name = message.formatted.name
            parameters = message.formatted.parameters
        message.content = dict(name=name, parameters=parameters)
        return message

    def after_action(self, executor, message, session_id):
        action_return = message.content
        if isinstance(action_return, ActionReturn):
            if action_return.state == ActionStatusCode.SUCCESS:
                response = action_return.format_result()
            else:
                response = action_return.errmsg
        else:
            response = action_return
        message.content = response
        return message


class InternLMActionProcessor(ActionPreprocessor):

    def __init__(self, code_parameter: str = 'command'):
        self.code_parameter = code_parameter

    def before_action(self, executor, message, session_id):
        message = deepcopy(message)
        assert isinstance(message.formatted, dict) and set(
            message.formatted).issuperset(
                {'tool_type', 'thought', 'action', 'status'})
        if isinstance(message.formatted['action'], str):
            # encapsulate code interpreter arguments
            action_name = next(iter(executor.actions))
            parameters = {self.code_parameter: message.formatted['action']}
            if action_name in ['AsyncIPythonInterpreter']:
                parameters['session_id'] = session_id
            message.formatted['action'] = dict(
                name=action_name, parameters=parameters)
        return super().before_action(executor, message, session_id)
