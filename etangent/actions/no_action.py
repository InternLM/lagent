from etangent.actions.base_action import BaseAction
from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode


class NoAction(BaseAction):

    def __init__(self,
                 enable=False,
                 description='please follow the format',
                 **kwargs):

        super().__init__(enable=enable, description=description, **kwargs)

    def __call__(self, parameter):
        action_return = ActionReturn(
            url=None,
            args=dict(text=parameter),
            type=self.name,
            errmsg=self._description,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR)
        return action_return
