from etangent.actions.base_action import BaseAction
from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode


class InvalidAction(BaseAction):

    def __init__(self,
                 enable=False,
                 description='你的返回格式不符合要求，请按照要求的格式回复',
                 **kwargs):

        super().__init__(enable=enable, description=description, **kwargs)

    def __call__(self, parameter):
        action_return = ActionReturn(
            url=None,
            args=dict(text=parameter),
            errmsg=self._description,
            type=self.name,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR)
        return action_return
