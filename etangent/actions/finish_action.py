from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode
from .base_action import BaseAction


class FinishAction(BaseAction):

    def __call__(self, parameter):
        action_return = ActionReturn(
            url=None,
            args=dict(text=parameter),
            result=dict(text=parameter),
            type=self.name,
            valid=ActionValidCode.FINISH,
            state=ActionStatusCode.SUCCESS)
        return action_return
