from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode
from .base_action import BaseAction


class FinishAction(BaseAction):
    """This is a finish action class, which is used to return the final
    result."""

    def __call__(self, response: str) -> ActionReturn:
        """Return the final result.

        Args:
            response (str): The final result.

        Returns:
            ActionReturn: The action return.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=response),
            result=dict(text=response),
            type=self.name,
            valid=ActionValidCode.FINISH,
            state=ActionStatusCode.SUCCESS)
        return action_return
