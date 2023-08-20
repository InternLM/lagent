from typing import Optional

from etangent.actions.base_action import BaseAction
from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode


class InvalidAction(BaseAction):
    """This is a invalid action class, which is used to return error message
    when the action is invalid.

    Args:
        err_msg (str): The error message. Defaults to 'The action is invalid,
            please check the action name'.

    Returns:
        ActionReturn: The action return.
    """

    def __init__(self,
                 err_msg:
                 str = 'The action is invalid, please check the action name.',
                 **kwargs) -> None:

        super().__init__(enable=False, **kwargs)
        self._err_msg = err_msg

    def __call__(self, err_msg: Optional[str] = None):
        """Return the error message.

        Args:
            err_msg (str, optional): The error message. If err_msg is not None,
                it will be returned, otherwise the default error message will
                be returned. Defaults to None.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=err_msg),
            errmsg=self._description,
            type=self.name,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR)
        return action_return
