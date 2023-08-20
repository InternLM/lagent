from typing import Optional

from etangent.actions.base_action import BaseAction
from etangent.schema import ActionReturn, ActionStatusCode, ActionValidCode


class NoAction(BaseAction):
    """This is a no action class, which is used to return error message when
    the response does not follow the format.

    Args:
        err_msg (str): The error message. Defaults to
            'Please follow the format'.
    """

    def __init__(self, err_msg: str = 'Please follow the format', **kwargs):

        super().__init__(enable=False, **kwargs)
        self._err_msg = err_msg

    def __call__(self, err_msg: Optional[str] = None):
        """Return the error message.

        Args:
            err_msg (str, optional): The error message. If err_msg is not None,
                it will be returned, otherwise the default error message will
                be returned. Defaults to None.

        Returns:
            ActionReturn: The action return.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=err_msg),
            type=self.name,
            errmsg=err_msg if err_msg else self._err_msg,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR)
        return action_return
