from unittest import TestCase

from lagent.actions.builtin_actions import FinishAction, InvalidAction, NoAction
from lagent.schema import ActionStatusCode


class TestFinishAction(TestCase):

    def test_call(self):
        action = FinishAction()
        response = 'finish'
        action_return = action(response)
        self.assertEqual(action_return.state, ActionStatusCode.SUCCESS)
        self.assertDictEqual(action_return.result, dict(text='finish'))


class TestInvalidAction(TestCase):

    def test_call(self):
        action = InvalidAction()
        response = 'invalid'
        action_return = action(response)
        self.assertEqual(action_return.state, ActionStatusCode.API_ERROR)
        self.assertEqual(action_return.errmsg, response)

        action = InvalidAction(err_msg='error')
        action_return = action()
        self.assertEqual(action_return.state, ActionStatusCode.API_ERROR)
        self.assertEqual(action_return.errmsg, 'error')


class TestNoAction(TestCase):

    def test_call(self):
        action = NoAction()
        response = 'no'
        action_return = action(response)
        self.assertEqual(action_return.state, ActionStatusCode.API_ERROR)
        self.assertEqual(action_return.errmsg, response)

        action = NoAction(err_msg='error')
        action_return = action()
        self.assertEqual(action_return.state, ActionStatusCode.API_ERROR)
        self.assertEqual(action_return.errmsg, 'error')
