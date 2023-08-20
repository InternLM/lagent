from unittest import TestCase, mock

from lagent.actions import ActionExecutor
from lagent.actions.llm_qa import LLMQA
from lagent.actions.serper_search import SerperSearch
from lagent.agents.rewoo import ReWOO, ReWOOProtocol
from lagent.schema import ActionReturn, ActionStatusCode


class TestReWOO(TestCase):

    @mock.patch.object(SerperSearch, 'run')
    @mock.patch.object(LLMQA, 'run')
    @mock.patch.object(ReWOOProtocol, 'parse_worker')
    def test_normal_chat(self, mock_parse_worker_func, mock_qa_func,
                         mock_search_func):
        mock_model = mock.Mock()
        mock_model.generate_from_template.return_value = 'LLM response'

        mock_parse_worker_func.return_value = (['Thought1', 'Thought2'
                                                ], ['LLMQA', 'SerperSearch'],
                                               ['abc', 'abc'])

        search_return = ActionReturn(args=None)
        search_return.state = ActionStatusCode.SUCCESS
        search_return.result = dict(text='search_return')
        mock_search_func.return_value = search_return

        qa_return = ActionReturn(args=None)
        qa_return.state = ActionStatusCode.SUCCESS
        qa_return.result = dict(text='qa_return')
        mock_qa_func.return_value = qa_return

        chatbot = ReWOO(
            llm=mock_model,
            action_executor=ActionExecutor(actions=[
                LLMQA(mock_model),
                SerperSearch(api_key=''),
            ]))
        agent_return = chatbot.chat('abc')
        self.assertEqual(agent_return.response, 'LLM response')

    def test_parse_worker(self):
        prompt = ReWOOProtocol()
        message = """
        Plan: a.
        #E1 = tool1["a"]
        #E2 = tool2["b"]
        """
        try:
            thoughts, actions, actions_input = prompt.parse_worker(message)
        except Exception as e:
            self.assertEqual(
                'Each Plan should only correspond to only ONE action', str(e))
        else:
            self.assertFalse(
                True, 'it should raise exception when the format is incorrect')

        message = """
        Plan: a.
        #E1 = tool1("a")
        Plan: b.
        #E2 = tool2["b"]
        """
        try:
            thoughts, actions, actions_input = prompt.parse_worker(message)
        except Exception as e:
            self.assertIsInstance(e, BaseException)
        else:
            self.assertFalse(
                True, 'it should raise exception when the format is incorrect')

        message = """
        Plan: a.
        #E1 = tool1["a"]
        Plan: b.
        #E2 = tool2["b"]
        """
        try:
            thoughts, actions, actions_input = prompt.parse_worker(message)
        except Exception:
            self.assertFalse(
                True,
                'it should not raise exception when the format is correct')
        self.assertEqual(thoughts, ['a.', 'b.'])
        self.assertEqual(actions, ['tool1', 'tool2'])
        self.assertEqual(actions_input, ['"a"', '"b"'])
