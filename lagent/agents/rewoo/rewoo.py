import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

from lagent.actions import ActionExecutor
from lagent.agents.base_agent import BaseAgent
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn
from .prompt_cn import (PLANER_PROMPT, REFORMAT_PROMPT, SOLVER_PROMPT,
                        WORKER_PROMPT)


class ReWOOProtocol:
    """A wrapper of ReWOO prompt which manages the response from LLM and
    generate desired prompts in a ReWOO format.

    Args:
        planner_prompt (str): prompt template for planner.
        worker_prompt (str): prompt template for workers/actions.
        solver_prompt (str): prompt template for solver.
        reformat_prompt (str): prompt template to regenerate
            response for LLM.
    """

    def __init__(
        self,
        planner_prompt: str = PLANER_PROMPT,
        worker_prompt: str = WORKER_PROMPT,
        solver_prompt: str = SOLVER_PROMPT,
        reformat_prompt: str = REFORMAT_PROMPT,
    ) -> None:
        self.planner_prompt = planner_prompt
        self.worker_prompt = worker_prompt
        self.solver_prompt = solver_prompt
        self.reformat_prompt = reformat_prompt

    def format_planner(self,
                       chat_history: List[Dict],
                       inner_step: List[Dict],
                       action_executor: ActionExecutor,
                       reformat_request: Optional[str] = '') -> List[Dict]:
        """Generate the planner prompt required by ReWOO.

        Args:
            chat_history (List[Dict]): The history log in previous runs.
            inner_step (List[Dict]): The log in the current run.
            action_executor (ActionExecutor): the action manager to execute
                 actions.
            reformat_request (str): the error feedback if the LLM fails to
                generate required format for planner.

        Returns:
            List[Dict]: ReWOO format prompt for planner.
        """
        planner_prompt = self.planner_prompt.format(
            tool_description=action_executor.get_actions_info(), )
        formatted = []
        formatted.append(dict(role='system', content=planner_prompt))
        formatted += chat_history
        formatted += inner_step
        if reformat_request != '':
            formatted.append(
                dict(
                    role='system',
                    content=self.reformat_prompt.format(
                        err_msg=reformat_request)))
        return formatted

    def parse_worker(
        self,
        message: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parse the LLM generated planner response and convert it into the
        worker format.

        Args:
            message (str): The response from LLM with ReWOO planner format.

        Returns:
            tuple: the return value is a tuple contains:
                - thought_list (List(str)): contain LLM thoughts of the user
                    request.
                - action_list (List(str)): contain actions scheduled by LLM.
                - action_input_list (List(str)): contain the required action
                     input for above actions.
        """
        action_list = []
        action_input_list = []
        thought_list = []
        thoughts = re.findall('Plan: (.+)', message)
        action_units = re.findall('#E[0-9]* = (.+)', message)
        assert len(thoughts) == len(action_units), \
            'Each Plan should only correspond to only ONE action'
        for thought, action_unit in zip(thoughts, action_units):
            action_name, action_input = re.findall(r'(.*?)\[(.*?)\]',
                                                   action_unit.strip())[0]
            action_list.append(action_name.strip())
            action_input_list.append(action_input.strip())
            thought_list.append(thought.strip())
        return thought_list, action_list, action_input_list

    def format_solver(
            self, question: str, thought_list: List[str],
            action_return_list: List[ActionReturn]) -> Tuple[str, str]:
        """Generate the prompt for solver in a ReWOO format.

        Args:
            question (str): The user request in the current run.
            thought_list (List[str]): thoughts generated from LLM for
                each action.
            action_return_list (List[ActionReturn]): action returns
                from workers.

        Returns:
            tuple: the return value is a tuple contains:
                - solver_prompt (str): the generated prompt for solver
                     in a ReWOO format.
                - worker_log (str): contain action responses from workers.
                    Used for inner log.
        """
        worker_log = ''
        for thought, action_return in zip(thought_list, action_return_list):
            if action_return.state == ActionStatusCode.SUCCESS:
                action_resp = action_return.result['text']
            else:
                action_resp = action_return.errmsg
            worker_response = self.worker_prompt.format(
                thought=thought, action_resp=action_resp)
            worker_log += worker_response
        solver_prompt = self.solver_prompt.format(
            question=question, worker_log=worker_log)
        return solver_prompt, worker_log


class ReWOO(BaseAgent):
    """An implementation of ReWOO (https://arxiv.org/abs/2305.18323)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as planner / solver.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReWOOProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReWOO protocol.
    """

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 action_executor: ActionExecutor,
                 protocol: ReWOOProtocol = ReWOOProtocol(),
                 max_turn: int = 2) -> None:
        super().__init__(
            llm=llm, action_executor=action_executor, protocol=protocol)

        self.max_turn = max_turn

    def chat(self, message: str) -> AgentReturn:
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()

        # planner
        turn_id = 0
        reformat_request = ''
        while turn_id < self.max_turn:
            planner_prompt = self._protocol.format_planner(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                reformat_request=reformat_request)
            response = self._llm.generate_from_template(planner_prompt, 512)
            self._inner_history.append(
                dict(role='assistant', content=response))
            try:
                thoughts, actions, actions_input = self._protocol.parse_worker(
                    response)
                break
            except Exception as e:
                turn_id += 1
                reformat_request = str(e)

        if turn_id >= self.max_turn:
            warnings.warn('\nUnable to parse LLM responses in %d turns, '
                          'directly request solver for question answer...' %
                          self.max_turn)
            actions = []
            thoughts = []
            action_responses = []
        # workers
        action_responses = []
        for action_id in range(len(actions)):
            # we need to change actions_input inplace
            prev_ptrs = re.findall(r'#E\d+', actions_input[action_id])
            for prev_ptr in prev_ptrs:
                ptr_num = int(prev_ptr.strip('#E')) - 1  # start from 0
                actions_input[action_id] = actions_input[action_id].replace(
                    prev_ptr, action_responses[ptr_num].result['text'])
            action_return: ActionReturn = self._action_executor(
                actions[action_id], actions_input[action_id])
            action_responses.append(action_return)

        solver_prompt, worker_log = self._protocol.format_solver(
            message, thoughts, action_responses)
        self._inner_history.append(dict(role='system', content=worker_log))

        final_response = self._llm.generate_from_template(solver_prompt, 512)
        self._inner_history.append(
            dict(role='assistant', content=final_response))
        agent_return.response = final_response
        return agent_return
