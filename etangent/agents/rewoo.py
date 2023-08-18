import re

from etangent.actions import ActionExecutor
from etangent.schema import ActionReturn, ActionStatusCode, AgentReturn
from .base_agent import BaseAgent

CALL_PROTOCOL = """你是一个任务分解器, 你需要将用户的问题拆分成多个简单的子任务。
请拆分出多个子任务项，从而能够得到充分的信息以解决问题, 返回格式如下：
```
Plan: 当前子任务要解决的问题
#E[id] = 工具名称[工具参数]
Plan: 当前子任务要解决的问题
#E[id] = 工具名称[工具参数]
```
其中
1. #E[id] 用于存储Plan id的执行结果, 可被用作占位符。
2. 每个 #E[id] 所执行的内容应与当前Plan解决的问题严格对应。
3. 工具参数可以是正常输入text, 或是 #E[依赖的索引], 或是两者都可以。
4. 工具名称必须从一下工具中选择：
{tool_description}
注意: 每个Plan后有且仅能跟随一个#E。
开始！"""

SOLVER_PROMPT = """解决接下来的任务或者问题。为了帮助你，我们提供了一些相关的计划
和相应的解答。注意其中一些信息可能存在噪声，因此你需要谨慎的使用它们。\n
{question}\n{worker_log}\n现在开始回答这个任务或者问题。请直接回答这个问题，
不要包含其他不需要的文字。{question}\n
"""


class ReWOOProtocol:

    def __init__(
        self,
        response=dict(role='RESPONSE', begin='Response', end='\n'),
        thought=dict(role='THOUGHT', begin='Thought', end='\n'),
        call_protocol=CALL_PROTOCOL,
    ) -> None:
        self.call_protocol = call_protocol
        self.response = response
        self.thought = thought

    def format(self,
               chat_history,
               inner_step,
               action_executor: ActionExecutor,
               reformat_request='') -> list:
        call_protocol = self.call_protocol.format(
            tool_description=action_executor.get_actions_info(),
            response=self.response['begin'],
            # example=EXAMPLE,
        )
        formatted = []
        formatted.append(dict(role='system', content=call_protocol))
        formatted += chat_history
        formatted += inner_step
        if reformat_request != '':
            formatted.append(
                dict(
                    role='system',
                    content='回答格式错误: %s. 请重新重新回答: ' % reformat_request))
        return formatted

    def parse(
        self,
        message,
        action_executor: ActionExecutor,
    ):
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

    def format_response(self, thought_list, action_return_list):
        responses = ''
        for thought, action_return in zip(thought_list, action_return_list):
            response = self.thought['begin'] + ': ' + thought + self.thought[
                'end']
            response += self.response['begin'] + ': '
            if action_return.state == ActionStatusCode.SUCCESS:
                response += action_return.result['text']
            else:
                response += action_return.errmsg
            response += self.response['end']
            responses += response
        return responses


class ReWOO(BaseAgent):

    def __init__(self,
                 llm,
                 action_executor,
                 max_turn=2,
                 prompter=ReWOOProtocol()):
        super().__init__(
            llm=llm, action_executor=action_executor, prompter=prompter)

        self.max_turn = max_turn

    def chat(self, message):
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()

        # planner
        turn_id = 0
        reformat_request = ''
        while turn_id <= self.max_turn:
            prompt = self._prompter.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                reformat_request=reformat_request)
            response = self._llm.generate_from_template(prompt, 512)
            self._inner_history.append(
                dict(role='assistant', content=response))
            try:
                thoughts, actions, actions_input = self._prompter.parse(
                    response, self._action_executor)
                break
            except Exception as e:
                turn_id += 1
                reformat_request = str(e)

        if turn_id > self.max_turn:
            print('[Warning] Unable to parse LLM response for workers'
                  '[Warning] Directly request solver for question answer...')
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

        worker_log = self._prompter.format_response(thoughts, action_responses)
        self._inner_history.append(dict(role='system', content=worker_log))

        # solver
        solver_prompt = SOLVER_PROMPT.format(
            worker_log=worker_log,
            question=message,
        )
        final_response = self._llm.generate_from_template(solver_prompt, 512)
        self._inner_history.append(
            dict(role='assistant', content=final_response))
        agent_return.response = final_response
        return agent_return
