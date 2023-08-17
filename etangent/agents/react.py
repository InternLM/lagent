from etangent.actions import ActionExecutor
from etangent.schema import ActionReturn, ActionStatusCode, AgentReturn
from .base_agent import BaseAgent

CALL_PROTOCOL = """你是一个可以调用外部工具的助手，可以使用的工具包括：
{tool_description}
如果使用工具请遵循以下格式回复：
```
{thought}思考你当前步骤需要解决什么问题，是否需要使用工具
{action}工具名称，你的工具必须从 [{action_names}] 选择
{action_input}工具输入参数
```
工具返回按照以下格式回复：
```
{response}调用工具后的结果
```
如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
```
{thought}给出最终答案的思考过程
{finish}最终答案
```
开始!"""


class ReACTProtocol:

    def __init__(self,
                 thought=dict(
                     role='THOUGHT',
                     begin='Thought:',
                     end='\n',
                     belong='assistant'),
                 action=dict(role='ACTION', begin='Action:', end='\n'),
                 action_input=dict(
                     role='ARGS', begin='ActionInput:', end='\n'),
                 response=dict(role='RESPONSE', begin='Response:', end='\n'),
                 finish=dict(role='FINISH', begin='FinalAnswer:', end='\n'),
                 call_protocol=CALL_PROTOCOL,
                 force_stop='你需要基于历史消息返回一个最终结果') -> None:
        self.call_protocol = call_protocol
        self.force_stop = force_stop
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.response = response
        self.finish = finish

    def format(self,
               chat_history,
               inner_step,
               action_executor: ActionExecutor,
               force_stop=False) -> list:
        call_protocol = self.call_protocol.format(
            tool_description=action_executor.get_actions_info(),
            action_names=action_executor.action_names(),
            thought=self.thought['begin'],
            action=self.action['begin'],
            action_input=self.action_input['begin'],
            response=self.response['begin'],
            finish=self.finish['begin'],
        )
        formatted = []
        formatted.append(dict(role='system', content=call_protocol))
        formatted += chat_history
        formatted += inner_step
        if force_stop:
            formatted.append(dict(role='system', content=self.force_stop))
        return formatted

    def parse(
        self,
        message,
        action_executor: ActionExecutor,
    ):
        import re
        thought = message.split(self.action['begin'])[0]
        thought = thought.split(self.thought['begin'])[-1]
        thought = thought.split(self.finish['begin'])[0]
        if self.finish['begin'] in message:
            final_answer = message.split(self.finish['begin'])[-1]
            return thought, action_executor.finish_action.name, final_answer

        action_regex = f"{self.action['begin']}(.*?)\n"
        args_regex = f"{self.action_input['begin']}(.*)"
        action_match = re.findall(action_regex, message)
        if not action_match:
            return thought, action_executor.no_action.name, ''
        action = action_match[-1]
        arg_match = re.findall(args_regex, message, re.DOTALL)

        if not arg_match:
            return thought, action_executor.no_action.name, ''
        action_input = arg_match[-1]
        return thought, action.strip(), action_input.strip().strip('"')

    def format_response(self, action_return: ActionReturn):
        if action_return.state == ActionStatusCode.SUCCESS:
            response = action_return.result['text']
        else:
            response = action_return.errmsg
        return self.response['begin'] + response + self.response['end']


class ReACT(BaseAgent):

    def __init__(self,
                 llm,
                 action_executor,
                 prompter=ReACTProtocol(),
                 max_turn=2):
        self.max_turn = max_turn
        super().__init__(
            llm=llm, action_executor=action_executor, prompter=prompter)

    def chat(self, message):
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()
        force_stop = False
        default_response = '对不起，我无法回答你的问题'
        for turn in range(self.max_turn):
            prompt = self._prompter.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                force_stop=force_stop)
            response = self._llm.generate_from_template(prompt, 512)
            self._inner_history.append(
                dict(role='assistant', content=response))
            thought, action, action_input = self._prompter.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            action_return.thought = thought
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.result['text']
                return agent_return
            self._inner_history.append(
                dict(
                    role='system',
                    content=self._prompter.format_response(action_return)))
            if turn == self.max_turn - 1:
                force_stop = True
        agent_return.response = default_response
        # only append the user and final response
        self._session_history.append(dict(role='user', content=message))
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return
