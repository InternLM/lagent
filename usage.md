# How to Use Lagent

Lagent v1.0 is inspired by the design philosophy of PyTorch. This is a simple usage tutorial.

## Core Ideas

### Models as Agents

Agents use `AgentMessage` for communication.

```python
from typing import Dict, List
from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import VllModel, INTERNLM2_META

llm = VllmModel(
    path='Qwen/Qwen2-7B-Instruct',
    meta_template=INTERNLM2_META,
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>'],
    max_new_tokens=1024,
)
system_prompt = '你的回答只能从“典”、“孝”、“急”三个字中选一个。'
agent = Agent(llm, system_prompt)

user_msg = AgentMessage(sender='user', content='今天天气情况')
bot_msg = agent(user_msg)
print(bot_msg)
```

```
content='急' sender='Agent' formatted=None extra_info=None type=None receiver=None stream_state=<AgentStatusCode.END: 0>
```

### Memory as State

Both input and output messages will be added to the memory of `Agent` in each forward pass.

```python
memory: List[AgentMessage] = agent.memory.get_memory()
print(memory)
print('-' * 80)
dumped_memory: Dict[str, List[dict]] = agent.state_dict()
print(dumped_memory['memory'])
```

```
[AgentMessage(content='今天天气情况', sender='user', formatted=None, extra_info=None, type=None, receiver=None, stream_state=<AgentStatusCode.END: 0>), AgentMessage(content='急', sender='Agent', formatted=None, extra_info=None, type=None, receiver=None, stream_state=<AgentStatusCode.END: 0>)]
--------------------------------------------------------------------------------
[{'content': '今天天气情况', 'sender': 'user', 'formatted': None, 'extra_info': None, 'type': None, 'receiver': None, 'stream_state': <AgentStatusCode.END: 0>}, {'content': '急', 'sender': 'Agent', 'formatted': None, 'extra_info': None, 'type': None, 'receiver': None, 'stream_state': <AgentStatusCode.END: 0>}]
```


Clear the memory of this session(`session_id=0` by default):

```python
agent.memory.reset()
```

### Custom Message Aggregation

`DefaultAggregator` is called under the hood to assemble and convert `AgentMessage` to OpenAI message format.

```python
    def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> Union[AgentMessage, str]:
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.output_format,
            self.template,
        )
        ...
```

Implement a simple aggregator that can receive few-shots

```python
from typing import List, Union
from lagent.memory import Memory
from lagent.prompts import StrParser
from lagent.agents.aggregator import DefaultAggregator

class FewshotAggregator(DefaultAggregator):
    def __init__(self, few_shot: List[dict] = None):
        self.few_shot = few_shot or []

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  parser: StrParser = None,
                  system_instruction: Union[str, dict, List[dict]] = None) -> List[dict]:
        _message = []
        if system_instruction:
            _message.extend(
                self.aggregate_system_intruction(system_instruction))
        _message.extend(self.few_shot)
        messages = messages.get_memory()
        for message in messages:
            if message.sender == name:
                _message.append(
                    dict(role='assistant', content=str(message.content)))
            else:
                user_message = message.content
                if len(_message) > 0 and _message[-1]['role'] == 'user':
                    _message[-1]['content'] += user_message
                else:
                    _message.append(dict(role='user', content=user_message))
        return _message

agent = Agent(
    llm,
    aggregator=FewshotAggregator(
        [
            {"role": "user", "content": "今天天气"},
            {"role": "assistant", "content": "【晴】"},
        ]
    )
)
user_msg = AgentMessage(sender='user', content='昨天天气')
bot_msg = agent(user_msg)
print(bot_msg)
```

```
content='【多云转晴，夜间有轻微降温】' sender='Agent' formatted=None extra_info=None type=None receiver=None stream_state=<AgentStatusCode.END: 0>
```

### Flexible Response Formatting

In `AgentMessage`, `formatted` is reserved to store information parsed by `output_format` after forward pass.

```python
    def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> Union[AgentMessage, str]:
        ...
        llm_response = self.llm.chat(formatted_messages, **kwargs)
        if self.output_format:
            formatted_messages = self.output_format.parse_response(llm_response)
            return AgentMessage(
                sender=self.name,
                content=llm_response,
                formatted=formatted_messages,
            )
        ...
```

Use a tool parser as follows

```python
from lagent.prompts.parsers import ToolParser

system_prompt = "逐步分析并编写Python代码解决以下问题。"
parser = ToolParser(tool_type='code interpreter', begin='```python\n', end='\n```\n')
llm.gen_params['stop_words'].append('\n```\n')
agent = Agent(llm, system_prompt, output_format=parser)

user_msg = AgentMessage(
    sender='user',
    content='Marie is thinking of a multiple of 63, while Jay is thinking of a '
    'factor of 63. They happen to be thinking of the same number. There are '
    'two possibilities for the number that each of them is thinking of, one '
    'positive and one negative. Find the product of these two numbers.')
bot_msg = agent(user_msg)
print(bot_msg.model_dump_json(indent=4))
```

```
{
    "content": "首先，我们需要找出63的所有正因数和负因数。63的正因数可以通过分解63的质因数来找出，即\\(63 = 3^2 \\times 7\\)。因此，63的正因数包括1, 3, 7, 9, 21, 和 63。对于负因数，我们只需将上述正因数乘以-1。\n\n接下来，我们需要找出与63的正因数相乘的结果为63的数，以及与63的负因数相乘的结果为63的数。这可以通过将63除以每个正因数和负因数来实现。\n\n最后，我们将找到的两个数相乘得到最终答案。\n\n下面是Python代码实现：\n\n```python\ndef find_numbers():\n    # 正因数\n    positive_factors = [1, 3, 7, 9, 21, 63]\n    # 负因数\n    negative_factors = [-1, -3, -7, -9, -21, -63]\n    \n    # 找到与正因数相乘的结果为63的数\n    positive_numbers = [63 / factor for factor in positive_factors]\n    # 找到与负因数相乘的结果为63的数\n    negative_numbers = [-63 / factor for factor in negative_factors]\n    \n    # 计算两个数的乘积\n    product = positive_numbers[0] * negative_numbers[0]\n    \n    return product\n\nresult = find_numbers()\nprint(result)",
    "sender": "Agent",
    "formatted": {
        "tool_type": "code interpreter",
        "thought": "首先，我们需要找出63的所有正因数和负因数。63的正因数可以通过分解63的质因数来找出，即\\(63 = 3^2 \\times 7\\)。因此，63的正因数包括1, 3, 7, 9, 21, 和 63。对于负因数，我们只需将上述正因数乘以-1。\n\n接下来，我们需要找出与63的正因数相乘的结果为63的数，以及与63的负因数相乘的结果为63的数。这可以通过将63除以每个正因数和负因数来实现。\n\n最后，我们将找到的两个数相乘得到最终答案。\n\n下面是Python代码实现：\n\n",
        "action": "def find_numbers():\n    # 正因数\n    positive_factors = [1, 3, 7, 9, 21, 63]\n    # 负因数\n    negative_factors = [-1, -3, -7, -9, -21, -63]\n    \n    # 找到与正因数相乘的结果为63的数\n    positive_numbers = [63 / factor for factor in positive_factors]\n    # 找到与负因数相乘的结果为63的数\n    negative_numbers = [-63 / factor for factor in negative_factors]\n    \n    # 计算两个数的乘积\n    product = positive_numbers[0] * negative_numbers[0]\n    \n    return product\n\nresult = find_numbers()\nprint(result)",
        "status": 1
    },
    "extra_info": null,
    "type": null,
    "receiver": null,
    "stream_state": 0
}
```

### Consistency of Tool Calling

`ActionExecutor` uses the same communication data structure as `Agent`, but requires the content of input `AgentMessage` to be a dict containing:

- `name`: tool name, e.g. `'IPythonInterpreter'`, `'WebBrowser.search'`.
- `parameters`: keyword arguments of the tool API, e.g. `{'command': 'import math;math.sqrt(2)'}`, `{'query': ['recent progress in AI']}`.

You can register custom hooks for message conversion.

```python
from lagent.hooks import Hook
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage
from lagent.actions import ActionExecutor, IPythonInteractive

class CodeProcessor(Hook):
    def before_action(self, executor, message, session_id):
        message = message.copy(deep=True)
        message.content = dict(
            name='IPythonInteractive', parameters={'command': message.formatted['action']}
        )
        return message

    def after_action(self, executor, message, session_id):
        action_return = message.content
        if isinstance(action_return, ActionReturn):
            if action_return.state == ActionStatusCode.SUCCESS:
                response = action_return.format_result()
            else:
                response = action_return.errmsg
        else:
            response = action_return
        message.content = response
        return message

executor = ActionExecutor(actions=[IPythonInteractive()], hooks=[CodeProcessor()])
bot_msg = AgentMessage(
    sender='Agent',
    content='首先，我们需要...',
    formatted={
        'tool_type': 'code interpreter',
        'thought': '首先，我们需要...', 
        'action': 'def find_numbers():\n    # 正因数\n    positive_factors = [1, 3, 7, 9, 21, 63]\n    # 负因数\n    negative_factors = [-1, -3, -7, -9, -21, -63]\n    \n    # 找到与正因数相乘的结果为63的数\n    positive_numbers = [63 / factor for factor in positive_factors]\n    # 找到与负因数相乘的结果为63的数\n    negative_numbers = [-63 / factor for factor in negative_factors]\n    \n    # 计算两个数的乘积\n    product = positive_numbers[0] * negative_numbers[0]\n    \n    return product\n\nresult = find_numbers()\nprint(result)', 
        'status': 1
    })
executor_msg = executor(bot_msg)
print(executor_msg)
```

```
content='3969.0' sender='ActionExecutor' formatted=None extra_info=None type=None receiver=None stream_state=<AgentStatusCode.END: 0>
```

**For convenience, Lagent provides `InternLMActionProcessor` which is adapted to messages formatted by `ToolParser` as mentioned above.**

---

## Practice

- **Try to implement `forward` instead of `__call__` of subclasses unless neccesary.**
- **Always include the `session_id` argument explicitly, which is designed for isolation of memory and LLM requests in concurrency.**

### Single Agent

Math agents that solve problems by programming

```python
from lagent.agents.aggregator import InternLMToolAggregator

class Coder(Agent):
    def __init__(self, model_path, system_prompt, max_turn=3):
        llm = VllmModel(
            path=model_path,
            meta_template=INTERNLM2_META,
            tp=1,
            top_k=1,
            temperature=1.0,
            stop_words=['<|im_end|>'],
            max_new_tokens=1024,
        )
        self.agent = Agent(
            llm,
            system_prompt,
            output_format=ToolParser(
                tool_type='code interpreter', begin='```python\n', end='\n```\n'
            ),
            # `InternLMToolAggregator` is adapted to `ToolParser` for aggregating 
            # messages with tool invocations and execution results
            aggregator=InternLMToolAggregator(),
        )
        self.executor = ActionExecutor([IPythonInteractive()], hooks=[CodeProcessor()])
        self.max_turn = max_turn
        super().__init__()
    
    def forward(self, message: AgentMessage, session_id=0) -> AgentMessage:
        for _ in range(self.max_turn):
            message = self.agent(message, session_id=session_id)
            if message.formatted['tool_type'] is None:
                return message
            message = self.executor(message, session_id=session_id)
        return message

coder = Coder('Qwen/Qwen2-7B-Instruct', 'Solve the problem step by step with assistance of Python code')
query = AgentMessage(sender='user', content='Let $m$ and $n$ satisfy $mn=7$ and $m+n=8$. What is $|m-n|$?')
ans = coder(query)
print(ans.content)
print('-' * 80)
print(coder.state_dict()['agent.memory'])
```

### Multiple Agents

Blogging agents that improve writing quality by self-refinement

```python
class PrefixedMessageHook(Hook):
    def __init__(self, prefix: str, senders: list = None):
        self.prefix = prefix
        self.senders = senders or []

    def before_agent(self, agent, messages, session_id):
        for i, message in enumerate(messages):
            if message.sender in self.senders:
                message = message.copy(deep=True)
                message.content = self.prefix + message.content
                messages[i] = message
        return messages

class Blogger(Agent):
    def __init__(self, model_path, writer_prompt, critic_prompt, critic_prefix='', max_turn=3):
        llm = VllmModel(
            path=model_path,
            meta_template=INTERNLM2_META,
            tp=1,
            top_k=1,
            temperature=1.0,
            stop_words=['<|im_end|>'],
            max_new_tokens=1024,
        )
        self.writer = Agent(llm, writer_prompt, name='writer')
        self.critic = Agent(
            llm, critic_prompt, name='critic', hooks=[PrefixedMessageHook(critic_prefix, ['writer'])]
        )
        self.max_turn = max_turn
        super().__init__()
    
    def forward(self, message: AgentMessage, session_id=0) -> AgentMessage:
        for _ in range(self.max_turn):
            message = self.writer(message, session_id=session_id)
            message = self.critic(message, session_id=session_id)
        return self.writer(message, session_id=session_id)

blogger = Blogger(
    'Qwen/Qwen2-7B-Instruct',
    writer_prompt="You are an writing assistant tasked to write engaging blogpost. You try generate the best blogpost possible for the user's request. "
    "If the user provides critique, respond with a revised version of your previous attempts",
    critic_prompt="Generate critique and recommendations on the writing. Provide detailed recommendations, including requests for length, depth, style, etc..",
    critic_prefix='Reflect and provide critique on the following writing. \n\n',
)
user_msg = AgentMessage(
    sender='user',
    content="Write an engaging blogpost on the recent updates in AI. "
    "The blogpost should be engaging and understandable for general audience. "
    "Should have more than 3 paragraphes but no longer than 500 words."
)
bot_msg = blogger(user_msg)
print(bot_msg.content)
print('-' * 80)
for msg in blogger.state_dict()['writer.memory']:
    print('*' * 120)
    print(f'{msg["sender"]}:\n\n{msg["content"]}')
print('-' * 80)
for msg in blogger.state_dict()['critic.memory']:
    print('*' * 120)
    print(f'{msg["sender"]}:\n\n{msg["content"]}')
```
