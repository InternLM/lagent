import copy
import warnings
from collections import OrderedDict, UserDict, UserList, abc
from functools import wraps
from itertools import chain, repeat
from typing import Any, AsyncGenerator, Callable, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union

from lagent.agents.aggregator import DefaultAggregator
from lagent.hooks import Hook, RemovableHandle
from lagent.llms import BaseLLM
from lagent.memory import Memory
from lagent.prompts.parsers import StrParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.schema import AgentMessage, ModelStatusCode
from lagent.utils import create_object


class Agent:
    """Agent is the basic unit of the system. It is responsible for
    communicating with the LLM, managing the memory, and handling the
    message aggregation and parsing. It can also be extended with hooks

    Args:
        llm (Union[BaseLLM, Dict]): The language model used by the agent.
        template (Union[PromptTemplate, str]): The template used to format the
            messages.
        memory (Dict): The memory used by the agent.
        output_format (Dict): The output format used by the agent.
        aggregator (Dict): The aggregator used by the agent.
        name (Optional[str]): The name of the agent.
        description (Optional[str]): The description of the agent.
        hooks (Optional[Union[List[Dict], Dict]]): The hooks used by the agent.

    Returns:
        AgentMessage: The response message.
    """

    def __init__(
        self,
        llm: Union[BaseLLM, Dict] = None,
        template: Union[PromptTemplate, str, dict, List[dict]] = None,
        memory: Dict = dict(type=Memory),
        output_format: Optional[Dict] = None,
        aggregator: Dict = dict(type=DefaultAggregator),
        name: Optional[str] = None,
        description: Optional[str] = None,
        hooks: Optional[Union[List[Dict], Dict]] = None,
    ):
        self.name = name or self.__class__.__name__
        self.llm: BaseLLM = create_object(llm)
        self.memory: Memory = create_object(memory)
        self.output_format: StrParser = create_object(output_format)
        self.template = template
        self.description = description
        self.aggregator: DefaultAggregator = create_object(aggregator)
        self._hooks: Dict[int, Hook] = OrderedDict()
        if hooks:
            for hook in hooks:
                hook = create_object(hook)
                self.register_hook(hook)
        self._scroll_mode: bool = False

    def __call__(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        # message.receiver = self.name
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message)
            if result:
                message = result

        # resume aborted rollout
        _message = self._scroll_buffer(message[-1])
        if _message is not None:
            if _message.finish_reason != 'abort':
                _message = copy.deepcopy(_message)
                for hook in self._hooks.values():
                    result = hook.after_agent(self, _message)
                    if result:
                        _message = result
                return _message
            message[-1].extra_info['partial_response'] = _message
        else:
            self.memory and self.memory.add(message)
        response_message = self.forward(*message, **kwargs)
        if _message and _message.finish_reason == 'abort':
            message[-1].extra_info.pop('partial_response', None)
        if not isinstance(response_message, AgentMessage):
            if isinstance(response_message, str):
                response_message = AgentMessage(sender=self.name, content=response_message)
            else:
                response_message = AgentMessage.from_model_response(response_message, self.name)
        self.memory and self.memory.add(response_message)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        return response_message

    def forward(self, *message: AgentMessage, **kwargs) -> Union[AgentMessage, str]:
        formatted_messages, tools = self.aggregator.aggregate(
            self.memory, self.name, self.output_format, self.template
        )
        llm_response = self.llm.chat(formatted_messages, tools=tools, **kwargs)
        if self.output_format:
            formatted_messages = self.output_format.parse_response(llm_response)
            return AgentMessage(sender=self.name, content=llm_response, formatted=formatted_messages)
        return llm_response

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Agent):
            _agents = getattr(self, '_agents', OrderedDict())
            _agents[__name] = __value
            super().__setattr__('_agents', _agents)
        super().__setattr__(__name, __value)

    def state_dict(self, prefix='', destination=None) -> Dict:
        if destination is None:
            destination = {}
        if self.memory is not None:
            saved_memory = self.memory and self.memory.save() or []
            destination.update({prefix + 'memory': saved_memory})
        for name, agent in getattr(self, '_agents', {}).items():
            if isinstance(agent, Agent):
                agent.state_dict(destination=destination, prefix=prefix + name + ".")
        return destination

    def load_state_dict(self, state_dict: Dict):
        _state_dict = self.state_dict()
        missing_keys = set(_state_dict) - set(state_dict)
        if missing_keys:
            raise KeyError(f'Missing keys: {missing_keys}')
        extra_keys = set(state_dict) - set(_state_dict)
        if extra_keys:
            warnings.warn(f'Mismatch keys which are not used: {extra_keys}')
        for key in _state_dict:
            obj = self
            for attr in key.split('.')[:-1]:
                if isinstance(obj, AgentList):
                    assert attr.isdigit()
                    obj = obj[int(attr)]
                elif isinstance(obj, AgentDict):
                    obj = obj[attr]
                else:
                    obj = getattr(obj, attr)
            if obj.memory is not None:
                obj.memory.load(state_dict[key] or [])

    def register_hook(self, hook: Callable):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def reset(self, keypath: Optional[str] = None, recursive: bool = False):
        assert not (keypath and recursive), 'keypath and recursive can\'t be used together'
        if keypath:
            keys, agent = keypath.split('.'), self
            for key in keys:
                agents = getattr(agent, '_agents', {})
                if key not in agents:
                    raise KeyError(f'No sub-agent named {key} in {agent}')
                agent = agents[key]
            agent.reset(recursive=False)
        else:
            if self.memory:
                self.memory.reset()
            if recursive:
                for agent in getattr(self, '_agents', {}).values():
                    agent.reset(recursive=True)

    def get_messages(self, keypath: Optional[str] = None) -> List[dict]:
        """Get OpenAI format messages from memory.

        Args:
            keypath (Optional[str]): The keypath of the sub-agent to get messages from. Default is None.

        Returns:
            List[dict]: The messages from the memory including the sub-agent's system prompt.
        """
        if keypath:
            keys, agent = keypath.split('.'), self
            for key in keys:
                agents = getattr(agent, '_agents', {})
                if key not in agents:
                    raise KeyError(f'No sub-agent named {key} in {agent}')
                agent = agents[key]
            return agent.get_messages()
        if self.aggregator:
            return self.aggregator.aggregate(self.memory, self.name, self.output_format, self.template)
        raise ValueError(f'{self.name} has no aggregator to get messages')

    def _scroll_buffer(self, message, hash_func=lambda m: m.uid):
        if not self.memory:
            return
        mem = self.memory.get_memory()
        finish_reasons = [m.finish_reason for m in mem]
        if not ('abort' in finish_reasons or self._scroll_mode):
            return
        if not self._scroll_mode:
            self._enable_scroll_mode(recursive=True)
        aborted_msg_idx = finish_reasons.index('abort') if 'abort' in finish_reasons else len(mem) - 1
        self.memory.delete(range(aborted_msg_idx + 1, len(mem)))
        enc = hash_func(message)
        for i in range(0, aborted_msg_idx):
            if hash_func(mem[i]) == enc:
                ret = mem[i + 1]
                if i + 1 == aborted_msg_idx:
                    if ret.finish_reason == 'abort':
                        self.memory.delete(aborted_msg_idx)
                    self._disable_scroll_mode()
                return ret
        self._disable_scroll_mode(recursive=True)

    def _enable_scroll_mode(self, recursive=False):
        self._scroll_mode = True
        if recursive:
            for sub_agent in getattr(self, '_agents', {}).values():
                sub_agent._enable_scroll_mode(True)

    def _disable_scroll_mode(self, recursive=False):
        self._scroll_mode = False
        if recursive:
            for sub_agent in getattr(self, '_agents', {}).values():
                sub_agent._disable_scroll_mode(True)

    def __repr__(self):

        def _rcsv_repr(agent, n_indent=1):
            res = agent.__class__.__name__ + (f"(name='{agent.name}')" if agent.name else '')
            modules = [
                f"{n_indent * '  '}({name}): {_rcsv_repr(agent, n_indent + 1)}"
                for name, agent in getattr(agent, '_agents', {}).items()
            ]
            if modules:
                res += '(\n' + '\n'.join(modules) + f'\n{(n_indent - 1) * "  "})'
            elif not res.endswith(')'):
                res += '()'
            return res

        return _rcsv_repr(self)


class AsyncAgentMixin:

    async def __call__(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message)
            if result:
                message = result

        # resume aborted rollout
        _message = self._scroll_buffer(message[-1])
        if _message is not None:
            if _message.finish_reason != 'abort':
                _message = copy.deepcopy(_message)
                for hook in self._hooks.values():
                    result = hook.after_agent(self, _message)
                    if result:
                        _message = result
                return _message
            message[-1].extra_info['partial_response'] = _message
        else:
            self.memory and self.memory.add(message)
        response_message = await self.forward(*message, **kwargs)
        if _message and _message.finish_reason == 'abort':
            message[-1].extra_info.pop('partial_response', None)
        if not isinstance(response_message, AgentMessage):
            if isinstance(response_message, str):
                response_message = AgentMessage(sender=self.name, content=response_message)
            else:
                response_message = AgentMessage.from_model_response(response_message, self.name)
        self.memory and self.memory.add(response_message)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        return response_message

    async def forward(self, *message: AgentMessage, **kwargs) -> Union[AgentMessage, str]:
        formatted_messages, tools = self.aggregator.aggregate(
            self.memory, self.name, self.output_format, self.template
        )
        llm_response = await self.llm.chat(formatted_messages, tools=tools, **kwargs)
        if self.output_format:
            formatted_messages = self.output_format.parse_response(llm_response)
            return AgentMessage(sender=self.name, content=llm_response, formatted=formatted_messages)
        return llm_response


class AsyncAgent(AsyncAgentMixin, Agent):
    """Asynchronous variant of the Agent class"""

    pass


class StreamingAgentMixin:
    """Component that makes agent calling output a streaming response."""

    def __call__(self, *message: AgentMessage, **kwargs) -> Generator[AgentMessage, None, None]:
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message)
            if result:
                message = result
        self.memory.add(message)
        response_message = AgentMessage(sender=self.name, content="")
        for response_message in self.forward(*message, **kwargs):
            if not isinstance(response_message, AgentMessage):
                model_state, response = response_message
                response_message = AgentMessage(sender=self.name, content=response, stream_state=model_state)
            yield response_message.model_copy()
        self.memory.add(response_message)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        yield response_message

    def forward(
        self, *message: AgentMessage, **kwargs
    ) -> Generator[Union[AgentMessage, Tuple[ModelStatusCode, str]], None, None]:
        formatted_messages = self.aggregator.aggregate(self.memory, self.name, self.output_format, self.template)
        for model_state, response, *_ in self.llm.stream_chat(formatted_messages, **kwargs):
            yield (
                AgentMessage(
                    sender=self.name,
                    content=response,
                    formatted=self.output_format.parse_response(response),
                    stream_state=model_state,
                )
                if self.output_format
                else (model_state, response)
            )


class AsyncStreamingAgentMixin:
    """Component that makes asynchronous agent calling output a streaming response."""

    async def __call__(self, *message: AgentMessage, **kwargs) -> AsyncGenerator[AgentMessage, None]:
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message)
            if result:
                message = result
        self.memory.add(message)
        response_message = AgentMessage(sender=self.name, content="")
        async for response_message in self.forward(*message, **kwargs):
            if not isinstance(response_message, AgentMessage):
                model_state, response = response_message
                response_message = AgentMessage(sender=self.name, content=response, stream_state=model_state)
            yield response_message.model_copy()
        self.memory.add(response_message)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        yield response_message

    async def forward(
        self, *message: AgentMessage, **kwargs
    ) -> AsyncGenerator[Union[AgentMessage, Tuple[ModelStatusCode, str]], None]:
        formatted_messages = self.aggregator.aggregate(self.memory, self.name, self.output_format, self.template)
        async for model_state, response, *_ in self.llm.stream_chat(formatted_messages, **kwargs):
            yield (
                AgentMessage(
                    sender=self.name,
                    content=response,
                    formatted=self.output_format.parse_response(response),
                    stream_state=model_state,
                )
                if self.output_format
                else (model_state, response)
            )


class StreamingAgent(StreamingAgentMixin, Agent):
    """Streaming variant of the Agent class"""

    pass


class AsyncStreamingAgent(AsyncStreamingAgentMixin, Agent):
    """Streaming variant of the AsyncAgent class"""

    pass


class Sequential(Agent):
    """Sequential is an agent container that forwards messages to each agent
    in the order they are added."""

    def __init__(self, *agents: Union[Agent, Iterable], **kwargs):
        super().__init__(**kwargs)
        self._agents = OrderedDict()
        if not agents:
            raise ValueError('At least one agent should be provided')
        if isinstance(agents[0], Iterable) and not isinstance(agents[0], Agent):
            if not agents[0]:
                raise ValueError('At least one agent should be provided')
            agents = agents[0]
        for key, agent in enumerate(agents):
            if isinstance(agents, Mapping):
                key, agent = agent, agents[agent]
            elif isinstance(agent, tuple):
                key, agent = agent
            self.add_agent(key, agent)

    def add_agent(self, name: str, agent: Agent):
        assert isinstance(agent, Agent), f'{type(agent)} is not an Agent subclass'
        self._agents[str(name)] = agent

    def forward(self, *message: AgentMessage, exit_at: Optional[int] = None, **kwargs) -> AgentMessage:
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            message = agent(*message, **kwargs)
        return message

    def __getitem__(self, key):
        if isinstance(key, int) and key < 0:
            assert key >= -len(self), 'index out of range'
            key = len(self) + key
        return self._agents[str(key)]

    def __len__(self):
        return len(self._agents)


class AsyncSequential(AsyncAgentMixin, Sequential):

    async def forward(self, *message: AgentMessage, exit_at: Optional[int] = None, **kwargs) -> AgentMessage:
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            message = await agent(*message, **kwargs)
        return message


class StreamingSequential(StreamingAgentMixin, Sequential):
    """Streaming variant of the Sequential class"""

    def forward(self, *message: AgentMessage, exit_at: Optional[int] = None, **kwargs):
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            for message in agent(*message, **kwargs):
                yield message


class AsyncStreamingSequential(AsyncStreamingAgentMixin, Sequential):
    """Streaming variant of the AsyncSequential class"""

    async def forward(self, *message: AgentMessage, exit_at: Optional[int] = None, **kwargs):
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message,)
            async for message in agent(*message, **kwargs):
                yield message


class AgentContainerMixin:

    def __init_subclass__(cls):
        super().__init_subclass__()

        def wrap_api(func):

            @wraps(func)
            def wrapped_func(self, *args, **kwargs):
                data = self.data.copy() if hasattr(self, 'data') else None

                def _backup(d):
                    if d is None:
                        self.data.clear()
                    else:
                        self.data = d

                ret = func(self, *args, **kwargs)
                agents = OrderedDict()
                for k, item in self.data.items() if isinstance(self.data, abc.Mapping) else enumerate(self.data):
                    if isinstance(self.data, abc.Mapping) and not isinstance(k, str):
                        _backup(data)
                        raise KeyError(f'agent name should be a string, got {type(k)}')
                    if isinstance(k, str) and '.' in k:
                        _backup(data)
                        raise KeyError(f'agent name can\'t contain ".", got {k}')
                    if not isinstance(item, Agent):
                        _backup(data)
                        raise TypeError(f'{type(item)} is not an Agent subclass')
                    agents[str(k)] = item
                self._agents = agents
                return ret

            return wrapped_func

        # fmt: off
        for method in [
            'append', 'sort', 'reverse', 'pop', 'clear', 'update',
            'insert', 'extend', 'remove', '__init__', '__setitem__',
            '__delitem__', '__add__', '__iadd__', '__radd__', '__mul__',
            '__imul__', '__rmul__'
        ]:
            if hasattr(cls, method):
                setattr(cls, method, wrap_api(getattr(cls, method)))


class AgentList(Agent, UserList, AgentContainerMixin):

    def __init__(self, agents: Optional[Iterable[Agent]] = None):
        Agent.__init__(self, memory=None)
        UserList.__init__(self, agents)
        self.name = None


class AgentDict(Agent, UserDict, AgentContainerMixin):

    def __init__(self, agents: Optional[Mapping[str, Agent]] = None):
        Agent.__init__(self, memory=None)
        UserDict.__init__(self, agents)
        self.name = None
