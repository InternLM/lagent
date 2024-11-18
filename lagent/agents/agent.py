import copy
import warnings
from collections import OrderedDict, UserDict, UserList, abc
from functools import wraps
from itertools import chain, repeat
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from lagent.agents.aggregator import DefaultAggregator
from lagent.hooks import Hook, RemovableHandle
from lagent.llms import BaseLLM
from lagent.memory import Memory, MemoryManager
from lagent.prompts.parsers import StrParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.schema import AgentMessage
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
        self.memory: MemoryManager = MemoryManager(memory) if memory else None
        self.output_format: StrParser = create_object(output_format)
        self.template = template
        self.description = description
        self.aggregator: DefaultAggregator = create_object(aggregator)
        self._hooks: Dict[int, Hook] = OrderedDict()
        if hooks:
            for hook in hooks:
                hook = create_object(hook)
                self.register_hook(hook)

    def update_memory(self, message, session_id=0):
        if self.memory:
            self.memory.add(message, session_id=session_id)

    def __call__(
        self,
        *message: Union[str, AgentMessage, List[AgentMessage]],
        session_id=0,
        **kwargs,
    ) -> AgentMessage:
        # message.receiver = self.name
        message = [
            AgentMessage(sender='user', content=m)
            if isinstance(m, str) else copy.deepcopy(m) for m in message
        ]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result
        self.update_memory(message, session_id=session_id)
        response_message = self.forward(
            *message, session_id=session_id, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
            )
        self.update_memory(response_message, session_id=session_id)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message, session_id)
            if result:
                response_message = result
        return response_message

    def forward(self,
                *message: AgentMessage,
                session_id=0,
                **kwargs) -> Union[AgentMessage, str]:
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.output_format,
            self.template,
        )
        llm_response = self.llm.chat(formatted_messages, **kwargs)
        if self.output_format:
            formatted_messages = self.output_format.parse_response(
                llm_response)
            return AgentMessage(
                sender=self.name,
                content=llm_response,
                formatted=formatted_messages,
            )
        return llm_response

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Agent):
            _agents = getattr(self, '_agents', OrderedDict())
            _agents[__name] = __value
            super().__setattr__('_agents', _agents)
        super().__setattr__(__name, __value)

    def state_dict(self, session_id=0):
        state_dict, stack = {}, [('', self)]
        while stack:
            prefix, node = stack.pop()
            key = prefix + 'memory'
            if node.memory is not None:
                if session_id not in node.memory.memory_map:
                    warnings.warn(f'No session id {session_id} in {key}')
                memory = node.memory.get(session_id)
                state_dict[key] = memory and memory.save() or []
            if hasattr(node, '_agents'):
                for name, value in reversed(node._agents.items()):
                    stack.append((prefix + name + '.', value))
        return state_dict

    def load_state_dict(self, state_dict: Dict, session_id=0):
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
                if session_id not in obj.memory.memory_map:
                    obj.memory.create_instance(session_id)
                obj.memory.memory_map[session_id].load(state_dict[key] or [])

    def register_hook(self, hook: Callable):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def reset(self,
              session_id=0,
              keypath: Optional[str] = None,
              recursive: bool = False):
        assert not (keypath and
                    recursive), 'keypath and recursive can\'t be used together'
        if keypath:
            keys, agent = keypath.split('.'), self
            for key in keys:
                agents = getattr(agent, '_agents', {})
                if key not in agents:
                    raise KeyError(f'No sub-agent named {key} in {agent}')
                agent = agents[key]
            agent.reset(session_id, recursive=False)
        else:
            if self.memory:
                self.memory.reset(session_id=session_id)
            if recursive:
                for agent in getattr(self, '_agents', {}).values():
                    agent.reset(session_id, recursive=True)

    def __repr__(self):

        def _rcsv_repr(agent, n_indent=1):
            res = agent.__class__.__name__ + (f"(name='{agent.name}')"
                                              if agent.name else '')
            modules = [
                f"{n_indent * '  '}({name}): {_rcsv_repr(agent, n_indent + 1)}"
                for name, agent in getattr(agent, '_agents', {}).items()
            ]
            if modules:
                res += '(\n' + '\n'.join(
                    modules) + f'\n{(n_indent - 1) * "  "})'
            elif not res.endswith(')'):
                res += '()'
            return res

        return _rcsv_repr(self)


class AsyncAgent(Agent):

    async def __call__(self,
                       *message: AgentMessage | List[AgentMessage],
                       session_id=0,
                       **kwargs) -> AgentMessage:
        message = [
            AgentMessage(sender='user', content=m)
            if isinstance(m, str) else copy.deepcopy(m) for m in message
        ]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result
        self.update_memory(message, session_id=session_id)
        response_message = await self.forward(
            *message, session_id=session_id, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
            )
        self.update_memory(response_message, session_id=session_id)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message, session_id)
            if result:
                response_message = result
        return response_message

    async def forward(self,
                      *message: AgentMessage,
                      session_id=0,
                      **kwargs) -> Union[AgentMessage, str]:
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.output_format,
            self.template,
        )
        llm_response = await self.llm.chat(formatted_messages, session_id,
                                           **kwargs)
        if self.output_format:
            formatted_messages = self.output_format.parse_response(
                llm_response)
            return AgentMessage(
                sender=self.name,
                content=llm_response,
                formatted=formatted_messages,
            )
        return llm_response


class Sequential(Agent):
    """Sequential is an agent container that forwards messages to each agent 
    in the order they are added."""

    def __init__(self, *agents: Union[Agent, AsyncAgent, Iterable], **kwargs):
        super().__init__(**kwargs)
        self._agents = OrderedDict()
        if not agents:
            raise ValueError('At least one agent should be provided')
        if isinstance(agents[0],
                      Iterable) and not isinstance(agents[0], Agent):
            if not agents[0]:
                raise ValueError('At least one agent should be provided')
            agents = agents[0]
        for key, agent in enumerate(agents):
            if isinstance(agents, Mapping):
                key, agent = agent, agents[agent]
            elif isinstance(agent, tuple):
                key, agent = agent
            self.add_agent(key, agent)

    def add_agent(self, name: str, agent: Union[Agent, AsyncAgent]):
        assert isinstance(
            agent, (Agent, AsyncAgent
                    )), f'{type(agent)} is not an Agent or AsyncAgent subclass'
        self._agents[str(name)] = agent

    def forward(self,
                *message: AgentMessage,
                session_id=0,
                exit_at: Optional[int] = None,
                **kwargs) -> AgentMessage:
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message, )
            message = agent(*message, session_id=session_id, **kwargs)
        return message

    def __getitem__(self, key):
        if isinstance(key, int) and key < 0:
            assert key >= -len(self), 'index out of range'
            key = len(self) + key
        return self._agents[str(key)]

    def __len__(self):
        return len(self._agents)


class AsyncSequential(Sequential, AsyncAgent):

    async def forward(self,
                      *message: AgentMessage,
                      session_id=0,
                      exit_at: Optional[int] = None,
                      **kwargs) -> AgentMessage:
        assert exit_at is None or exit_at >= 0, 'exit_at should be greater than or equal to 0'
        if exit_at is None:
            exit_at = len(self) - 1
        iterator = chain.from_iterable(repeat(self._agents.values()))
        for _ in range(exit_at + 1):
            agent = next(iterator)
            if isinstance(message, AgentMessage):
                message = (message, )
            message = await agent(*message, session_id=session_id, **kwargs)
        return message


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
                for k, item in (self.data.items() if isinstance(
                        self.data, abc.Mapping) else enumerate(self.data)):
                    if isinstance(self.data,
                                  abc.Mapping) and not isinstance(k, str):
                        _backup(data)
                        raise KeyError(
                            f'agent name should be a string, got {type(k)}')
                    if isinstance(k, str) and '.' in k:
                        _backup(data)
                        raise KeyError(
                            f'agent name can\'t contain ".", got {k}')
                    if not isinstance(item, (Agent, AsyncAgent)):
                        _backup(data)
                        raise TypeError(
                            f'{type(item)} is not an Agent or AsyncAgent subclass'
                        )
                    agents[str(k)] = item
                self._agents = agents
                return ret

            return wrapped_func

        for method in [
                'append', 'sort', 'reverse', 'pop', 'clear', 'update',
                'insert', 'extend', 'remove', '__init__', '__setitem__',
                '__delitem__', '__add__', '__iadd__', '__radd__', '__mul__',
                '__imul__', '__rmul__'
        ]:
            if hasattr(cls, method):
                setattr(cls, method, wrap_api(getattr(cls, method)))


class AgentList(Agent, UserList, AgentContainerMixin):

    def __init__(self,
                 agents: Optional[Iterable[Union[Agent, AsyncAgent]]] = None):
        Agent.__init__(self, memory=None)
        UserList.__init__(self, agents)
        self.name = None


class AgentDict(Agent, UserDict, AgentContainerMixin):

    def __init__(self,
                 agents: Optional[Mapping[str, Union[Agent,
                                                     AsyncAgent]]] = None):
        Agent.__init__(self, memory=None)
        UserDict.__init__(self, agents)
        self.name = None
