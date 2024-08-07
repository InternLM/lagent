import copy
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

from lagent.agents.aggregator import DefaultAggregator
from lagent.hooks import Hook, RemovableHandle
from lagent.llms import BaseLLM
from lagent.memory import MemoryManager
from lagent.prompts.parsers import StrParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.registry import (
    AGENT_REGISTRY,
    AGGREGATOR_REGISTRY,
    HOOK_REGISTRY,
    LLM_REGISTRY,
    PARSER_REGISTRY,
    AutoRegister,
    ObjectFactory,
)
from lagent.schema import AgentMessage


class Agent(metaclass=AutoRegister(AGENT_REGISTRY)):
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
        template: Union[PromptTemplate, str] = None,
        memory: Dict = dict(type='Memory'),
        output_format: Optional[Dict] = None,
        aggregator: Dict = dict(type='DefaultAggregator'),
        name: Optional[str] = None,
        description: Optional[str] = None,
        hooks: Optional[Union[List[Dict], Dict]] = None,
    ):
        self.name = name or self.__class__.__name__
        self.llm: BaseLLM = ObjectFactory.create(llm, LLM_REGISTRY)

        self.memory: MemoryManager = MemoryManager(memory) if memory else None
        self.output_format: StrParser = ObjectFactory.create(
            output_format, PARSER_REGISTRY)
        self.template = template
        self.description = description
        self.aggregator: DefaultAggregator = ObjectFactory.create(
            aggregator, AGGREGATOR_REGISTRY)
        self._hooks: Dict[int, Hook] = OrderedDict()
        if hooks:
            for hook in hooks:
                hook = ObjectFactory.create(hook, HOOK_REGISTRY)
                self.register_hook(hook)

    def update_memory(self, message, session_id=0):
        if self.memory:
            self.memory.add(message, session_id=session_id)

    def parse_response(self, response: str) -> str:
        if self.output_format:
            return self.output_format.parse(response)
        return response

    def __call__(self,
                 *message: Union[AgentMessage, List[AgentMessage]],
                 session_id=0,
                 **kwargs) -> AgentMessage:
        # message.receiver = self.name
        for hook in self._hooks.values():
            message = copy.deepcopy(message)
            result = hook.before_agent(self, message)
            if result:
                message = result

        self.update_memory(message, session_id=session_id)
        response_message = self.forward(
            *message, session_id=session_id, **kwargs)
        if not isinstance(response_message, AgentMessage):
            parsed_response = self.parse_response(response_message)
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
                formatted=parsed_response,
            )
        self.update_memory(response_message, session_id=session_id)
        for hook in self._hooks.values():
            response_message = copy.deepcopy(response_message)
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        return response_message

    def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> Any:

        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.template,
        )
        llm_response = self.llm.chat(formatted_messages, session_id, **kwargs)

        return llm_response

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Agent):
            getattr(self, '_agents', OrderedDict())[__name] = __value
        super().__setattr__(__name, __value)

    def state_dict(self, session_id=0):

        state_dict = {
            'name': self.name,
            'llm': self.llm,
            'template': self.template,
            'memory': self.memory.get(session_id).save_state()
            if self.memory else None,
            'output_format': self.output_format,
            'description': self.description,
        }

        return state_dict

    def load_state_dict(self, state_dict: Dict, session_id=0):
        self.name = state_dict['name']
        self.llm = state_dict['llm']
        self.template = state_dict['template']
        self.memory = self.memory.get(session_id).load_state(
            state_dict['memory']) if state_dict['memory'] else None
        self.output_format = state_dict['output_format']
        self.description = state_dict['description']

    def register_hook(self, hook: Callable):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle


class AsyncAgent(Agent):

    async def __call__(self,
                       *message: AgentMessage | List[AgentMessage],
                       session_id=0,
                       **kwargs) -> AgentMessage:
        for hook in self._hooks.values():
            message = copy.deepcopy(message)
            result = hook.before_agent(self, message)
            if result:
                message = result

        self.update_memory(message, session_id=session_id)
        response_message = await self.forward(
            *message, session_id=session_id, **kwargs)
        if not isinstance(response_message, AgentMessage):
            parsed_response = self.parse_response(response_message)
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
                formatted=parsed_response,
            )
        self.update_memory(response_message, session_id=session_id)
        for hook in self._hooks.values():
            response_message = copy.deepcopy(response_message)
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        return response_message

    async def forward(self,
                      *message: AgentMessage,
                      session_id=0,
                      **kwargs) -> Any:

        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.template,
        )
        llm_response = await self.llm.chat(formatted_messages, session_id,
                                           **kwargs)

        return llm_response
