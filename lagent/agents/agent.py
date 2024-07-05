from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

from lagent.agents.aggregator import DefaultAggregator
from lagent.agents.hooks import Hook
from lagent.agents.hooks.hook import RemovableHandle
from lagent.llms.base_llm import BaseModel as BaseLLM
from lagent.memory.base_memory import Memory
from lagent.prompts.parsers import StrParser
from lagent.prompts.prompt_template import PromptTemplate
from lagent.registry import (AGENT_REGISTRY, AGGREGATOR_REGISTRY,
                             HOOK_REGISTRY, LLM_REGISTRY, MEMORY_REGISTRY,
                             PARSER_REGISTRY, AutoRegister, ObjectFactory,
                             RegistryMeta)
from lagent.schema import AgentMessage


class Agent(metaclass=AutoRegister(AGENT_REGISTRY, RegistryMeta)):
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
        output_format: Dict = dict(type='StrParser'),
        aggregator: Dict = dict(type='DefaultAggregator'),
        name: Optional[str] = None,
        description: Optional[str] = None,
        hooks: Optional[Union[List[Dict], Dict]] = None,
    ):
        self.name = name
        self.llm: BaseLLM = ObjectFactory.create(llm, LLM_REGISTRY)
        self.memory: Memory = ObjectFactory.create(memory, MEMORY_REGISTRY)
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

    def update_memory(self, message):
        self.memory.add(message)

    def parse_response(self, response: str) -> str:
        if self.output_format:
            return self.output_format.parse(response)
        return response

    def __call__(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        # message.receiver = self.name
        for hook in self._hooks.values():
            result = hook.forward_pre_hook(message)
            if result:
                message = result

        self.update_memory(message)
        response_message = self.forward(*message, **kwargs)
        if not isinstance(response_message, AgentMessage):
            response_message = AgentMessage(
                sender=self.name,
                content=response_message,
            )
        self.update_memory(message)
        for hook in self._hooks.values():
            result = hook.forward_post_hook(message)
            if result:
                message = result
        return response_message

    def forward(self, *message: AgentMessage, **kwargs) -> Any:

        formatted_messages = self.aggregator.aggregate(
            self.memory,
            self.name,
            self.template,
        )
        llm_response = self.llm.chat(formatted_messages)
        parsed_response = self.parse_response(llm_response)
        return llm_response, parsed_response

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Agent):
            getattr(self, '_agents', OrderedDict())[__name] = __value
        super().__setattr__(__name, __value)

    def state_dict(self):

        state_dict = {
            'name': self.name,
            'llm': self.llm,
            'template': self.template,
            'memory': self.memory,
            'output_format': self.output_format,
            'description': self.description,
        }
        state_dict['memory'] = self.memory.save_state()

        return state_dict

    def load_state_dict(self, state_dict: Dict):
        self.name = state_dict['name']
        self.llm = state_dict['llm']
        self.template = state_dict['template']
        self.memory = Memory.load_state(state_dict['memory'])
        self.output_format = state_dict['output_format']
        self.description = state_dict['description']

    def register_hook(self, hook: Callable):
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle
