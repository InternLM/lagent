from lagent.actions import ActionExecutor
from lagent.actions.base_action import BaseAction
from lagent.llms.base_llm import BaseModel
from lagent.schema import AgentReturn


class BaseAgent:
    """BaseAgent is the base class of all agents.

    Args:
        llm (BaseModel): the language model.
        action_executor (ActionExecutor): the action executor.
        protocol (object): the protocol of the agent, which is used to
            generate the prompt of the agent and parse the response from
            the llm.
    """

    def __init__(self, llm: BaseModel, action_executor: ActionExecutor,
                 protocol: object) -> None:
        self._llm = llm
        self._action_executor = action_executor
        self._protocol = protocol

    def add_action(self, action: BaseAction) -> None:
        """Add an action to the action executor.

        Args:
            action (BaseAction): the action to be added.
        """
        self._action_executor.add_action(action)

    def del_action(self, name: str) -> None:
        """Delete an action from the action executor.

        Args:
            name (str): the name of the action to be deleted.
        """
        self._action_executor.del_action(name)

    def chat(self, message: str, **kwargs) -> AgentReturn:
        raise NotImplementedError
