from etangent.actions import ActionExecutor


class BaseAgent:

    def __init__(self, llm, action_executor: ActionExecutor, prompter):

        self._session_history = []
        self._llm = llm
        self._action_executor = action_executor
        self._prompter = prompter

    def add_action(self, tools):
        self._action_executor.add_action(tools)

    def del_action(self, name):
        self._action_executor.del_action(name)

    def chat(self, message):
        raise NotImplementedError

    def save_session(self):
        raise NotImplementedError

    def load_session(self):
        raise NotImplementedError

    @property
    def session_history(self):
        return self._session_history
