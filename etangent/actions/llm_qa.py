from etangent.schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction

DEFAULT_DESCRIPTION = """一个像你一样的大语言预训练模型，当你需要获得一些常识或简单世界知识时可以问它。
当你很有把握自己直接解决问题时可以优先使用它。输入应该是一个询问语句, 且每个问题尽可能简单。
"""


class LLMQA(BaseAction):
    """An LLM Wrapper as BaseAction type."""

    def __init__(self,
                 llm,
                 description=DEFAULT_DESCRIPTION,
                 name=None,
                 enable=True,
                 disable_description=None):
        super().__init__(description, name, enable, disable_description)

        self._llm = llm

    def __call__(self, query):
        tool_return = ActionReturn(url=None, args=None)
        try:
            response = self._llm.generate_from_template(query, 512)
            tool_return.result = dict(text=str(response))
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.result = dict(text=str(e))
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
