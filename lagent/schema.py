from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from .utils import truncate_text


def enum_dict_factory(inputs):
    inputs = [(i[0], i[-1].value) if isinstance(i[-1], IntEnum) else i for i in inputs]
    return dict(inputs)


def dataclass2dict(data):
    return asdict(data, dict_factory=enum_dict_factory)


@dataclass
class FunctionCall:
    name: str
    parameters: Union[Dict, str]


class ActionStatusCode(IntEnum):
    ING = 1
    SUCCESS = 0
    HTTP_ERROR = -1000  # http error
    ARGS_ERROR = -1001  # parameter error
    API_ERROR = -1002  # unknown error


class ActionValidCode(IntEnum):
    FINISH = 1
    OPEN = 0
    CLOSED = -1
    INVALID = -2
    ABSENT = -3  # NO ACTION


@dataclass
class ActionReturn:
    args: Optional[dict] = None
    url: Optional[str] = None
    type: Optional[str] = None
    result: Optional[List[dict]] = None
    errmsg: Optional[str] = None
    state: Union[ActionStatusCode, int] = ActionStatusCode.SUCCESS
    thought: Optional[str] = None
    valid: Optional[ActionValidCode] = ActionValidCode.OPEN
    tool_call_id: Optional[str] = None
    max_tool_response_length: Optional[int] = None
    tool_response_truncate_side: Optional[Literal['left', 'right', 'middle']] = None

    def format_result(self) -> str:
        """Concatenate items in result, falling back to errmsg when result is empty."""
        result = []
        for item in self.result or []:
            if item['type'] == 'text':
                result.append(item['content'])
            else:
                result.append(f"[{item['type']}]({item['content']})")
        result = '\n'.join(result)
        if not result and self.errmsg:
            return self.errmsg
        if self.max_tool_response_length is not None and len(result) > self.max_tool_response_length:
            result = truncate_text(
                result, max_num=self.max_tool_response_length, side=self.tool_response_truncate_side or 'middle'
            )
        return result


# need to integrate int, so asdict can convert AgentStatusCode to int
class ModelStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference


class AgentStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference
    PLUGIN_START = 3  # start tool
    PLUGIN_END = 4  # finish tool
    PLUGIN_RETURN = 5  # finish tool
    CODING = 6  # start python
    CODE_END = 7  # end python
    CODE_RETURN = 8  # python return


class AgentMessage(BaseModel):
    content: Any
    sender: str = 'user'
    role: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    formatted: Optional[Any] = None
    extra_info: dict = Field(default_factory=dict)
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END
    finish_reason: Optional[str] = None
    uid: Union[int, str] = Field(default_factory=lambda: uuid4().hex, repr=False)
    env_info: Optional[Dict[str, Any]] = None

    def model_post_init(self, _):
        if isinstance(self.content, list):
            self.role = 'tool'
        elif self.sender not in ['system', 'user']:
            self.role = 'assistant'
        else:
            self.role = self.sender

    @classmethod
    def from_model_response(cls, response: Union[ChatCompletion, dict], sender: str) -> "AgentMessage":
        """Convert model response (ChatCompletion object or model_dump dict) to AgentMessage."""
        if not isinstance(response, dict):
            response = response.model_dump()

        choice = response['choices'][0]
        msg = choice.get('message', {})
        finish_reason = choice.get('finish_reason')
        return cls(
            sender=sender,
            content=msg.get('content') or "",
            reasoning_content=msg.get('reasoning_content'),
            tool_calls=msg.get('tool_calls'),
            extra_info=msg.get('extra_info') or {},
            stream_state=choice.get('stream_state', AgentStatusCode.END),
            finish_reason=finish_reason,
        )

    def to_model_request(self, role: Optional[str] = None) -> Union[dict, List[dict]]:
        """Convert AgentMessage to model request dict."""
        final_role = role if role is not None else self.role

        if final_role == 'tool' and isinstance(self.content, list):
            res = []
            for item in self.content:
                if isinstance(item, dict):
                    item = ActionReturn(**item)
                assert isinstance(item, ActionReturn), f"Expected item to be ActionReturn, but got {type(item)}"
                res.append(
                    dict(
                        role=final_role,
                        tool_call_id=item.tool_call_id,
                        content=item.format_result(),
                        name=item.type,
                        extra_info=self.extra_info,
                    )
                )
            return res

        msg = {'role': final_role, 'content': self.content}
        if final_role != 'assistant':
            msg['extra_info'] = self.extra_info
        if self.reasoning_content is not None:
            msg['reasoning_content'] = self.reasoning_content
        if self.tool_calls is not None:
            msg['tool_calls'] = self.tool_calls
        return msg
