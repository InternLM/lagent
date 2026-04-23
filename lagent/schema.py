from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field


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

    def format_result(self) -> str:
        """Concatenate items in result."""
        result = []
        for item in self.result or []:
            if item['type'] == 'text':
                result.append(item['content'])
            else:
                result.append(f"[{item['type']}]({item['content']})")
        result = '\n'.join(result)
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
    thinking: Optional[str] = None
    content_ids: Optional[List[int]] = Field(default=None, repr=False)
    content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    thinking_ids: Optional[List[int]] = Field(default=None, repr=False)
    thinking_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    raw_content: Optional[str] = None
    raw_content_ids: Optional[List[int]] = Field(default=None, repr=False)
    raw_content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    sender: str = 'user'
    tool_calls: Optional[List[dict]] = None
    tool_calls_ids: Optional[List[str]] = None
    formatted: Optional[Any] = None
    extra_info: dict = Field(default_factory=dict)
    type: Optional[str] = None
    receiver: Optional[str] = None
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END
    finish_reason: Optional[str] = None
    uid: Union[int, str] = Field(default_factory=lambda: uuid4().hex, repr=False)
    env_info: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None

    @classmethod
    def from_model_response(cls, response: Union[ChatCompletion, dict], sender: str) -> "AgentMessage":
        """Convert model response (ChatCompletion object or model_dump dict) to AgentMessage."""
        if isinstance(response, dict):
            choice = response['choices'][0]
            msg = choice['message']
            finish_reason = choice.get('finish_reason')
            # tool_calls_raw = msg.get('tool_calls')  # list of dicts or None
            return cls(
                sender=sender,
                content=msg.get('content') or "",
                thinking=msg.get('reasoning_content'),
                raw_content=msg.get('raw_content'),
                content_ids=msg.get('content_ids'),
                content_logprobs=msg.get('content_logprobs'),
                thinking_ids=msg.get('reasoning_content_ids'),
                thinking_logprobs=msg.get('reasoning_content_logprobs'),
                raw_content_ids=msg.get('raw_content_ids'),
                raw_content_logprobs=msg.get('raw_content_logprobs'),
                extra_info=msg.get('extra_info') or {},
                tool_calls=msg.get('tool_calls'),
                # tool_calls=[tc['function'] for tc in tool_calls_raw] if tool_calls_raw else None,
                # tool_calls_ids=[tc['id'] for tc in tool_calls_raw] if tool_calls_raw else None,
                stream_state=(
                    ModelStatusCode.SESSION_OUT_OF_LIMIT if finish_reason == 'length' else ModelStatusCode.END
                ),
                finish_reason=finish_reason,
            )
        # ChatCompletion object (or subclass)
        chat_message = response.choices[0].message
        tool_calls = chat_message.tool_calls and [tool_call.model_dump() for tool_call in chat_message.tool_calls]
        return cls(
            sender=sender,
            content=chat_message.content or "",
            thinking=getattr(chat_message, 'reasoning_content', None),
            raw_content=getattr(chat_message, 'raw_content', None),
            content_ids=getattr(chat_message, 'content_ids', None),
            content_logprobs=getattr(chat_message, 'content_logprobs', None),
            thinking_ids=getattr(chat_message, 'reasoning_content_ids', None),
            thinking_logprobs=getattr(chat_message, 'reasoning_content_logprobs', None),
            raw_content_ids=getattr(chat_message, 'raw_content_ids', None),
            raw_content_logprobs=getattr(chat_message, 'raw_content_logprobs', None),
            extra_info=getattr(chat_message, 'extra_info', {}) or {},
            tool_calls=tool_calls,
            # tool_calls=[tool_call['function'] for tool_call in tool_calls] if tool_calls else None,
            # tool_calls_ids=[tool_call['id'] for tool_call in tool_calls] if tool_calls else None,
            stream_state=(
                ModelStatusCode.SESSION_OUT_OF_LIMIT
                if response.choices[0].finish_reason == 'length'
                else ModelStatusCode.END
            ),
            finish_reason=response.choices[0].finish_reason,
        )

    def to_model_request(self, role: str = 'assistant') -> dict:
        """Convert AgentMessage to model request dict."""
        msg = {'role': role, 'content': self.content}
        # tool_calls = [
        #     {'id': tool_call_id, 'function': tool_call, 'type': 'function'}
        #     for tool_call, tool_call_id in zip(self.tool_calls or [], self.tool_calls_ids or [])
        # ]
        # if tool_calls:
        #     msg['tool_calls'] = tool_calls
        for key in [
            'tool_calls',
            'raw_content',
            'content_ids',
            'content_logprobs',
            'raw_content_ids',
            'raw_content_logprobs',
            'extra_info',
        ]:
            if getattr(self, key, None) is not None:
                msg[key] = getattr(self, key)
        for key in ['thinking', 'thinking_ids', 'thinking_logprobs']:
            if getattr(self, key, None) is not None:
                msg[key.replace("thinking", "reasoning_content")] = getattr(self, key)
        return msg
