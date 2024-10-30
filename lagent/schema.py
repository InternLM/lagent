from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


def enum_dict_factory(inputs):
    inputs = [(i[0], i[-1].value) if isinstance(i[-1], IntEnum) else i
              for i in inputs]
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
    sender: str = 'user'
    formatted: Optional[Any] = None
    extra_info: Optional[Any] = None
    type: Optional[str] = None
    receiver: Optional[str] = None
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END
