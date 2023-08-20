from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from lagent.utils import is_module_exist


def enum_dict_factory(inputs):
    inputs = [(i[0], i[-1].value) if isinstance(i[-1], Enum) else i
              for i in inputs]
    return dict(inputs)


def dataclass2dict(data):
    return asdict(data, dict_factory=enum_dict_factory)


class ActionStatusCode(int, Enum):
    ING = 1
    SUCCESS = 0
    HTTP_ERROR = -1000  # http error
    ARGS_ERROR = -1001  # 参数错误
    API_ERROR = -1002  # 不知道的API错误


class ActionValidCode(int, Enum):
    FINISH = 1
    OPEN = 0
    CLOSED = -1
    INVALID = -2
    ABSENT = -3  # NO ACTION


@dataclass
class ActionReturn:
    args: Dict
    url: Optional[str] = None
    type: Optional[str] = None
    result: Optional[str] = None
    errmsg: Optional[str] = None
    state: Union[ActionStatusCode, int] = ActionStatusCode.SUCCESS
    thought: Optional[str] = None
    valid: Optional[ActionValidCode] = ActionValidCode.OPEN


class AgentStatusCode(Enum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    CMD = 2  # return command
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 3  # session is ready for inference


@dataclass
class AgentReturn:
    actions: List[ActionReturn] = field(default_factory=list)
    response: str = ''
    inner_steps: List = field(default_factory=list)
    errmsg: Optional[str] = None


if is_module_exist('lmdeploy'):
    from lmdeploy.serve.turbomind.chatbot import StatusCode
    STATE_MAP = {
        StatusCode.TRITON_STREAM_END: AgentStatusCode.END,
        StatusCode.TRITON_SERVER_ERR: AgentStatusCode.SERVER_ERR,
        StatusCode.TRITON_SESSION_CLOSED: AgentStatusCode.SESSION_CLOSED,
        StatusCode.TRITON_STREAM_ING: AgentStatusCode.STREAM_ING,
        StatusCode.TRITON_SESSION_OUT_OF_LIMIT:
        AgentStatusCode.SESSION_OUT_OF_LIMIT,
        StatusCode.TRITON_SESSION_INVALID_ARG:
        AgentStatusCode.SESSION_INVALID_ARG,
        StatusCode.TRITON_SESSION_READY: AgentStatusCode.SESSION_READY
    }
else:
    STATE_MAP = {}
