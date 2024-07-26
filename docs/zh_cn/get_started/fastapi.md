# 项目名称

这是一个基于 FastAPI 构建的应用程序，用于初始化和运行智能体。该智能体可以执行各种操作，并通过自然语言处理模型（LLM）进行交互。

## 目录

- [项目简介](#项目简介)
- [安装](#安装)
- [使用方法](#使用方法)
- [API 端点](#api-端点)
- [配置说明](#配置说明)
  - [LLM 配置](#llm-配置)
  - [Protocol 配置](#protocol-配置)
  - [Agent 配置](#agent-配置)
- [示例](#示例)

## 项目简介

该项目使用 FastAPI 框架构建了一个 Web 应用程序，能够初始化和运行智能体。智能体通过配置文件进行初始化，并使用自然语言处理模型（LLM）来处理输入和生成响应。

## 安装

1. 安装依赖：

   ```bash
   pip install sse-starlette janus pyvis fastapi uvicorn termcolor
   ```

2. 启动应用程序：

   ```bash
   uvicorn main:app --reload
   ```

## 使用方法

启动应用程序后，可以通过浏览器访问 `http://127.0.0.1:8000/docs` 查看自动生成的 API 文档。

## API 端点

### `/solve`

- **方法**: `POST`

- **描述**: 初始化并运行智能体。

- **请求体**:

  - `inputs` (str 或 List\[Dict\]): 输入数据，如`你好`或`[{"role": "user", "content": "你好"}]`。
  - `agent_cfg` (Dict): 智能体的配置文件。

- **响应**: 智能体的返回是一个 `AgentReturn` 数据类。

  ```python
  @dataclass
  class AgentReturn:
      type: str  # planner, searcher
      content: str  # searcher正在处理的问题
      state: Union[AgentStatusCode, int]
      response: str = ''  # thought or final answer
      nodes: Dict[str, 'AgentReturn']
      adjacency_list: Dict[str, List[str]]
      actions: List[ActionReturn] = field(default_factory=list)
      inner_steps: List = field(default_factory=list)

  class AgentStatusCode(Enum):
      END = 0  # end of streaming 返回本次history
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
      ANSWER_ING = 9  # final answer is in streaming

  @dataclass
  class ActionReturn:
      args: Union[Dict, str]  # input parameters or code
      url: Optional[str] = None
      type: Optional[str] = None  # IPython, GoogleScholar, ArxivSearch
      result: Optional[dict] = None
      errmsg: Optional[str] = None
      state: Union[ActionStatusCode, int] = ActionStatusCode.SUCCESS
      thought: Optional[str] = None
      valid: Optional[ActionValidCode] = ActionValidCode.OPEN

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
  ```

## 配置说明

智能体的配置文件 `agent_cfg` 包含以下几个部分：

- `type`: 智能体的类型，如 `MindSearchAgent`。
- `llm`: 语言模型的配置。
- `searcher`: 搜索器的配置（可选）。
- `protocol`: 协议的配置。
- `max_turn`: 最大对话轮数，默认为 3。

### LLM 配置

LLM（大语言模型）配置用于指定使用的语言模型及其相关参数。以下是一些示例配置：

1. **LMDeploy**：

   ```python
   from lagent.llms import INTERNLM2_META
   llm_cfg = {
       "type": "LMDeployClient",
       "model_name": "internlm2-chat-7b",
       "url": "http://ip:port",
       "meta_template": INTERNLM2_META,
       "max_new_tokens": 4096,
       "top_p": 0.8,
       "top_k": 1,
       "temperature": 0.8,
       "repetition_penalty": 1.0
   }
   ```

   注意，你需要额外安装 `lmdeploy` 包。

2. **API**：

   对于 OpenAI API：

   ```python
   llm_cfg = {
       "type": "GPTAPI",
       "model_type": "gpt-4o",
       "key": "YOUR OPENAI API KEY"
   }
   ```

   对于 Puyu API：

   ```python
   llm_cfg = {
       "type": "GPTAPI",
       "model_type": "internlm2.5-latest",
       "openai_api_base": "https://puyu.openxlab.org.cn/puyu/api/v1/chat/completions",
       "key": "YOUR PUYU API KEY",
       "meta_template": [
           {"role": "system", "api_role": "system"},
           {"role": "user", "api_role": "user"},
           {"role": "assistant", "api_role": "assistant"},
           {"role": "environment", "api_role": "environment"}
       ],
       "retry": 1000,
       "top_p": 0.8,
       "top_k": 1,
       "temperature": 0.8,
       "max_new_tokens": 4096,
       "repetition_penalty": 1.02
   }
   ```

3. **HuggingFace**：

   ```python
   llm_cfg = {
       "type": "HFTransformerChat",
       "path": "model path",
       "meta_template": "META",
       "max_new_tokens": 1024,
       "top_p": 0.8,
       "top_k": None,
       "temperature": 0.1,
       "repetition_penalty": 1.0
   }
   ```

   注意，你需要额外安装 `transformers` 包。

### Protocol 配置

Protocol 配置用于定义代理与 LLM 交互的协议。以下是一个示例配置：

```python
from datetime import datetime
from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_CN
protocol_cfg = {
    "type": "MindSearchProtocol",
    "meta_prompt": datetime.now().strftime('The current date is %Y-%m-%d.'),
    "interpreter_prompt": GRAPH_PROMPT_CN,
    "response_prompt": "请根据上文内容对问题给出详细的回复"
}
```

### Agent 配置

Agent 配置用于定义智能代理的整体行为和参数。以下是一个示例配置：

```python
llm = init_module(llm_cfg, llm_factory)
searcher_cfg = {
    "llm": llm,
    "plugin": [
        {
            "type": "BingBrowser",
            "api_key": "YOUR BING API KEY"
        }
    ],
    "protocol": {
        "type": "MindSearchProtocol",
        "meta_prompt": datetime.now().strftime('The current date is %Y-%m-%d.'),
        "plugin_prompt": searcher_system_prompt_cn,
    },
    "template": searcher_input_template_cn
}
agent_cfg = {
    "type": "MindSearchAgent",
    "llm": llm_cfg,
    "protocol": protocol_cfg,
    "searcher_cfg": searcher_cfg,
    "max_turn": 10
}
agent = init_module(agent_cfg, agent_factory)
```

注意，具体代码实现请参考 `app.py`。

## 示例

```python
for response in agent.stream_chat(inputs):
    if isinstance(response, tuple):
        agent_return, node_name = response
    else:
        agent_return = response
        node_name = None
```

`agent_return` 是 `AgentReturn` 数据类。如果 `agent` 是 `MindSearchAgent` 的一个实例（表征为planner），`node_name` 指代planner分配的任务名称，其会实例化一个 `searcher`来处理该任务，`agent_return.nodes` 中每一项的 `detail` 也是 `AgentReturn` 数据类，其记录了searcher的处理过程。
