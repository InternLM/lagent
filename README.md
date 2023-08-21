# Lagent: Large Language Model as Agent

English | [简体中文](README_zh-CN.md)

## Introduction

Lagent is an open source LLM agent framework, which enables people to efficiently turn a large language model to agent. It also provides some typical tools to enlighten the ablility of LLM. The overview of our framework is shown below:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### Major Features

- **Support multiple agent frameworks out of box.** We implement ReAct, AutoGPT and ReWOO, which enables the agents to drive LLMs for multiple trails of reasoning and tool utilization.

- **Extremely simple and easy to extend.** The framework is quite simple with clear project structure. With only 20 lines of code, you are able to construct your own agent. It also supports three typical tools: Python interpreter, API call, and google search.

- **Support various large language models.** We support different LLMs, including API-based (GPT3.5/4) and HuggingFace-based (LLaMa2, InternLM) models.

## Getting Started

Please see [Overview](docs/en/get_started/overview.md) for the general introduction of Lagent. Meanwhile, we provide extremely simple code for quick start. You may refer to [examples](examples/) for more details.

### Installation

Install with pip (Recommended).

```
pip install lagent
```

Optionally, you could also build Lagent from source in case you want to modify the code:

```
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### Run a ReWOO agent with GPT3.5

```python
from lagent.agents import ReWOO
from lagent.actions import ActionExecutor, GoogleSearch, LLMQA
from lagent.llms import GPTAPI

llm = GPTAPI(model_type='gpt-3.5-turbo', key=['OPENAI_API_KEY'])
search_tool = GoogleSearch(api_key='SERPER_API_KEY')
llmqa_tool = LLMQA(llm)

chatbot = ReWOO(
    llm=llm,
    action_executor=ActionExecutor(
        actions=[search_tool, llmqa_tool]),
)

response = chatbot.chat('What profession does Nicholas Ray and Elia Kazan have in common')
print(response.response)
>>> Film director.
```

### Run a ReAct agent with InternLM

NOTE: If you want to run a HuggingFace model, please run `pip install -e .[all]` first.

```python
from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.llms import HFTransformer

llm = HFTransformer('internlm/internlm-7b-chat-v1.1')
search_tool = GoogleSearch()
python_interpreter = PythonInterpreter()

chatbot = ReAct(
    llm=llm,
    action_executor=ActionExecutor(
        actions=[search_tool, python_interpreter]),
)

response = chatbot.chat('若$z=-1+\sqrt{3}i$,则$\frac{z}{{z\overline{z}-1}}=\left(\ \ \right)$ (A) $-1+\sqrt{3}i$ (B) $-1-\sqrt{3}i$ (C) $-\frac{1}{3}+\frac{{\sqrt{3}}}{3}i$ (D) $-\frac{1}{3}-\frac{{\sqrt{3}}}{3}i$')
print(response.response)
>>> 根据已有的信息，可以求得$z=-1+\\sqrt{3}i$，然后代入计算，得到结果为$-\\frac{1}{3}+\\frac{{\\sqrt{3}}}{3}i$。因此，答案是（C）。
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
