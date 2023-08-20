# LAgent: Large Language Model as Agent

English | [简体中文](README_zh-CN.md)

## Introduction

LAgent is an open source LLM agent framework, which enables people to efficiently turn a large language model to agent. It also provides some typical tools to enlighten the ablility of LLM. The overview of our framework is shown below:

![image](https://github.com/InternLM/lagent/assets/24351120/e104171e-4baf-43b3-8e6d-90cff1b298b6)

### Major Features

- **Support multiple agent frameworks out of box.** We implement ReAct, AutoGPT and ReWOO, which enables the agents to drive LLMs for multiple trails of reasoning and tool utilization.

- **Extremely simple and easy to extend.** The framework is quite simple with clear project structure. With only 20 lines of code, you are able to construct your own agent. It also supports three typical tools: Python interpreter, API call, and google search.

- **Support various large language models.** We support different LLMs, including API-based (GPT3.5/4) and HuggingFace-based (LLaMa2, InternLM) models.

## Installation

Please refer to [Installation](docs/get_started.md) for installation instructions.

## Getting Started

Please see [Overview](docs/overview.md) for the general introduction of LAgent. Meanwhile, we provide an extremely simple code for quick start. You may refer to [examples](examples/) for more details.

```python
from lagent.agents import ReAct
from lagent.llms import HFTransformer
from lagent.tools import SerperSearch, PythonInterpreter

llm = HFTransformer('internlm/internlm-7b-chat')
search_tool = SerperSearch()
python_interpreter = PythonInterpreter()

chatbot = ReAct(llm=llm, tools=[search_tool, get_weather_tool])

response = chatbot.chat('若$z=-1+\sqrt{3}i$,则$\frac{z}{{z\overline{z}-1}}=\left(\ \ \right)$ (A) $-1+\sqrt{3}i$ (B) $-1-\sqrt{3}i$ (C) $-\frac{1}{3}+\frac{{\sqrt{3}}}{3}i$ (D) $-\frac{1}{3}-\frac{{\sqrt{3}}}{3}i$')
print(response['response'])
>>> 根据已有的信息，可以求得$z=-1+\\sqrt{3}i$，然后代入计算，得到结果为$-\\frac{1}{3}+\\frac{{\\sqrt{3}}}{3}i$。因此，答案是（C）。
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
