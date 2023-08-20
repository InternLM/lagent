# LAgent: Large Language model as Agent

[English](README.md) | 简体中文

## 简介

Lagent是一个开源的LLM代理框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:

![image](https://github.com/InternLM/lagent/assets/24351120/e104171e-4baf-43b3-8e6d-90cff1b298b6)

### 主要特点

- **实现了多种类型的智能体，** 我们支持了经典的 ReAct，AutoGPT 和 ReWoo 等智能体，这些智能体能够调用大语言模型进行多轮的推理和工具调用。

- **框架简单易拓展.** 框架的代码结构清晰且简单，只需要不到20行代码你就能够创造出一个你自己的agent。同时我们支持了Python解释器、API 调用和搜索三类常用典型工具。

- **灵活支持多个大语言模型.** 我们提供了多种大语言模型支持，包括 InternLM、Llama-2 等开源模型和 GPT-4/3.5 等基于 API 的闭源模型。

## 安装

请参考[快速入门文档](docs/get_started.md)进行安装。

## 教程

请阅读[概述](docs/overview.md)对Lagent进行初步的了解。同时, 我们提供了一个非常简单的code帮助你快速入门。 你也可以阅读[examples](examples/)获得更多的例子参考。

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

## 开源许可证

该项目采用[Apache 2.0 开源许可证](LICENSE)。
