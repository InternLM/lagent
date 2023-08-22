# Lagent: A lightweight framework for building LLM-based agents

[English](README.md) | 简体中文

## 简介

Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### 主要特点

- **实现了多种类型的智能体，** 我们支持了经典的 [ReAct](https://arxiv.org/abs/2210.03629)，[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) 和 [ReWoo](https://arxiv.org/abs/2305.18323) 等智能体，这些智能体能够调用大语言模型进行多轮的推理和工具调用。
- **框架简单易拓展.** 框架的代码结构清晰且简单，只需要不到20行代码你就能够创造出一个你自己的智能体（agent）。同时我们支持了 Python 解释器、API 调用和搜索三类常用典型工具。
- **灵活支持多个大语言模型.** 我们提供了多种大语言模型支持，包括 InternLM、Llama-2 等开源模型和 GPT-4/3.5 等基于 API 的闭源模型。

## 教程

请阅读[概述](docs/en/get_started/overview.md)对 Lagent 项目进行初步的了解。同时, 我们提供了两个非常简单的样例帮助你快速入门。 你也可以阅读[示例代码](examples/)获得更多的例子参考。

### 安装

通过 pip 进行安装 (推荐)。

```bash
pip install lagent
```

同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 Lagent:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### 用 GPT-3.5 构建一个 ReWOO 智能体

```python
from lagent.agents import ReWOO
from lagent.actions import ActionExecutor, GoogleSearch, LLMQA
from lagent.llms import GPTAPI

llm = GPTAPI(model_type='gpt-3.5-turbo', key='OPENAI_API_KEY')
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

### 用 InternLM 构建一个 ReAct 智能体

注意：如果你想要启动一个HuggingFace的模型，请先运行 `pip install -e .[all]`。

```python
from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.llms import HFTransformer

llm = HFTransformer('internlm/internlm-chat-7b-v1_1')
search_tool = GoogleSearch(api_key='SERPER_API_KEY')
python_interpreter = PythonInterpreter()

chatbot = ReAct(
    llm=llm,
    action_executor=ActionExecutor(
        actions=[search_tool, python_interpreter]),
)

response = chatbot.chat('若$z=-1+\sqrt{3}i$,则$\frac{z}{{z\overline{z}-1}}=\left(\ \ \right)$')
print(response.response)
>>> $-\\frac{1}{3}+\\frac{{\\sqrt{3}}}{3}i
```

## 开源许可证

该项目采用[Apache 2.0 开源许可证](LICENSE)。
