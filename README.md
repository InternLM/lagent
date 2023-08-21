# Lagent: A lightweight framework for building LLM-based agents

English | [简体中文](README_zh-CN.md)

## Introduction

Lagent is a lightweight open-source framework that allows users to efficiently build large language model(LLM)-based agents. It also provides some typical tools to augment LLM. The overview of our framework is shown below:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### Major Features

- **Support multiple kinds of agents out of box.** Lagent now supports [ReAct](https://arxiv.org/abs/2210.03629), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) and [ReWOO](https://arxiv.org/abs/2305.18323), which can drive the large language models(LLMs) for multiple trials of reasoning and function calling.

- **Extremely simple and easy to extend.** The framework is quite simple with a clear structure. With only 20 lines of code, you are able to construct your own agent. It also supports three typical tools: Python interpreter, API call, and google search.

- **Support various large language models.** We support different LLMs, including API-based (GPT-3.5/4) and open-source (LLaMA 2, InternLM) models.

## Getting Started

Please see the [overview](docs/en/get_started/overview.md) for the general introduction of Lagent. Meanwhile, we provide extremely simple code for quick start. You may refer to [examples](examples/) for more details.

### Installation

Install with pip (Recommended).

```bash
pip install lagent
```

Optionally, you could also build Lagent from source in case you want to modify the code:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### Run a ReWOO agent with GPT-3.5

Below is an example for running ReWOO with GPT-3.5

```python
from lagent.agents import ReWOO
from lagent.actions import ActionExecutor, GoogleSearch, LLMQA
from lagent.llms import GPTAPI

llm = GPTAPI(model_type='gpt-3.5-turbo', key=['Your OPENAI_API_KEY'])
search_tool = GoogleSearch(api_key='Your SERPER_API_KEY')
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
search_tool = GoogleSearch(api_key='Your SERPER_API_KEY')
python_interpreter = PythonInterpreter()

chatbot = ReAct(
    llm=llm,
    action_executor=ActionExecutor(
        actions=[search_tool, python_interpreter]),
)

response = chatbot.chat('若$z=-1+\sqrt{3}i$,则$\frac{z}{{z\overline{z}-1}}=\left(\ \ \right)$')
print(response.response)
>>> $-\\frac{1}{3}+\\frac{{\\sqrt{3}}}{3}i$
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
