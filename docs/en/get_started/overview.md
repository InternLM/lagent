# Overview

This chapter introduces you to the framework of Lagent and provides links to detailed tutorials about Lagent.

## What is Lagent

Lagent is an open source LLM agent framework, which enables people to efficiently turn a large language model into an agent. It also provides some tools to enhance the ability of LLMs. The whole framework is shown below:

![image](https://github.com/InternLM/lagent/assets/24351120/e104171e-4baf-43b3-8e6d-90cff1b298b6)

Lagent consists of 3 main parts, agents, llms, and actions.

- **agents** provides agent implementation, such as ReAct, AutoGPT.
- **llms** supports various large language models, including open-sourced models (Llama-2, InternLM) through HuggingFace models or closed-source models like GPT3.5/4.
- **actions** contains a series of actions, as well as an action executor to manage all actions.

## How to Use

Here is a detailed step-by-step guide to learn more about Lagent:

1. For installation instructions, please see [README](https://github.com/InternLM/lagent/blob/main/README.md).

2. We provide several examples to build agents with Lagent in [examples](https://github.com/InternLM/lagent/tree/main/examples) by simply running `python examples/react_example.py`.
