<div id="top"></div>
<div align="center">
  <img src="docs/imgs/lagent_logo.png" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lagent.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/lagent)](https://pypi.org/project/lagent)
[![license](https://img.shields.io/github/license/InternLM/lagent.svg)](https://github.com/InternLM/lagent/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)

English | [简体中文](README_zh-CN.md) | [日本語](README_ja_JP.md) | [हिंदी](README_in_HIN.md) | [বাংলা](README_in_beng.md) | [한국어](README_KR_Kr.md)

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

<div align="center">

https://github.com/InternLM/lagent/assets/24622904/cb851b31-6932-422e-a776-b1aa68f2a64f

</div>

[English](README.md) | 简体中文

## 教程

请阅读[概述](docs/en/get_started/overview.md)对 Lagent 项目进行初步的了解。同时, 我们提供了两个非常简单的样例帮助你快速入门。 你也可以阅读[示例代码](examples/)获得更多的例子参考。

### 安装

通过 pip 进行安装 (推荐)。

```bash
pip install lagent
```

### 运行一个智能体的网页样例

你可能需要先安装 Streamlit 包

```bash
# pip install streamlit
streamlit run examples/internlm2_agent_web_demo.py
```

## 简介

Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

## 特性

- 流式输出：提供 `stream_chat` 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括
  - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
  - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
  - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。

## 引用

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 Lagent：

```latex
@misc{lagent2023,
    title={{Lagent: InternLM} a lightweight open-source framework that allows users to efficiently build large language model(LLM)-based agents},
    author={Lagent Developer Team},
    howpublished = {\url{https://github.com/InternLM/lagent}},
    year={2023}
}
```

## 开源许可证

该项目采用[Apache 2.0 开源许可证](LICENSE)。

<p align="right"><a href="#top">🔼 Back to top</a></p>
