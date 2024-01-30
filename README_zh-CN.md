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

<p align="center">
    <iframe src="https://upos-hz-mirrorakam.akamaized.net/upgcxcode/99/71/1412447199/1412447199-1-16.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1706626499&gen=playurlv2&os=akam&oi=804486655&trid=b0750df67f8a4dfdb7021782a73a2b3eh&mid=0&platform=html5&upsig=7cbe56bea911db3153660c6a94eaa187&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform&hdnts=exp=1706626499~hmac=965cf78a445fa19afb6ba490c602c155b5a0baae9ec1ff609cb91023ceca9de3&bvc=vod&nettype=0&f=h_0_0&bw=39605&logo=80000000" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" height=360 width=640></iframe>
</p>


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
