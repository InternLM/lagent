<div id="top"></div>

<div align="center">

# 思·索 MindSearch<br>Towards Deeper and Wider Answer Engine with LLM Agents
| [Research Preview](https://mindsearch.netlify.app/) | [Paper]() |
</div>

<div align="center">

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lagent.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/lagent)](https://pypi.org/project/lagent)
[![license](https://img.shields.io/github/license/InternLM/lagent.svg)](https://github.com/InternLM/lagent/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
![Visitors](https://api.visitorbadge.io/api/visitors?path=InternLM%2Flagent%20&countColor=%23263759&style=flat)
![GitHub forks](https://img.shields.io/github/forks/InternLM/lagent)
![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/lagent)

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">𝕏 (Twitter)</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

## ✨ Introduction

#### Demo Video here

MindSearch is an open-source AI Search Engine Framework with Perplexity.ai Pro performance. You can simply deploy it with your own perplexity.ai style search engine with either close-source LLMs (GPT, Claude) or open-source LLMs (InternLM2.5-7b-chat). It owns following features:
- 🤔 **Ask everything you want to know**: MindSearch is designed to solve any question in your life and use web knowledge.
- 📚 **In-depth Knowledge Discovery**: MindSearch browses hundreds of web pages to answer your question, providing deeper and wider knowledge base answer.
- 🔍 **Detailed Solution Path**: MindSearch exposes all details, allowing users to check everything they want. This greatly improves the credibility of its final response as well as usability.
- 💻 **Optimized UI Experimence**: Providing all kinds of interfaces for users, including React, Streamlit, Terminal. Choose any type based on your need.

## 👀 How MindSearch Works

<img src="docs/imgs/mindsearch_framework.png">

MindSearch consists of a Web Planner and Web Searcher. WebPlanner models the complex problem-solving minds as a dynamic graph construction process: it decomposes the question into sub-queries as graph nodes and progressively extends the graph based on the search result from WebSearcher. Tasked with each sub-query, WebSearcher performs hierarchical information retrieval with search engines and collects valuable information for WebPlanner.
The multi-agent design of MindSearch dispatches a load of processing massive information to different agents, enabling the whole framework to process a much longer context.

## ⚽️ Getting Started

### Lagent Installation

MindSearch backend with Lagent, please see the [overview](docs/en/get_started/overview.md) for the general introduction of Lagent. Meanwhile, we provide extremely simple code for quick start. You may refer to [examples](examples/) for more details.

### MindSearch

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{lagent2023,
    title={{Lagent: InternLM} a lightweight open-source framework that allows users to efficiently build large language model(LLM)-based agents},
    author={Lagent Developer Team},
    howpublished = {\url{https://github.com/InternLM/lagent}},
    year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).