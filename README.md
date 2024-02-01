<div id="top"></div>
<div align="center">
  <img src="docs/imgs/lagent_logo.png" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lagent.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/lagent)](https://pypi.org/project/lagent)
[![license](https://img.shields.io/github/license/InternLM/lagent.svg)](https://github.com/InternLM/lagent/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
![Visitors](https://api.visitorbadge.io/api/visitors?path=InternLM%2Flagent%20&countColor=%23263759&style=flat)
![GitHub forks](https://img.shields.io/github/forks/InternLM/lagent)
![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/lagent)
![GitHub contributors](https://img.shields.io/github/contributors/InternLM/lagent)

English | [简体中文](README_zh-CN.md) | [日本語](README_ja_JP.md) | [हिंदी](README_in_HIN.md) | [বাংলা](README_in_beng.md) | [한국어](README_KR_Kr.md)

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">𝕏 (Twitter)</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

<div align="center">

https://github.com/InternLM/lagent/assets/24622904/3242f9bf-32d2-4907-8815-e16a75a4ac0e

</div>

## Getting Started

Please see the [overview](docs/en/get_started/overview.md) for the general introduction of Lagent. Meanwhile, we provide extremely simple code for quick start. You may refer to [examples](examples/) for more details.

### Installation

Install with pip (Recommended).

```bash
pip install lagent
```

### Run a Web Demo

You need to install Streamlit first.

```bash
# pip install streamlit
streamlit run examples/internlm2_agent_web_demo.py
```

## What's Lagent?

Lagent is a lightweight open-source framework that allows users to efficiently build large language model(LLM)-based agents. It also provides some typical tools to augment LLM. The overview of our framework is shown below:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

## Major Features

- Stream Output: Provides the `stream_chat` interface for streaming output, allowing cool streaming demos right at your local setup.
- Interfacing is unified, with a comprehensive design upgrade for enhanced extensibility, including:
  - Model: Whether it's the OpenAI API, Transformers, or LMDeploy inference acceleration framework, you can seamlessly switch between models.
  - Action: Simple inheritance and decoration allow you to create your own personal toolkit, adaptable to both InternLM and GPT.
  - Agent: Consistent with the Model's input interface, the transformation from model to intelligent agent only takes one step, facilitating the exploration and implementation of various agents.
- Documentation has been thoroughly upgraded with full API documentation coverage.

## 💻Tech Stack

<p>
  <a href="">
    <img src="https://img.shields.io/badge/Python-007ACC?style=for-the-badge&logo=python&logoColor=yellow" alt="python" />
  </a>

### All Thanks To Our Contributors:

<a href="https://github.com/InternLM/lagent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=InternLM/lagent" />
</a>

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

<p align="right"><a href="#top">🔼 Back to top</a></p>
