<div align="center">
  <img src="docs/imgs/lagent_logo.png" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lagent.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/lagent)](https://pypi.org/project/lagent)
[![license](https://img.shields.io/github/license/InternLM/lagent.svg)](https://github.com/InternLM/lagent/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)

[English](README.md) | [简体中文](README_zh-CN.md) | 日本語 | [हिंदी](README_in_HIN.md)

</div>

<p align="center">
    👋 <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> そして <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a> に参加する
</p>

## はじめに

Lagent は、大規模言語モデル(LLM)ベースのエージェントを効率的に構築できる軽量なオープンソースフレームワークです。また、LLM を拡張するための典型的なツールも提供します。我々のフレームワークの概要を以下に示します:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### 主な特徴

- **複数のエージェントをすぐにサポート** Lagent は現在、[ReAct](https://arxiv.org/abs/2210.03629)、[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)、[ReWOO](https://arxiv.org/abs/2305.18323) をサポートしており、推論や関数呼び出しの複数の試行に対して大規模言語モデル(LLM)を駆動することができる。

- **非常にシンプルで、拡張も簡単。** フレームワークは非常にシンプルで、明確な構造を持っています。わずか 20 行のコードで、独自のエージェントを構築することができます。また、3 つの代表的なツールをサポートしています： Python インタプリタ、API コール、google 検索です。

- **様々な大規模言語モデルをサポート。** API ベース(GPT-3.5/4)やオープンソース(LLaMA 2, InternLM)を含む様々な LLM をサポートしています。

## はじめに

Lagent の概要については[概要](docs/ja/get_started/overview.md)をご覧ください。また、クイックスタートのために非常にシンプルなコードを用意しています。詳細は [examples](examples/) を参照してください。

### インストール

pip でインストールする（推奨）。

```bash
pip install lagent
```

オプションとして、コードを修正したい場合に備えて、Lagent をソースからビルドすることもできる:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### ReAct ウェブデモの実行

```bash
# 最初に streamlit をインストールする必要があります
# pip install streamlit
streamlit run examples/react_web_demo.py
```

その後、以下のような UI からチャットができます
![image](https://github.com/InternLM/lagent/assets/24622904/3aebb8b4-07d1-42a2-9da3-46080c556f68)

### GPT-3.5 で ReWOO エージェントを動かす

以下は、GPT-3.5 で ReWOO を実行する例です

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

### InternLM で ReAct エージェントを動かす

注: Hugging Face モデルを実行したい場合は、まず `pip install -e .[all]` を実行してください。

```python
from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.llms import HFTransformer

llm = HFTransformer('internlm/internlm-chat-7b-v1_1')
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

## ライセンス

このプロジェクトは [Apache 2.0 license](LICENSE) の下でリリースされています。
