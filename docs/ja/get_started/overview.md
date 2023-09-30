# 概要

この章では Lagent のフレームワークを紹介し、Lagent に関する詳細なチュートリアルへのリンクを提供します。

## Lagent とは

Lagent はオープンソースの LLM エージェントフレームワークで、大規模な言語モデルを効率的にエージェントに変換することができます。また、LLM の能力を啓発するためのいくつかの典型的なツールも提供します:

![image](https://github.com/InternLM/lagent/assets/24351120/e104171e-4baf-43b3-8e6d-90cff1b298b6)

Lagent はエージェント、LLMS、アクションの 3 つの主要部分から構成されています。

- **agents** ReAct、AutoGPT などのエージェント実装を提供する。
- **llms** は、オープンソース・モデル（Llama-2、InterLM）から Hugging Face モデル、あるいは GPT3.5/4 のようなクローズドソースモデルを含む、さまざまな大規模言語モデルをサポートしています。
- **actions** には一連のアクションと、すべてのアクションを管理するアクションエグゼキュータが含まれている。

## 使用方法

Lagent についての詳しいステップバイステップガイドはこちら:

1. インストール方法については、[README](../README.md)を参照してください。

2. python の `examples/react_example.py` を実行するだけで、Lagent でエージェントをビルドする例を [examples](examples/) にいくつか用意しています。
