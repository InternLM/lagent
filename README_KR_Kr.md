<div id="top"></div>
<div align="center">
  <img src="docs/imgs/lagent_logo.png" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lagent.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/lagent)](https://pypi.org/project/lagent)
[![license](https://img.shields.io/github/license/InternLM/lagent.svg)](https://github.com/InternLM/lagent/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lagent)](https://github.com/InternLM/lagent/issues)

English | [简体中文](README_zh-CN.md) | [日本語](README_ja_JP.md) | [हिंदी](README_in_HIN.md) | [বাংলা](README_in_beng.md)

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

## 소개

Lagent는 사용자가 효율적으로 대규모 언어 모델(LLM) 기반 에이전트를 구축할 수 있게 해주는 경량의 오픈 소스 프레임워크입니다. 또한 LLM을 보강하기 위한 몇 가지 일반적인 도구도 제공합니다. 우리 프레임워크의 개요는 아래와 같이 나와 있습니다:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### 주요 기능

**0.1.2** 은 2023년 10월 24일에 릴리스되었습니다:

- **효율적인 추론 엔진 지원.** Lagent는 이제 효율적인 추론 엔진 [lmdeploy turbomind](https://github.com/InternLM/lmdeploy/tree/main) 을 지원합니다.

- **다양한 종류의 에이전트를 기본으로 지원.** Lagent는 이제 [ReAct](https://arxiv.org/abs/2210.03629), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) and [ReWOO](https://arxiv.org/abs/2305.18323), 을 지원합니다. 이는 대규모 언어 모델(LLMs)을 이용하여 추론과 기능 호출의 여러 시행을 가능하게 합니다.

- **매우 간단하고 확장하기 쉽습니다.** 이 프레임워크는 구조가 명확한 간단한 구조를 가지고 있습니다. 20줄의 코드로 자체 에이전트를 구축할 수 있습니다. 또한 Python 인터프리터, API 호출, 구글 검색과 같은 세 가지 일반적인 도구를 지원합니다.

- **다양한 대규모 언어 모델 지원.** 우리는 다른 LLMs를 지원하며, API 기반(GPT-3.5/4) 및 오픈 소스 (LLaMA 2, InternLM) 모델을 포함합니다.

## 시작하기

일반적인 Lagent 소개에 대한 [overview](docs/en/get_started/overview.md) 를 확인하십시오. 동시에 빠른 시작을 위한 매우 간단한 코드를 제공합니다. 자세한 내용은 [examples](examples/) 를 참조하십시오.

### 설치

pip를 사용하여 설치하십시오 (권장).

```bash
pip install lagent
```

원하는 경우 코드를 수정하려면 Lagent를 원본에서 빌드할 수도 있습니다:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### ReAct 웹 데모 실행

```bash
# 먼저 streamlit을 설치해야 합니다
# pip install streamlit
streamlit run examples/react_web_demo.py
```

그런 다음 아래와 같이 표시된 UI를 통해 채팅할 수 있습니다.
![image](https://github.com/InternLM/lagent/assets/24622904/3aebb8b4-07d1-42a2-9da3-46080c556f68)

### GPT-3.5로 ReWOO 에이전트 실행

아래는 GPT-3.5로 ReWOO를 실행하는 예입니다.

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

### InternLM과 함께 ReAct 에이전트 실행

참고: HuggingFace 모델을 실행하려면 먼저 pip install -e .[all]을 실행하십시오.

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

## 인용

이 프로젝트가 귀하의 연구에 유용하다고 생각하면 다음과 같이 인용해 주십시오:

```latex
@misc{lagent2023,
    title={{Lagent: InternLM} a lightweight open-source framework that allows users to efficiently build large language model(LLM)-based agents},
    author={Lagent Developer Team},
    howpublished = {\url{https://github.com/InternLM/lagent}},
    year={2023}
}
```

## 라이선스

이 프로젝트는 [Apache 2.0](LICENSE) 하에 공개되었습니다.
<p align="right"><a href="#top">🔼 Back to top</a></p>
