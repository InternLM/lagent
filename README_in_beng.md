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
    👋 <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> এবং <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a> সাথে আমাদের সাথে যোগদান করুন
</p>

## পরিচিতি

লেজেন্ট হল একটি হালকা ওপেন-সোর্স ফ্রেমওয়ার্ক, যা ব্যবহারকারীদের দ্বারা প্রশাসক ভাষা মডেল (LLM) ভিত্তিক এজেন্ট সৃজনশীলভাবে তৈরি করতে দেয়। এটি লেজেন্ট যেসব প্রধান সরঞ্জাম সরবরাহ করে, সেটি নীচে দেখানো হয়:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### মৌলিক বৈশিষ্ট্য

- **বাক্যের কিছু প্রকারের এজেন্টের সাথে সমর্থন।** লেজেন্ট এখন [ReAct](https://arxiv.org/abs/2210.03629), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) এবং [ReWOO](https://arxiv.org/abs/2305.18323) সহ বড় ভাষা মডেল (LLM)-কে যুক্ত করতে পারে, যা বড় ভাষা মডেল (LLM) ব্যবহার করে বিভিন্ন কারণের এবং কার্য করার জন্য অনেক পরীক্ষা করতে পারে।

- **অত্যন্ত সহজ এবং বড় করা সম্ভব।** এই ফ্রেমওয়ার্কটি খুব সাধারণ, একটি স্পষ্ট স্ট্রাকচার সহ। শুধুমাত্র 20 লাইনের কোডের সাথে, আপনি নিজের এজেন্ট তৈরি করতে পারেন। এটি সাথে একই সময়ে তিনটি সাধারণ টুলও সমর্থন করে: পাইথন ইন্টারপ্রিটার, API কল, এবং গুগল সার্চ।

- **বিভিন্ন বড় ভাষা মডেলের সমর্থন।** আমরা API-ভিত্তিক (GPT-3.5/4) এবং ওপেন-সোর্স (LLaMA 2, InternLM) মডেলগুলির মধ্যে বিভিন্ন LLM-এসকে সমর্থন করি।

## শুরু করা

লেজেন্টের সাধারণ পরিচিতির জন্য [অবলো](docs/en/get_started/overview.md) দেখ

## ইনস্টলেশন

পিপ দিয়ে ইনস্টল করুন (সুপারিশ).

```bash
pip install lagent
```

আপনি চাইলে সোর্স থেকে লেজেন্ট তৈরি করতে পারেন, কোড পরিবর্তন করতে চাইলে:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### ReAct ওয়েব ডেমো চালান

```bash
# You need to install streamlit first
# pip install streamlit
streamlit run examples/react_web_demo.py
```

তারপর আপনি নীচে দেওয়া ছবির মাধ্যমে ইউআই দিয়ে চ্যাট করতে পারেন
![image](https://github.com/InternLM/lagent/assets/24622904/3aebb8b4-07d1-42a2-9da3-46080c556f68)

### GPT-3.5 সহ একটি ReWOO এজেন্ট চালান

নীচে একটি উদাহরণ দেওয়া হল ReWOO সহ GPT-3.5 চালানোর জন্য

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

### InternLM দিয়ে একটি ReAct এজেন্ট চালান

নোট: আপনি যদি একটি HuggingFace মডেল চালাতে চান, তবে প্রথমে pip install -e .[all] চালানো দরকার।

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

## লাইসেন্স

এই প্রকল্পটি [Apache 2.0 license](LICENSE) অনুসরণ করে প্রকাশিত হয়।

<p align="right"><a href="#top">🔼 Back to top</a></p>
