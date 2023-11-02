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
    👋 <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> और <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a> पर हमसे जुड़ें
</p>

## परिचय

Lagent एक हल्का ओपन-सोर्स फ्रेमवर्क है जो उपयोगकर्ताओं को बड़े भाषा मॉडल (एलएलएम)-आधारित एजेंटों को कुशलतापूर्वक बनाने की अनुमति देता है। यह एलएलएम को बढ़ाने के लिए कुछ विशिष्ट उपकरण भी प्रदान करता है। हमारे ढांचे का अवलोकन नीचे दिखाया गया है:

![image](https://github.com/InternLM/lagent/assets/24351120/cefc4145-2ad8-4f80-b88b-97c05d1b9d3e)

### प्रमुख विशेषताएं

- **बॉक्स से बाहर कई प्रकार के एजेंटों का समर्थन करें।** लैजेंट अब समर्थन करता है [ReAct](https://arxiv.org/abs/2210.03629), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) और [ReWOO](https://arxiv.org/abs/2305.18323), जो तर्क और फ़ंक्शन कॉलिंग के कई परीक्षणों के लिए बड़े भाषा मॉडल (एलएलएम) को संचालित कर सकता है।

- **बेहद सरल और विस्तार करने में आसान।** स्पष्ट संरचना के साथ ढांचा काफी सरल है। कोड की केवल 20 पंक्तियों के साथ, आप अपना स्वयं का एजेंट बनाने में सक्षम हैं। यह तीन विशिष्ट टूल का भी समर्थन करता है: पायथन इंटरप्रेटर, एपीआई कॉल और गूगल सर्च।

- **विभिन्न बड़े भाषा मॉडल का समर्थन करें।** हम एपीआई-आधारित (जीपीटी-3.5/4) और ओपन-सोर्स (एलएलएएमए 2, इंटर्नएलएम) मॉडल सहित विभिन्न एलएलएम का समर्थन करते हैं।

## शुरू करना

लैजेंट के सामान्य परिचय के लिए कृपया [अवलोकन](docs/in/get_started/overview.md) देखें। इस बीच, हम त्वरित शुरुआत के लिए अत्यंत सरल कोड प्रदान करते हैं। अधिक जानकारी के लिए आप [उदाहरण](examples/) अधिक जानकारी के लिए।

### इंस्टालेशन

pip के साथ स्थापित करें (अनुशंसित)।

```bash
pip install lagent
```

वैकल्पिक रूप से, यदि आप कोड को संशोधित करना चाहते हैं तो आप स्रोत से लैजेंट भी बना सकते हैं:

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

### रिएक्ट वेब डेमो चलाएँ

```bash
# You need to install streamlit first
# pip install streamlit
streamlit run examples/react_web_demo.py
```

फिर आप नीचे दिखाए गए यूआई के माध्यम से चैट कर सकते हैं
![image](https://github.com/InternLM/lagent/assets/24622904/3aebb8b4-07d1-42a2-9da3-46080c556f68)

### GPT-3.5 के साथ ReWOO एजेंट चलाएँ

GPT-3.5 के साथ ReWOO चलाने का एक उदाहरण नीचे दिया गया है

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

### InternLM के साथ एक ReAct एजेंट चलाएँ

नोट: यदि आप हगिंगफेस मॉडल चलाना चाहते हैं, तो कृपया पहले `pip install -e .[all]` चलाएं।

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

## लाइसेंस

यह प्रोजेक्ट [Apache 2.0 license](LICENSE) के तहत जारी किया गया है।
<p align="right"><a href="#top">🔼 Back to top</a></p>
