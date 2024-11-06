import asyncio
import json
import time

from datasets import load_dataset

from lagent.agents import AsyncMathCoder
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.llms import AsyncGPTAPI
from lagent.prompts.parsers import ToolParser

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

interpreter_prompt = (
    'Below is a math problem. Please solve it step by step with the assistance of Python programming. Consider using Sympy or Numpy library '
    'to facilitate your derivation, calculation and equation solving. Utilize the "pi" symbol and "Rational" from Sympy '
    'for $$\pi$$ and fractions, and simplify all fractions and square roots without converting them to decimal values. '
    'Please encapsulate each generated Jupyter Python code block with tags "{begin}" and "{end}". Conclude the '
    r'final answer when observations are sufficient and encapsulate the numerical result with LaTeX syntax "\boxed{{}}" '
    'without any unit, and end your conclusion with the special token "[END]" to denote the completion of your response. '
    'Keep the following points in mind:\n'
    '- You must alternately use human and programming languages in the chain of thought;\n'
    '- The number of your reasoning steps should not exceed **three**, which means you may merge some intermediate steps when the original answer is tedious.'
)

async_llm = dict(
    type=AsyncGPTAPI,
    model='gpt-4o-2024-05-13',
    retry=50,
    key='',
    max_new_tokens=2048,
    stop_words=['</python'],
    proxies=dict(),
)
async_agent = AsyncMathCoder(
    llm=async_llm,
    output_format=ToolParser(
        tool_type='interpreter',
        template=interpreter_prompt,
        begin='<python>\n',
        end='\n</python>'),
    aggregator=InternLMToolAggregator(
        environment_role='system',
        environment_begin='<output>\n',
        environment_end='\n</output>'),
    finish_condition=lambda m: '[END]' in m.content,
)

ds = load_dataset('lighteval/MATH', split='train')
problems = [item['problem'] for item in ds.select(range(30))]

tic = time.time()
coros = [async_agent(q, session_id=i) for i, q in enumerate(problems)]
res = loop.run_until_complete(asyncio.gather(*coros))
print(time.time() - tic)
with open('tmp_6.json', 'w') as f:
    json.dump([async_agent.get_steps(i) for i in range(len(problems))],
              f,
              ensure_ascii=False,
              indent=4)
