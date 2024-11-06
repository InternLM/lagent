import asyncio
import json
import time

from datasets import load_dataset

from lagent.agents.stream import AsyncAgentForInternLM, AsyncMathCoder
from lagent.llms import INTERNLM2_META
from lagent.llms.vllm_wrapper import AsyncVllmModel
from lagent.prompts.parsers import ToolParser

# set up the loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# initialize the model
model = AsyncVllmModel(
    path='Qwen/Qwen2-7B-Instruct',
    meta_template=INTERNLM2_META,
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>', '\n```\n'],
    max_new_tokens=1024,
)

# ----------------------- interpreter -----------------------
print('-' * 80, 'interpreter', '-' * 80)

ds = load_dataset('lighteval/MATH', split='test')
problems = [item['problem'] for item in ds.select(range(50))]

coder = AsyncMathCoder(
    llm=model,
    interpreter=dict(
        type='lagent.actions.AsyncIPythonInterpreter', max_kernels=200),
    output_format=ToolParser(
        'interpreter',
        template=
        ('Integrate step-by-step reasoning and Python code to solve math problems '
         'using the following guidelines:\n'
         '- Analyze the question and write jupyter code to solve the problem;\n'
         r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
         'units. \n'),
        begin='\n```python\n',
        end='\n```\n'))

tic = time.time()
coros = [coder(query, session_id=i) for i, query in enumerate(problems)]
res = loop.run_until_complete(asyncio.gather(*coros))
# print([r.model_dump_json() for r in res])
print('-' * 120)
print(f'time elapsed: {time.time() - tic}')

with open('./tmp_3.json', 'w') as f:
    json.dump([coder.get_steps(i) for i in range(len(res))],
              f,
              ensure_ascii=False,
              indent=4)
