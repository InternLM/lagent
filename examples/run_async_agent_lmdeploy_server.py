import asyncio
import json
import time

from datasets import load_dataset

from lagent.agents.stream import AsyncAgentForInternLM, AsyncMathCoder, get_plugin_prompt
from lagent.llms import INTERNLM2_META
from lagent.llms.lmdeploy_wrapper import AsyncLMDeployClient, AsyncLMDeployServer

# set up the loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# initialize the model
model = AsyncLMDeployServer(
    path='internlm/internlm2_5-7b-chat',
    meta_template=INTERNLM2_META,
    model_name='internlm-chat',
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>', '<|action_end|>'],
    max_new_tokens=1024,
)

# ----------------------- interpreter -----------------------
print('-' * 80, 'interpreter', '-' * 80)

ds = load_dataset('lighteval/MATH', split='test')
problems = [item['problem'] for item in ds.select(range(50))]


# coder = AsyncMathCoder(
#     llm=model,
#     interpreter=dict(type='AsyncIPythonInterpreter', max_kernels=250))
# tic = time.time()
# coros = [coder(query, session_id=i) for i, query in enumerate(problems)]
# res = loop.run_until_complete(asyncio.gather(*coros))
# # print([r.model_dump_json() for r in res])
# print('-' * 120)
# print(f'time elapsed: {time.time() - tic}')
# with open('./tmp_4.json', 'w') as f:
#     json.dump([coder.get_steps(i) for i in range(len(res))],
#               f,
#               ensure_ascii=False,
#               indent=4)

# ----------------------- streaming chat -----------------------
async def streaming(llm, problem):
    async for out in llm.stream_chat([{'role': 'user', 'content': problem}]):
        print(out)


tic = time.time()
client = AsyncLMDeployClient(
    url='http://127.0.0.1:23333',
    meta_template=INTERNLM2_META,
    model_name='internlm2_5-7b-chat',
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>', '<|action_end|>'],
    max_new_tokens=1024,
)
# loop.run_until_complete(streaming(model, problems[0]))
loop.run_until_complete(streaming(client, problems[0]))
print(time.time() - tic)

# ----------------------- plugin -----------------------
# print('-' * 80, 'plugin', '-' * 80)
# plugins = [dict(type='AsyncArxivSearch')]
# agent = AsyncAgentForInternLM(
#     llm=model,
#     plugins=plugins,
#     aggregator=dict(
#         type='InternLMToolAggregator',
#         plugin_prompt=get_plugin_prompt(plugins)))

# tic = time.time()
# coros = [
#     agent(query, session_id=i)
#     for i, query in enumerate(['LLM智能体方向的最新论文有哪些？'] * 50)
# ]
# res = loop.run_until_complete(asyncio.gather(*coros))
# # print([r.model_dump_json() for r in res])
# print('-' * 120)
# print(f'time elapsed: {time.time() - tic}')
