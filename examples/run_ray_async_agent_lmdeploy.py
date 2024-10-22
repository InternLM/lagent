import asyncio
import json
import time

import ray
from datasets import load_dataset

from lagent.distributed.ray_serve import AsyncAgentRayActor
from lagent.llms import INTERNLM2_META
from lagent.llms.lmdeploy_wrapper import AsyncLMDeployPipeline

ray.init()

# set up the loop

# initialize the model
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
model = dict(
    type=AsyncLMDeployPipeline,
    path='internlm/internlm2_5-7b-chat',
    meta_template=INTERNLM2_META,
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>', '<|action_end|>'],
    max_new_tokens=1024,
)

# ----------------------- interpreter -----------------------
print('-' * 80, 'interpreter', '-' * 80)
ds = load_dataset('lighteval/MATH', split='test')
problems = [item['problem'] for item in ds.select(range(5000))]

coder = dict(
    type='lagent.agents.stream.AsyncMathCoder',
    llm=model,
    interpreter=dict(type='AsyncIPythonInterpreter', max_kernels=300),
)
tic = time.time()

actor1 = AsyncAgentRayActor(coder.copy(), num_gpus=1)
actor2 = AsyncAgentRayActor(coder.copy(), num_gpus=1)
corots = [
    actor1(query, session_id=i)
    for i, query in enumerate(problems[:len(problems) // 2])
]
corots += [
    actor2(query, session_id=i)
    for i, query in enumerate(problems[len(problems) // 2:])
]
results = loop.run_until_complete(asyncio.gather(*corots))

print('-' * 120)
print(f'time elapsed: {time.time() - tic}')
all_step = ray.get([
    actor1.agent_actor.get_steps.remote(i) for i in range(len(problems) // 2)
])
all_step += ray.get([
    actor2.agent_actor.get_steps.remote(i)
    for i in range(len(problems[len(problems) // 2:]))
])

with open('./tmp_1.json', 'w') as f:
    json.dump(all_step, f, ensure_ascii=False, indent=4)
