import asyncio

from lagent.distributed import AsyncHTTPAgentClient, AsyncHTTPAgentServer, HTTPAgentClient, HTTPAgentServer
from lagent.llms import INTERNLM2_META
from lagent.schema import AgentMessage

server = HTTPAgentServer(
    '1',
    {
        'type': 'lagent.agents.AsyncAgent',
        'llm': {
            'type': 'AsyncLMDeployPipeline',
            'model_name': 'internlm2',
            'path': '/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2',
            'meta_template': INTERNLM2_META,
        }
    },
    port=8090,
)
print(server.is_alive)
message = AgentMessage(sender='user', content='hello')
result = server(message)
print(result)
server.shutdown()

# math coder
server = AsyncHTTPAgentServer(
    '1',
    {
        'type': 'lagent.agents.AsyncMathCoder',
        'llm': {
            'type': 'AsyncLMDeployPipeline',
            'model_name': 'internlm2',
            'path': '/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2',
            'meta_template': INTERNLM2_META,
            'tp': 1,
            'top_k': 1,
            'temperature': 1.0,
            'stop_words': ['<|im_end|>', '<|action_end|>'],
            'max_new_tokens': 1024,
        },
        'interpreter': {
            'type': 'AsyncIPythonInterpreter',
            'max_kernels': 100
        },
    },
    port=8091,
)
message = AgentMessage(
    sender='user',
    content=
    ('Marie is thinking of a multiple of 63, while Jay is thinking of a factor '
     'of 63. They happen to be thinking of the same number. There are two '
     'possibilities for the number that each of them is thinking of, one '
     'positive and one negative. Find the product of these two numbers.'))
result = server(message)
print(asyncio.run(result))
print(server.state_dict())

client = AsyncHTTPAgentClient(port=8091)
result = client('hello', session_id=1)
print(asyncio.run(result))
print(client.state_dict(1))

client = HTTPAgentClient(port=8091)
print(client.state_dict(1))
print(client('introduce yourself', session_id=1))
print(client.state_dict(1))
server.shutdown()
