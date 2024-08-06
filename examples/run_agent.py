from lagent.agents.stream import AgentForInternLM, MathCoder
from lagent.llms import INTERNLM2_META, LMDeployPipeline
from lagent.prompts.protocols.tool_protocol import get_plugin_prompt

model = LMDeployPipeline(
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
coder = MathCoder(llm=model)
query = (
    'Marie is thinking of a multiple of 63, while Jay is thinking of a factor '
    'of 63. They happen to be thinking of the same number. There are two '
    'possibilities for the number that each of them is thinking of, one '
    'positive and one negative. Find the product of these two numbers.')
res = coder(query, session_id=0)
print(res.model_dump_json())
print('-' * 120)
print(coder.get_steps(0))

# ----------------------- plugin -----------------------
print('-' * 80, 'plugin', '-' * 80)
plugins = [dict(type='ArxivSearch')]
agent = AgentForInternLM(
    llm=model,
    plugins=plugins,
    aggregator=dict(
        type='InternLMToolAggregator',
        plugin_prompt=get_plugin_prompt(plugins)))

query = 'LLM智能体方向的最新论文有哪些？'
res = agent(query, session_id=0)
print(res.model_dump_json())
print('-' * 120)
print(agent.get_steps(0))
