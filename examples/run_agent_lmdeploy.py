import json
import time

from lagent.agents.aggregator import InternLMToolAggregator
from lagent.agents.stream import PLUGIN_CN, AgentForInternLM, MathCoder, get_plugin_prompt
from lagent.llms import INTERNLM2_META, LMDeployPipeline
from lagent.prompts.parsers import PluginParser

model = LMDeployPipeline(
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
coder = MathCoder(
    llm=model,
    aggregator=InternLMToolAggregator(few_shot=[[
        {
            'role':
            'user',
            'content':
            "Mike paid $\\$1.25$ for a stamp three years ago. He was just offered double that amount for the stamp. Assuming the stamp's offer price doubles every three years, how many dollars will he be offered in 12 more years?"
        },
        {
            'role':
            'language',
            'content':
            "Let's break down the problem into smaller steps to understand how to approach it.\n\n**Step 1:** Determine the initial amount paid for the stamp, which is $1.25.\n\n**Step 2:** Understand the pattern of the offer price increase. The offer price doubles every three years.\n\n**Step 3:** Calculate the offer price in 12 more years.\n\nSince the offer price doubles every three years, in 12 years, the price will double 12 / 3 = 4 times.\n\nWe can represent the offer price after 12 more years as:\n\\[ \\text{Offer Price} = \\text{Initial Price} \\times 2^{(\\text{Number of Doublings})} \\]\n\nLet's calculate this using Python to get the exact figure.\n\n"
        },
        {
            'role': 'tool',
            'content':
            '```python\nfrom sympy import symbols, Eq, solve\n\n# Define the variables\ninitial_price = 1.25\ndoublings_in_12_years = 12 / 3\n\n# Calculate the offer price after 12 years\noffer_price_12_years = initial_price * 2**doublings_in_12_years\n\noffer_price_12_years\n```',
            'name': 'interpreter',
        },
        {
            'role': 'environment',
            'content': '20.0',
            'name': 'interpreter'
        },
        {
            'role':
            'language',
            'content':
            'After calculating the given formula, the offer price for the stamp in 12 more years is found to be $\\boxed{20.0}$ dollars.'
        },
    ]]))
query = (
    'Marie is thinking of a multiple of 63, while Jay is thinking of a factor '
    'of 63. They happen to be thinking of the same number. There are two '
    'possibilities for the number that each of them is thinking of, one '
    'positive and one negative. Find the product of these two numbers.')
res = coder(query, session_id=0)
# print(res.model_dump_json())
# print('-' * 120)
print(coder.get_steps(0))
import json

print(
    json.dumps(
        coder.agent.aggregator.aggregate(coder.agent.memory, coder.agent.name,
                                         coder.agent.output_format),
        ensure_ascii=False,
        indent=4))

state_dict = {
    'memory': [{
        'sender': 'user',
        'content':
        'Marie is thinking of a multiple of 63, while Jay is thinking of a factor of 63. They happen to be thinking of the same number. There are two possibilities for the number that each of them is thinking of, one positive and one negative. Find the product of these two numbers.',
        'formatted': None,
        'type': None,
        'receiver': None
    }, {
        'sender': 'Agent',
        'content':
        'The product of the two numbers, \\(63\\) and \\(-63\\), is \\(-3969\\). Therefore, the product of the two numbers that Marie and Jay are thinking of is \\(\\boxed{-3969}\\).',
        'formatted': {
            'tool_type': None,
            'thought':
            'The product of the two numbers, \\(63\\) and \\(-63\\), is \\(-3969\\). Therefore, the product of the two numbers that Marie and Jay are thinking of is \\(\\boxed{-3969}\\).',
            'action': None,
            'status': 0
        },
        'type': None,
        'receiver': None
    }],
    'agent.memory': [{
        'sender': 'user',
        'content':
        'Marie is thinking of a multiple of 63, while Jay is thinking of a factor of 63. They happen to be thinking of the same number. There are two possibilities for the number that each of them is thinking of, one positive and one negative. Find the product of these two numbers.',
        'formatted': None,
        'type': None,
        'receiver': None
    }, {
        'sender': 'Agent',
        'content':
        "To solve this problem, we first need to identify the factors of 63. Since Marie is thinking of a multiple of 63 and Jay is thinking of a factor of 63, and they are thinking of the same number, we need to find the factors of 63 and then determine the multiples of 63 that match these factors.\n\nLet's start by finding the factors of 63 using Python.<|action_start|><|interpreter|>\n```python\nfrom sympy import divisors\n\n# Find the factors of 63\nfactors_of_63 = divisors(63)\nfactors_of_63\n```",
        'formatted': {
            'tool_type': 'interpreter',
            'thought':
            "To solve this problem, we first need to identify the factors of 63. Since Marie is thinking of a multiple of 63 and Jay is thinking of a factor of 63, and they are thinking of the same number, we need to find the factors of 63 and then determine the multiples of 63 that match these factors.\n\nLet's start by finding the factors of 63 using Python.",
            'action':
            '```python\nfrom sympy import divisors\n\n# Find the factors of 63\nfactors_of_63 = divisors(63)\nfactors_of_63\n```',
            'status': 1
        },
        'type': None,
        'receiver': None
    }, {
        'sender': 'ActionExecutor',
        'content': '[1, 3, 7, 9, 21, 63]',
        'formatted': None,
        'type': None,
        'receiver': None
    }, {
        'sender': 'Agent',
        'content':
        "The factors of 63 are \\(1, 3, 7, 9, 21,\\) and \\(63\\). Since Marie is thinking of a multiple of 63 and Jay is thinking of a factor of 63, and they are thinking of the same number, the number they are thinking of must be one of the factors of 63.\n\nGiven that there are two possibilities for the number they are thinking of, one positive and one negative, we need to consider both \\(63\\) and \\(-63\\) as the number they might be thinking of.\n\nNow, let's calculate the product of these two numbers, \\(63\\) and \\(-63\\).<|action_start|><|interpreter|>\n```python\n# Calculate the product of 63 and -63\nproduct = 63 * (-63)\nproduct\n```",
        'formatted': {
            'tool_type': 'interpreter',
            'thought':
            "The factors of 63 are \\(1, 3, 7, 9, 21,\\) and \\(63\\). Since Marie is thinking of a multiple of 63 and Jay is thinking of a factor of 63, and they are thinking of the same number, the number they are thinking of must be one of the factors of 63.\n\nGiven that there are two possibilities for the number they are thinking of, one positive and one negative, we need to consider both \\(63\\) and \\(-63\\) as the number they might be thinking of.\n\nNow, let's calculate the product of these two numbers, \\(63\\) and \\(-63\\).",
            'action':
            '```python\n# Calculate the product of 63 and -63\nproduct = 63 * (-63)\nproduct\n```',
            'status': 1
        },
        'type': None,
        'receiver': None
    }, {
        'sender': 'ActionExecutor',
        'content': '-3969',
        'formatted': None,
        'type': None,
        'receiver': None
    }, {
        'sender': 'Agent',
        'content':
        'The product of the two numbers, \\(63\\) and \\(-63\\), is \\(-3969\\). Therefore, the product of the two numbers that Marie and Jay are thinking of is \\(\\boxed{-3969}\\).',
        'formatted': {
            'tool_type': None,
            'thought':
            'The product of the two numbers, \\(63\\) and \\(-63\\), is \\(-3969\\). Therefore, the product of the two numbers that Marie and Jay are thinking of is \\(\\boxed{-3969}\\).',
            'action': None,
            'status': 0
        },
        'type': None,
        'receiver': None
    }]
}
coder.load_state_dict(state_dict)
print(coder.state_dict() == state_dict)
print(coder.state_dict())

# ----------------------- plugin -----------------------
print('-' * 80, 'plugin', '-' * 80)
plugins = [dict(type='lagent.actions.ArxivSearch')]
agent = AgentForInternLM(
    llm=model,
    plugins=plugins,
    output_format=dict(
        type=PluginParser,
        template=PLUGIN_CN,
        prompt=get_plugin_prompt(plugins)))

tic = time.time()
query = 'LLM智能体方向的最新论文有哪些？'
res = agent(query, session_id=0)
print(res.model_dump_json())
print('-' * 120)
print(agent.get_steps(0))
print(f'time elapsed: {time.time() - tic}')
