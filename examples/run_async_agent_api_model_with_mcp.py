import asyncio
import json

from lagent.actions import AsyncActionExecutor, AsyncMCPClient
from lagent.agents import AsyncAgentForInternLM
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.agents.stream import get_plugin_prompt
from lagent.llms import AsyncGPTAPI
from lagent.prompts import PluginParser
from lagent.schema import AgentMessage

TEMPLATE = (
    "You have access to the following tools:\n{tool_description}\nPlease provide"
    " your thought process when you need to use a tool, followed by the call statement in this format:"
    "\n{invocation_format}"
)
llm = dict(type=AsyncGPTAPI, model_type=None, retry=50, key=None, top_p=0.95, temperature=0.6, max_new_tokens=16384)
plugin = dict(
    type=AsyncMCPClient,
    name='PlayWright',
    server_type='stdio',
    command='npx',
    args=["@playwright/mcp@latest", '--isolated', '--no-sandbox'],
)
agent = AsyncAgentForInternLM(
    llm,
    plugin,
    template=TEMPLATE.format(
        tool_description=get_plugin_prompt(plugin),
        invocation_format='```json\n{"name": {{tool name}}, "parameters": {{keyword arguments}}}\n```\n',
    ),
    output_format=PluginParser(begin="```json\n", end="\n```\n", validate=lambda x: json.loads(x.rstrip('`'))),
    aggregator=InternLMToolAggregator(environment_role='system'),
)
msg = AgentMessage(
    sender='user',
    content='解释一下MCP中Sampling Flow的工作机制，参考https://modelcontextprotocol.io/docs/concepts/sampling',
)

# proj_dir = os.path.dirname(os.path.dirname(__file__))
# executor = AsyncActionExecutor(
#     dict(
#         type=AsyncMCPClient,
#         name='FS',
#         server_type='stdio',
#         command='npx',
#         args=['-y', '@modelcontextprotocol/server-filesystem', os.path.join(proj_dir, 'docs')],
#     )
# )
# msg = AgentMessage(
#     sender='assistant',
#     content=dict(
#         name='FS.read_file',
#         parameters=dict(path=os.path.join(proj_dir, 'docs/en/get_started/install.md')),
#     ),
# )
loop = asyncio.get_event_loop()
res = loop.run_until_complete(agent(msg))
print(res.content)
