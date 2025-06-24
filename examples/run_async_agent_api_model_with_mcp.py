import asyncio
import json
import os
import time
from typing import List

from volcenginesdkarkruntime import Ark, AsyncArk

from lagent.actions import AsyncActionExecutor, AsyncMCPClient
from lagent.agents import AsyncAgentForInternLM
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.agents.stream import get_plugin_prompt
from lagent.llms import AsyncGPTAPI
from lagent.prompts import PluginParser
from lagent.schema import AgentMessage


class LcAsyncAPI(AsyncGPTAPI):
    def __init__(self, model, api_key=None, max_tokens=4096, max_retries=4, **gen_params):
        if api_key is None:
            raise ValueError("api_key is required")
        self.model = model
        self.max_tokens = max_tokens
        self.retry = max_retries
        self.client = AsyncArk(
            api_key=api_key, timeout=900, max_retries=self.retry, base_url='https://ark.cn-beijing.volces.com/api/v3'
        )
        super().__init__(**gen_params)

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=gen_params.get('temperature', 0.8),
                    top_p=gen_params.get('top_p', 0.95),
                    stream=True,
                )
                reasoning_content = ""
                content = ""
                async for chunk in response:
                    if (
                        hasattr(chunk.choices[0].delta, 'reasoning_content')
                        and chunk.choices[0].delta.reasoning_content
                    ):
                        reasoning_content += chunk.choices[0].delta.reasoning_content
                        # print(chunk.choices[0].delta.reasoning_content, end="")
                    else:
                        content += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="")
                # response = json.loads(response.json())
                # reasoning_content = response['choices'][0]['message']['reasoning_content'].strip()
                # content = response['choices'][0]['message']['content'].strip()
                return content if reasoning_content == "" else "<think>" + reasoning_content + "</think>\n" + content

            except Exception as error:
                self.logger.error(str(error))
                time.sleep(20)
            max_num_retries += 1

        raise RuntimeError(
            'Calling OpenAI failed after retrying for ' f'{max_num_retries} times. Check the logs for ' 'details.'
        )


TEMPLATE = (
    "You have access to the following tools:\n{tool_description}\nPlease provide"
    " your thought process when you need to use a tool, followed by the call statement in this format:"
    "\n{invocation_format}"
)
llm = dict(type=LcAsyncAPI, model=None, api_key=None, top_p=0.95, temperature=0.6, max_tokens=16384, max_retries=50)
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
