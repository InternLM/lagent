"""Agent Project Config — PyConfig entry point.

The only required file for an agent project.
AgentLoader reads ``agent_config`` and calls ``build(agent_config)``
to create the Agent instance.

Required exports
----------------
agent_config : dict
    A PyConfig dict recognized by ``create_object()``.

Optional exports
----------------
name        : str       — Agent type name (defaults to directory name).
description : str       — One-line description.
max_turns   : int       — Max interaction turns (default 500).
background  : bool      — Run in background (default False).
build       : callable  — Custom build function: ``(config_dict) -> Agent``.
                          When unset, defaults to ``create_object(agent_config)``.
"""

from lagent.actions.mcp_client import AsyncMCPClientSandbox
from lagent.agents import AsyncAgent
from lagent.agents.aggregator.context import InternClawContextBuilder
from lagent.agents.internclaw_agent import AsyncEnvAgent, InternClawAgent
from lagent.llms.model import AsyncAPIClient

# ── Metadata ──────────────────────────────────────────────────────────

name = "internclaw-standard"
description = "Standard InternClaw Agent: Policy-Env dual loop with sandbox"
background = False

# ── Sub-component configs ─────────────────────────────────────────────

llm = dict(
    type=AsyncAPIClient,
    model=dict(
        model="gpt-5.4",
        base_url="http://35.220.164.252:3888/v1",
        api_key=" ",
        proxy="http://100.100.72.89:8899",
    ),
    sample_params=dict(temperature=0.7, top_p=1.0, top_k=50),
    timeout=600,
    max_retry=50,
)

sandbox_action = dict(
    type=AsyncMCPClientSandbox,
    server_type='http',
    url="http://simple-shell.ailab.ailab.ai/mcp",
)

# ── Full Agent Config ─────────────────────────────────────────────────

agent_config = dict(
    type=InternClawAgent,
    policy_agent=dict(
        type=AsyncAgent,
        llm=llm,
        aggregator=dict(type=InternClawContextBuilder),
        name="policy",
    ),
    env_agent=dict(
        type=AsyncEnvAgent,
        actions=[sandbox_action],
        name="env",
    ),
    max_turn=500,
    # workspace 默认 cwd，运行时可覆盖
    # finish_condition 用默认值（无 tool_calls 时停止）
)

# build = None  (default: create_object)
# To add runtime dependencies like CompactAction, define a custom build:
#
# async def build(config):
#     from lagent.utils import create_object
#     agent = create_object(config)
#     # ... inject runtime deps ...
#     return agent
