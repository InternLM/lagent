"""E2E test agent — real InternClawAgent with LLM.

Used by integration tests that need a real LLM call.
Requires network access to the API endpoint.
"""

from pathlib import Path

from lagent.agents.aggregator.context import InternClawContextBuilder
from lagent.agents.internclaw_agent import (
    AsyncEnvAgent,
    AsyncPolicyAgent,
    InternClawAgent,
)
from lagent.llms.model import AsyncAPIClient

name = "e2e-agent"
description = "E2E test agent with real LLM"
background = False

llm = dict(
    type=AsyncAPIClient,
    model=dict(
        model="gpt-5.4",
        base_url="http://35.220.164.252:3888/v1",
        api_key=" ",
        proxy="http://100.100.72.89:8899",
    ),
    sample_params=dict(temperature=0.1),
    timeout=60,
    max_retry=3,
    sleep_interval=1,
)

agent_config = dict(
    type=InternClawAgent,
    policy_agent=dict(
        type=AsyncPolicyAgent,
        llm=llm,
        aggregator=dict(type=InternClawContextBuilder, workspace=Path("/tmp")),
        name="policy",
    ),
    env_agent=dict(
        type=AsyncEnvAgent,
        actions=[],
        name="env",
    ),
    max_turn=3,
    finish_condition=lambda m, _: m is not None and not m.tool_calls,
)
