"""Example CLI agent config that wraps ``echo`` for testing."""
from lagent.adapters.cli_adapter import CLIAgentAdapter

name = "cli-echo"
description = "Echo agent for testing — returns the task as-is"
background = False

agent_config = dict(
    type=CLIAgentAdapter,
    name="cli-echo",
    description="Echo agent for testing",
    command_template="echo '{task}'",
    timeout=10,
)
