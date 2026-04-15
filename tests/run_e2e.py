"""E2E: Load agent from project_template/config.py and run a task."""

import asyncio
import sys
from pathlib import Path

# Ensure lagent is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lagent.utils import create_object
from lagent.utils.config import Config


async def main():
    # ── Step 1: Load config ──────────────────────────────────────────
    config_path = Path(__file__).parent / "project_template" / "config.py"
    print(f"Loading config from: {config_path}")

    cfg = Config.fromfile(str(config_path))
    agent_config = cfg.agent_config
    build_fn = getattr(cfg, 'build', None)

    print(f"  name: {cfg.name}")
    print(f"  agent type: {agent_config['type'].__name__}")
    print(f"  build: {build_fn or 'create_object (default)'}")

    # ── Step 2: Build agent ──────────────────────────────────────────
    # Inject workspace path into aggregator config
    import copy
    agent_config = copy.deepcopy(agent_config)
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    agent_config['policy_agent']['aggregator']['workspace'] = workspace

    if build_fn and callable(build_fn):
        agent = await build_fn(agent_config)
    else:
        agent = create_object(agent_config)

    print(f"  agent created: {agent.__class__.__name__}")
    print(f"  policy: {agent.policy_agent.__class__.__name__}")
    print(f"  env: {agent.env_agent.__class__.__name__}")
    print(f"  actions: {list(agent.env_agent.actions.actions.keys())}")

    # ── Step 3: Run a simple task ────────────────────────────────────
    task = "List the files in the current directory using ls -la"
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")

    try:
        response = await agent(task)
        print(f"\n{'='*60}")
        print(f"Response: {response.content[:500] if response.content else 'empty'}")
        print(f"Finish reason: {getattr(response, 'finish_reason', 'N/A')}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup MCP connections
        for action in agent.env_agent.actions.actions.values():
            if hasattr(action, 'close'):
                await action.close()

    print("\n✅ E2E complete")


if __name__ == "__main__":
    asyncio.run(main())
