"""Manual debug script for AgentDaemon E2E on Gateway sandbox.

Usage::

    /mnt/llm-ai-infra/miniconda3/envs/train/bin/python tests/test_actions/debug_agent_daemon.py

    # Custom gateway/image
    /mnt/llm-ai-infra/miniconda3/envs/train/bin/python tests/test_actions/debug_agent_daemon.py \\
        --gateway http://env-gateway.ailab.ailab.ai \\
        --image hb_3d-scan-calc \\
        --ttl 600
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lagent.serving.sandbox.providers.gateway import GatewayProvider
from lagent.serving.sandbox.agent import SandboxAgent
from workspace.agents.default_agent.config import agent_config

LAGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Minimal __init__.py to avoid optional dep issues
MINIMAL_ACTIONS_INIT = (
    "from .action_executor import ActionExecutor, AsyncActionExecutor\n"
    "from .base_action import AsyncActionMixin, BaseAction, tool_api\n"
    "from .builtin_actions import FinishAction, InvalidAction, NoAction\n"
    "from .parser import BaseParser, JsonParser, TupleParser\n"
)
MINIMAL_HOOKS_INIT = "from .hook import Hook, RemovableHandle\n"

# All deps needed by lagent in the sandbox
ALL_DEPS = (
    "griffe termcolor asyncer func_timeout openai jinja2 tiktoken "
    "aiohttp tenacity pydantic requests json5 jsonschema timeout-decorator"
)


def upload_text(client, path, content):
    b64 = base64.b64encode(content.encode()).decode()
    client.session.post(f"{client.url}/upload", json={
        "target_path": path, "content_b64": b64
    })


def upload_lagent_source(client):
    """Tar and upload lagent source + workspace to sandbox.

    - lagent source → /tmp/lagent/ (for PYTHONPATH)
    - workspace → /root/workspace/ (matches agent config)
    """
    # 1. Upload lagent source code → /tmp/
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ["lagent", "requirements"]:
            src = os.path.join(LAGENT_PATH, d)
            if not os.path.exists(src):
                continue
            for root, dirs, files in os.walk(src):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                for fname in files:
                    if fname.endswith(".pyc"):
                        continue
                    full = os.path.join(root, fname)
                    tar.add(full, arcname=os.path.relpath(full, LAGENT_PATH))
    content_b64 = base64.b64encode(buf.getvalue()).decode()
    client.session.post(f"{client.url}/upload", json={
        "target_path": "/tmp/lagent_src.tar.gz", "content_b64": content_b64
    })
    print(f"   Source: {len(buf.getvalue()) / 1024:.0f} KB → /tmp/")
    client.execute("cd /tmp && tar xzf lagent_src.tar.gz")

    # 2. Upload workspace → /root/workspace/ (matches config)
    ws_src = os.path.join(LAGENT_PATH, "workspace")
    if os.path.exists(ws_src):
        buf2 = io.BytesIO()
        with tarfile.open(fileobj=buf2, mode="w:gz") as tar:
            for root, dirs, files in os.walk(ws_src):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                for fname in files:
                    full = os.path.join(root, fname)
                    # arcname: workspace/skills/... → skills/...
                    arcname = os.path.relpath(full, ws_src)
                    tar.add(full, arcname=arcname)
        content_b64 = base64.b64encode(buf2.getvalue()).decode()
        client.session.post(f"{client.url}/upload", json={
            "target_path": "/root/workspace.tar.gz", "content_b64": content_b64
        })
        client.execute("mkdir -p /root/workspace && cd /root/workspace && tar xzf /root/workspace.tar.gz")
        print(f"   Workspace: {len(buf2.getvalue()) / 1024:.0f} KB → /root/workspace/")


def setup_sandbox(client):
    """Upload lagent, fix __init__.py, install deps."""
    print("[1/3] Uploading lagent source + workspace...")
    upload_lagent_source(client)

    print("[2/3] Fixing __init__.py (minimal imports)...")
    upload_text(client, "/tmp/lagent/actions/__init__.py", MINIMAL_ACTIONS_INIT)
    upload_text(client, "/tmp/lagent/hooks/__init__.py", MINIMAL_HOOKS_INIT)

    print("[3/3] Installing deps...")
    r = client.execute(f"/mnt/llm-ai-infra/miniconda3/envs/train/bin/python -m pip list 2>&1 ", timeout_sec=300)
    print(f"   {r.get('stdout', '').strip()[-120:]}")

    # Ensure workspace dirs exist (in case workspace upload didn't have them)
    client.execute("mkdir -p /root/workspace/memory /root/workspace/skills")


def verify_agent_creation(client):
    """Test that the agent config can be instantiated inside sandbox."""
    print("\n--- Verify agent creation ---")
    upload_text(client, "/tmp/agent_config.json", json.dumps(agent_config, ensure_ascii=False))
    upload_text(client, "/tmp/test_create.py", """
import json, traceback
from lagent.utils import create_object
config = json.load(open("/tmp/agent_config.json"))
try:
    agent = create_object(config)
    print("OK:", type(agent).__name__)
    print("PolicyAgent:", type(agent.policy_agent).__name__)
    print("EnvAgent:", type(agent.env_agent).__name__)
    if hasattr(agent.env_agent, 'actions'):
        print("Tools:", list(agent.env_agent.actions.actions.keys()))
except:
    traceback.print_exc()
""")
    r = client.execute(
        "PYTHONPATH=/tmp:$PYTHONPATH /mnt/llm-ai-infra/miniconda3/envs/train/bin/python /tmp/test_create.py 2>&1",
        timeout_sec=60,
    )
    # Read from file if stdout is truncated
    client.execute(
        "PYTHONPATH=/tmp:$PYTHONPATH /mnt/llm-ai-infra/miniconda3/envs/train/bin/python /tmp/test_create.py > /tmp/create_result.txt 2>&1",
        timeout_sec=60,
    )
    data = client.download_file("/tmp/create_result.txt")
    result = data.decode()
    print(result)
    return "OK:" in result


def start_agent_daemon(client, sock_path="/tmp/lagent_agent.sock"):
    """Start the AgentDaemon inside the sandbox."""
    print("\n--- Starting AgentDaemon ---")
    client.execute(
        f"PYTHONPATH=/tmp:$PYTHONPATH nohup /mnt/llm-ai-infra/miniconda3/envs/train/bin/python -m lagent.serving.sandbox.daemon start "
        f"--mode agent --config /tmp/agent_config.json "
        f"--sock {sock_path} "
        f"> /tmp/lagent_agent.log 2>&1 &"
    )
    print("   Waiting for socket...")
    for i in range(20):
        time.sleep(2)
        r = client.execute(f"test -S {sock_path} && echo 'ready' || echo 'waiting'")
        status = r.get("stdout", "").strip()
        if "ready" in status:
            print(f"   Socket ready! ({(i+1)*2}s)")
            return True
        print(f"   [{(i+1)*2}s] {status}")

    print("   FAILED. Daemon log:")
    data = client.download_file("/tmp/lagent_agent.log")
    print(data.decode()[:1000])
    return False


async def test_agent(client, sock_path="/tmp/lagent_agent.sock"):
    """Run interactive tests against the AgentDaemon."""
    # Patch client for PYTHONPATH
    original_exec = client.execute
    def patched(command, **kw):
        return original_exec(f"PYTHONPATH=/tmp:$PYTHONPATH {command}", **kw)
    client.execute = patched

    agent = SandboxAgent(
        sandbox_client=client,
        agent_config=agent_config,
        sock_path=sock_path,
    )
    agent._connected = True

    # Track log position for incremental tailing
    log_state = {"offset": 0}

    def show_daemon_logs(since_last=True):
        """Download and display daemon log (full or since last check)."""
        try:
            data = client.download_file("/tmp/lagent_agent.log")
            full_log = data.decode()
            if since_last:
                new_content = full_log[log_state["offset"]:]
                log_state["offset"] = len(full_log)
                if new_content.strip():
                    print(f"\n--- Daemon log (new) ---\n{new_content.rstrip()}")
            else:
                print(f"\n--- Daemon log (full) ---\n{full_log.rstrip()}")
        except Exception as e:
            print(f"   (failed to read log: {e})")

    # Ping
    print("\n--- Ping ---")
    r = await agent._daemon_call({"cmd": "ping"})
    print(f"   {r}")

    # List tools
    print("\n--- List tools ---")
    r = await agent._daemon_call({"cmd": "list_tools"})
    tools = [t["name"] for t in r.get("tools", [])]
    print(f"   {tools}")

    # Chat
    print("\n--- Chat ---")
    print("   Sending: '请执行 echo hello world 并告诉我结果'")
    response = await agent("请执行 echo hello world 并告诉我结果")
    print(f"   Response type: {type(response).__name__}")
    print(f"   Content: {str(response.content)[:500]}")
    show_daemon_logs()

    # State dict
    print("\n--- State dict ---")
    state = await agent.get_state_dict()
    print(f"   Keys: {list(state.keys())[:10]}")
    for k, v in state.items():
        if isinstance(v, list):
            print(f"   {k}: {len(v)} items")
        else:
            print(f"   {k}: {type(v).__name__}")

    # Interactive mode
    print("\n--- Interactive mode ---")
    print("   Commands: quit | state | reset | tools | logs | fulllog")
    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input == "state":
            state = await agent.get_state_dict()
            print(f"State: {json.dumps({k: type(v).__name__ for k, v in state.items()}, indent=2)}")
            continue
        if user_input == "reset":
            await agent.reset()
            log_state["offset"] = 0  # reset log tracking too
            print("Reset done.")
            continue
        if user_input == "tools":
            r = await agent._daemon_call({"cmd": "list_tools"})
            print(json.dumps([t["name"] for t in r.get("tools", [])], indent=2))
            continue
        if user_input == "logs":
            show_daemon_logs(since_last=True)
            continue
        if user_input == "fulllog":
            show_daemon_logs(since_last=False)
            continue

        try:
            response = await agent(user_input)
            print(f"\n[Agent] {str(response.content)[:1000]}")
        except Exception as e:
            print(f"\n[Error] {e}")
        # Show new daemon logs after every chat
        show_daemon_logs(since_last=True)


def main():
    parser = argparse.ArgumentParser(description="Debug AgentDaemon E2E on Gateway sandbox")
    parser.add_argument("--gateway", default="http://env-gateway.ailab.ailab.ai")
    parser.add_argument("--image", default="hb_3d-scan-calc")
    parser.add_argument("--ttl", type=int, default=600, help="Sandbox TTL in seconds")
    parser.add_argument("--sock", default="/tmp/lagent_agent.sock")
    parser.add_argument("--skip-setup", action="store_true", help="Skip upload/install (reuse existing sandbox)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AgentDaemon Debug — Gateway Sandbox")
    print("=" * 60)

    provider = GatewayProvider(args.gateway)
    print(f"\nCreating sandbox (image={args.image}, ttl={args.ttl}s)...")
    client, env_id = provider.create(image_tag=args.image, ttl_seconds=args.ttl)
    print(f"url: {client.url}")
    print(f"env_id: {env_id}")

    try:
        if not args.skip_setup:
            print(f"\n{'='*60}")
            print("  Setup")
            print(f"{'='*60}\n")
            setup_sandbox(client)

        print(f"\n{'='*60}")
        print("  Verify")
        print(f"{'='*60}")
        if not verify_agent_creation(client):
            print("\nAgent creation failed. Debug with:")
            print(f"  curl {client.url}/exec -X POST -H 'Content-Type: application/json' \\")
            print(f"    -d '{{\"command\": \"PYTHONPATH=/tmp:$PYTHONPATH /mnt/llm-ai-infra/miniconda3/envs/train/bin/python /tmp/test_create.py 2>&1\"}}'")
            return

        print(f"\n{'='*60}")
        print("  Daemon")
        print(f"{'='*60}")
        if not start_agent_daemon(client, args.sock):
            return

        print(f"\n{'='*60}")
        print("  Test")
        print(f"{'='*60}")
        asyncio.run(test_agent(client, args.sock))

    finally:
        print(f"\n{'='*60}")
        print(f"Cleaning up sandbox {env_id}...")
        provider.delete(env_id)
        print("Done.")


if __name__ == "__main__":
    main()
