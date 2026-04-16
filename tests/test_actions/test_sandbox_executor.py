"""Tests for ActionDaemon, SandboxActionExecutor and HybridActionExecutor.

Run with::

    # Local unit tests (no sandbox needed, uses local daemon subprocess)
    pytest tests/test_actions/test_sandbox_executor.py -v

    # Real sandbox E2E (requires gateway access)
    RUN_E2E=1 pytest tests/test_actions/test_sandbox_executor.py -v -k "E2E"
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import tarfile
import time

import pytest
import requests

from lagent.serving.sandbox.daemon import ActionDaemon, async_lagent_call, lagent_call
from lagent.actions.hybrid_executor import HybridActionExecutor
from lagent.actions.sandbox_executor import (
    SandboxActionExecutor,
    _ToolDescriptionProxy,
    _ToolDescriptionStub,
    _deserialize_action_return,
)
from lagent.schema import ActionReturn, ActionStatusCode, ActionValidCode, AgentMessage, FunctionCall

# E2E tests are skipped unless RUN_E2E=1 environment variable is set
e2e = pytest.mark.skipif(
    not os.environ.get("RUN_E2E"), reason="Set RUN_E2E=1 to run real sandbox E2E tests"
)


# =====================================================================
# Part 1: Local unit tests (daemon subprocess, no real sandbox)
# =====================================================================

SOCK_PATH = "/tmp/lagent_test_daemon.sock"
CONFIG_PATH = "/tmp/lagent_test_actions.json"
ACTIONS_CONFIG = [
    {"type": "lagent.actions.python_interpreter.PythonInterpreter"},
]


@pytest.fixture(scope="module")
def daemon_process():
    """Start an ActionDaemon in a subprocess for the entire test module."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(ACTIONS_CONFIG, f)

    proc = subprocess.Popen(
        [
            "python", "-m", "lagent.serving.sandbox.daemon", "start",
            "--sock", SOCK_PATH,
            "--actions-config", CONFIG_PATH,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(30):
        if os.path.exists(SOCK_PATH):
            break
        time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("Daemon did not start")

    yield proc

    try:
        lagent_call(SOCK_PATH, '{"cmd":"shutdown"}')
    except Exception:
        pass
    time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=5)
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)


class LocalSandboxClient:
    """Mock sandbox client that runs bash locally (async)."""

    async def execute(self, command: str, cwd: str = "/tmp", timeout_sec: int = 30) -> dict:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": proc.returncode,
        }


class SyncLocalSandboxClient:
    """Sync mock sandbox client (like the user's real SandboxClient)."""

    def execute(self, command: str, cwd: str = "/tmp", timeout_sec: int = 30) -> dict:
        import subprocess as sp
        result = sp.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }


# ---------------------------------------------------------------------------
# ActionDaemon tests
# ---------------------------------------------------------------------------


class TestActionDaemon:

    def test_ping(self, daemon_process):
        result = json.loads(lagent_call(SOCK_PATH, '{"cmd":"ping"}'))
        assert result["status"] == "ok"
        assert result["type"] == "action"

    def test_list_tools(self, daemon_process):
        result = json.loads(lagent_call(SOCK_PATH, '{"cmd":"list_tools"}'))
        assert "tools" in result
        tools = result["tools"]
        assert len(tools) >= 1
        assert tools[0]["name"] == "PythonInterpreter"
        assert "parameters" in tools[0]

    def test_action_call_success(self, daemon_process):
        request = json.dumps({
            "name": "PythonInterpreter",
            "parameters": {"command": "def solution():\n    return 42"},
        })
        result = json.loads(lagent_call(SOCK_PATH, request))
        assert result["state"] == ActionStatusCode.SUCCESS
        assert result["result"][0]["content"] == "42"

    def test_action_call_error(self, daemon_process):
        request = json.dumps({
            "name": "PythonInterpreter",
            "parameters": {"command": "invalid code without solution()"},
        })
        result = json.loads(lagent_call(SOCK_PATH, request))
        assert result["state"] == ActionStatusCode.API_ERROR

    def test_missing_name(self, daemon_process):
        result = json.loads(lagent_call(SOCK_PATH, '{"parameters":{}}'))
        assert result["state"] == ActionStatusCode.ARGS_ERROR

    def test_unknown_action(self, daemon_process):
        request = json.dumps({
            "name": "NonExistentAction",
            "parameters": {},
        })
        result = json.loads(lagent_call(SOCK_PATH, request))
        assert result["state"] in (ActionStatusCode.ARGS_ERROR, ActionStatusCode.API_ERROR)

    @pytest.mark.asyncio
    async def test_async_call(self, daemon_process):
        result = await async_lagent_call(SOCK_PATH, '{"cmd":"ping"}')
        parsed = json.loads(result)
        assert parsed["status"] == "ok"


# ---------------------------------------------------------------------------
# Deserialization tests
# ---------------------------------------------------------------------------


class TestDeserialization:

    def test_deserialize_success(self):
        text = json.dumps({
            "args": {"command": "ls"},
            "type": "ShellAction",
            "result": [{"type": "text", "content": "file.txt"}],
            "state": 0,
            "valid": 0,
            "errmsg": None,
            "url": None,
            "thought": None,
        })
        ar = _deserialize_action_return(text)
        assert isinstance(ar, ActionReturn)
        assert ar.state == ActionStatusCode.SUCCESS
        assert ar.valid == ActionValidCode.OPEN
        assert ar.result[0]["content"] == "file.txt"

    def test_deserialize_error(self):
        text = json.dumps({
            "args": {},
            "state": -1002,
            "valid": 0,
            "errmsg": "something broke",
        })
        ar = _deserialize_action_return(text)
        assert ar.state == ActionStatusCode.API_ERROR
        assert ar.errmsg == "something broke"


# ---------------------------------------------------------------------------
# ToolDescriptionProxy tests
# ---------------------------------------------------------------------------


class TestToolDescriptionProxy:

    def test_proxy_contains(self):
        proxy = _ToolDescriptionProxy({"shell": {"name": "shell"}})
        assert "shell" in proxy
        assert "missing" not in proxy

    def test_proxy_values(self):
        proxy = _ToolDescriptionProxy({
            "a": {"name": "a", "description": "A"},
            "b": {"name": "b", "description": "B"},
        })
        stubs = proxy.values()
        assert len(stubs) == 2
        assert all(isinstance(s, _ToolDescriptionStub) for s in stubs)
        assert stubs[0].name == "a"
        assert stubs[0].is_toolkit is False

    def test_proxy_keys(self):
        proxy = _ToolDescriptionProxy({"x": {"name": "x"}, "y": {"name": "y"}})
        assert proxy.keys() == ["x", "y"]


# ---------------------------------------------------------------------------
# SandboxActionExecutor tests (async mock client)
# ---------------------------------------------------------------------------


class TestSandboxActionExecutorAsync:

    @pytest.fixture()
    def executor(self, daemon_process):
        client = LocalSandboxClient()
        ex = SandboxActionExecutor(
            sandbox_client=client,
            actions_config=ACTIONS_CONFIG,
            sock_path=SOCK_PATH,
            cwd="/tmp",
        )
        ex._connected = True
        ex._tool_descriptions = {
            "PythonInterpreter": {
                "name": "PythonInterpreter",
                "description": "...",
                "parameters": [{"name": "command", "type": "STRING", "description": ""}],
                "required": ["command"],
            }
        }
        return ex

    @pytest.mark.asyncio
    async def test_forward_success(self, executor):
        result = await executor.forward(
            "PythonInterpreter",
            {"command": "def solution():\n    return 99"},
        )
        assert result.state == ActionStatusCode.SUCCESS
        assert result.result[0]["content"] == "99"
        assert result.valid == ActionValidCode.OPEN

    @pytest.mark.asyncio
    async def test_forward_unknown_action(self, executor):
        result = await executor.forward("NonExistent", {"x": 1})
        assert result.valid == ActionValidCode.INVALID

    @pytest.mark.asyncio
    async def test_contains(self, executor):
        assert "PythonInterpreter" in executor
        assert "NonExistent" not in executor

    @pytest.mark.asyncio
    async def test_description(self, executor):
        descs = executor.description()
        assert len(descs) == 1
        assert descs[0]["name"] == "PythonInterpreter"

    @pytest.mark.asyncio
    async def test_call_with_agent_message(self, executor):
        msg = AgentMessage(
            sender="test",
            content=FunctionCall(
                name="PythonInterpreter",
                parameters={"command": "def solution():\n    return 7"},
            ),
        )
        response = await executor(msg)
        assert isinstance(response, AgentMessage)
        ar = response.content
        assert isinstance(ar, ActionReturn)
        assert ar.result[0]["content"] == "7"


# ---------------------------------------------------------------------------
# SandboxActionExecutor with sync client
# ---------------------------------------------------------------------------


class TestSandboxActionExecutorSync:

    @pytest.fixture()
    def executor(self, daemon_process):
        client = SyncLocalSandboxClient()
        ex = SandboxActionExecutor(
            sandbox_client=client,
            actions_config=ACTIONS_CONFIG,
            sock_path=SOCK_PATH,
            cwd="/tmp",
        )
        ex._connected = True
        ex._tool_descriptions = {
            "PythonInterpreter": {
                "name": "PythonInterpreter",
                "description": "...",
                "parameters": [{"name": "command", "type": "STRING", "description": ""}],
                "required": ["command"],
            }
        }
        return ex

    @pytest.mark.asyncio
    async def test_sync_client_forward(self, executor):
        """Verify sync client is wrapped with to_thread and works."""
        result = await executor.forward(
            "PythonInterpreter",
            {"command": "def solution():\n    return 55"},
        )
        assert result.state == ActionStatusCode.SUCCESS
        assert result.result[0]["content"] == "55"


# ---------------------------------------------------------------------------
# HybridActionExecutor tests
# ---------------------------------------------------------------------------


class TestHybridActionExecutor:

    @pytest.fixture()
    def hybrid(self, daemon_process):
        client = LocalSandboxClient()
        sandbox = SandboxActionExecutor(
            sandbox_client=client,
            actions_config=ACTIONS_CONFIG,
            sock_path=SOCK_PATH,
            cwd="/tmp",
        )
        sandbox._connected = True
        sandbox._tool_descriptions = {
            "PythonInterpreter": {
                "name": "PythonInterpreter",
                "description": "...",
                "parameters": [{"name": "command", "type": "STRING", "description": ""}],
                "required": ["command"],
            }
        }
        return HybridActionExecutor(
            local_actions=[],
            sandbox_executor=sandbox,
        )

    @pytest.mark.asyncio
    async def test_route_to_sandbox(self, hybrid):
        result = await hybrid.forward(
            "PythonInterpreter",
            {"command": "def solution():\n    return 123"},
        )
        assert result.state == ActionStatusCode.SUCCESS
        assert result.result[0]["content"] == "123"

    @pytest.mark.asyncio
    async def test_route_unknown_to_invalid(self, hybrid):
        result = await hybrid.forward("DoesNotExist", {})
        assert result.valid == ActionValidCode.INVALID

    def test_description_merges(self, hybrid):
        descs = hybrid.description()
        names = [d["name"] for d in descs]
        assert "PythonInterpreter" in names

    def test_contains_sandbox_tools(self, hybrid):
        assert "PythonInterpreter" in hybrid

    def test_keys_merged(self, hybrid):
        assert "PythonInterpreter" in hybrid.keys()


# =====================================================================
# Part 2: Real sandbox E2E tests
#
# Run with: pytest tests/test_actions/test_sandbox_executor.py -v -k "E2E" --run-e2e
# =====================================================================

GATEWAY_URL = "http://env-gateway.ailab.ailab.ai"
IMAGE_TAG = "hb_3d-scan-calc"
E2E_ACTIONS_CONFIG = [
    {"type": "lagent.actions.shell.ShellAction", "working_dir": "/root"},
    {"type": "lagent.actions.filesystem.ReadFileAction", "workspace": "/root"},
    {"type": "lagent.actions.filesystem.WriteFileAction", "workspace": "/root"},
    {"type": "lagent.actions.filesystem.EditFileAction", "workspace": "/root"},
    {"type": "lagent.actions.python_interpreter.PythonInterpreter"},
]
# Minimal __init__.py — avoids importing actions with heavy optional deps
_MINIMAL_ACTIONS_INIT = (
    "from .action_executor import ActionExecutor, AsyncActionExecutor\n"
    "from .base_action import AsyncActionMixin, BaseAction, tool_api\n"
    "from .builtin_actions import FinishAction, InvalidAction, NoAction\n"
    "from .parser import BaseParser, JsonParser, TupleParser\n"
)
_MINIMAL_HOOKS_INIT = "from .hook import Hook, RemovableHandle\n"
_REQUIRED_PIPS = "griffe termcolor asyncer func_timeout"


class RealSandboxClient:
    """Wraps the real EnvGateway SandboxClient for use with SandboxActionExecutor.

    Prepends PYTHONPATH so the uploaded lagent source is importable.
    """

    def __init__(self, base_url: str, pythonpath: str = "/tmp"):
        self.base_url = base_url
        self.pythonpath = pythonpath
        self.session = requests.Session()
        self.session.headers.update({
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        })

    def execute(self, command: str, cwd: str = "/root", timeout_sec: int = 60) -> dict:
        command = f"PYTHONPATH={self.pythonpath}:$PYTHONPATH {command}"
        resp = self.session.post(
            f"{self.base_url}/exec",
            json={"command": command, "cwd": cwd, "timeout_sec": timeout_sec},
        )
        return resp.json()


@pytest.fixture(scope="module")
def sandbox_env():
    """Create a real sandbox, upload lagent, install deps, start daemon.

    Yields (sandbox_url, env_id, client).
    Cleans up the sandbox after all E2E tests finish.
    """
    gw = requests.Session()
    gw.headers.update({"Content-Type": "application/json"})

    # 1. Create sandbox
    resp = gw.post(
        f"{GATEWAY_URL}/envs",
        json={"image_tag": IMAGE_TAG, "ttl_seconds": 600},
        timeout=120,
    )
    ret = resp.json()
    assert ret["ok"], f"Failed to create sandbox: {ret}"
    sandbox_url = ret["env"]["url"]
    env_id = ret["env"]["env_id"]

    session = requests.Session()
    session.headers.update({"Connection": "keep-alive", "Content-Type": "application/json"})

    def _exec(cmd, timeout=120):
        r = session.post(
            f"{sandbox_url}/exec",
            json={"command": cmd, "cwd": "/root", "timeout_sec": timeout},
        )
        return r.json()

    # 2. Upload lagent source
    buf = io.BytesIO()
    lagent_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lagent_pkg = os.path.join(lagent_root, "lagent")
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(lagent_pkg):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if fname.endswith(".pyc"):
                    continue
                full_path = os.path.join(root, fname)
                tar.add(full_path, arcname=os.path.relpath(full_path, lagent_root))
    content_b64 = base64.b64encode(buf.getvalue()).decode()
    resp = session.post(
        f"{sandbox_url}/upload",
        json={"target_path": "/tmp/lagent_src.tar.gz", "content_b64": content_b64},
    )
    assert resp.json()["ok"], "Upload failed"
    _exec("cd /tmp && tar xzf lagent_src.tar.gz")

    # 3. Write minimal __init__.py (avoid optional deps)
    _exec(f"cat > /tmp/lagent/actions/__init__.py << 'EOF'\n{_MINIMAL_ACTIONS_INIT}EOF")
    _exec(f"cat > /tmp/lagent/hooks/__init__.py << 'EOF'\n{_MINIMAL_HOOKS_INIT}EOF")

    # 4. Install required deps
    _exec(f"pip install {_REQUIRED_PIPS} 2>&1 | tail -3", timeout=120)

    # 5. Start daemon
    config_json = json.dumps(E2E_ACTIONS_CONFIG)
    _exec(f"echo '{config_json}' > /tmp/lagent_actions_config.json")
    _exec(
        "PYTHONPATH=/tmp:$PYTHONPATH nohup python -m lagent.serving.sandbox.daemon start "
        "--sock /tmp/lagent_action.sock "
        "--actions-config /tmp/lagent_actions_config.json "
        "> /tmp/lagent_daemon.log 2>&1 &"
    )
    time.sleep(3)
    r = _exec("test -S /tmp/lagent_action.sock && echo 'ready'")
    assert "ready" in r.get("stdout", ""), (
        f"Daemon did not start: {_exec('cat /tmp/lagent_daemon.log')}"
    )

    client = RealSandboxClient(sandbox_url)

    yield sandbox_url, env_id, client

    # Cleanup
    try:
        gw.delete(f"{GATEWAY_URL}/envs/{env_id}", timeout=30)
    except Exception:
        pass


@pytest.fixture()
def e2e_executor(sandbox_env):
    """Create a connected SandboxActionExecutor for E2E tests."""
    sandbox_url, env_id, client = sandbox_env

    async def _setup():
        executor = SandboxActionExecutor(
            sandbox_client=client,
            actions_config=E2E_ACTIONS_CONFIG,
            sock_path="/tmp/lagent_action.sock",
        )
        executor._connected = True
        result = await executor._daemon_call({"cmd": "list_tools"})
        executor._tool_descriptions = {t["name"]: t for t in result["tools"]}
        return executor

    loop = asyncio.new_event_loop()
    executor = loop.run_until_complete(_setup())
    loop.close()
    return executor


# ---------------------------------------------------------------------------
# E2E test class
# ---------------------------------------------------------------------------


@e2e
class TestSandboxE2E:
    """Real sandbox E2E tests — shell, file ops, python, cross-action state."""

    def test_tools_registered(self, e2e_executor):
        tools = e2e_executor.keys()
        assert "ShellAction" in tools
        assert "ReadFileAction" in tools
        assert "WriteFileAction" in tools
        assert "EditFileAction" in tools
        assert "PythonInterpreter" in tools

    @pytest.mark.asyncio
    async def test_shell_echo(self, e2e_executor):
        ar = await e2e_executor.forward("ShellAction", {"command": "echo hello"})
        assert ar.state == ActionStatusCode.SUCCESS
        assert "hello" in ar.result[0]["content"]

    @pytest.mark.asyncio
    async def test_shell_ls(self, e2e_executor):
        ar = await e2e_executor.forward("ShellAction", {"command": "ls /"})
        assert ar.state == ActionStatusCode.SUCCESS
        assert "root" in ar.result[0]["content"] or "tmp" in ar.result[0]["content"]

    @pytest.mark.asyncio
    async def test_write_file(self, e2e_executor):
        ar = await e2e_executor.forward("WriteFileAction", {
            "path": "e2e_test.txt",
            "content": "Line 1: Hello\nLine 2: World\nLine 3: End\n",
        })
        assert ar.state == ActionStatusCode.SUCCESS

    @pytest.mark.asyncio
    async def test_read_file(self, e2e_executor):
        # Ensure file exists
        await e2e_executor.forward("WriteFileAction", {
            "path": "e2e_test.txt",
            "content": "Line 1: Hello\nLine 2: World\nLine 3: End\n",
        })
        ar = await e2e_executor.forward("ReadFileAction", {"path": "e2e_test.txt"})
        assert ar.state == ActionStatusCode.SUCCESS
        assert "Hello" in ar.result[0]["content"]
        assert "Line 2: World" in ar.result[0]["content"]

    @pytest.mark.asyncio
    async def test_edit_file(self, e2e_executor):
        # Write → Edit → Read-back
        await e2e_executor.forward("WriteFileAction", {
            "path": "e2e_edit.txt",
            "content": "before edit\n",
        })
        ar = await e2e_executor.forward("EditFileAction", {
            "path": "e2e_edit.txt",
            "search": "before edit",
            "replace": "AFTER EDIT",
        })
        assert ar.state == ActionStatusCode.SUCCESS

        ar = await e2e_executor.forward("ReadFileAction", {"path": "e2e_edit.txt"})
        assert "AFTER EDIT" in ar.result[0]["content"]

    @pytest.mark.asyncio
    async def test_python_interpreter(self, e2e_executor):
        ar = await e2e_executor.forward("PythonInterpreter", {
            "command": "def solution():\n    return 6 * 7",
        })
        assert ar.state == ActionStatusCode.SUCCESS
        assert ar.result[0]["content"] == "42"

    @pytest.mark.asyncio
    async def test_cross_action_state(self, e2e_executor):
        """File created by WriteFileAction should be visible to ShellAction."""
        await e2e_executor.forward("WriteFileAction", {
            "path": "cross_test.txt",
            "content": "CROSS_ACTION_OK\n",
        })
        ar = await e2e_executor.forward("ShellAction", {"command": "cat /root/cross_test.txt"})
        assert ar.state == ActionStatusCode.SUCCESS
        assert "CROSS_ACTION_OK" in ar.result[0]["content"]

    @pytest.mark.asyncio
    async def test_hybrid_routing(self, e2e_executor):
        """HybridActionExecutor routes sandbox tools correctly."""
        hybrid = HybridActionExecutor(local_actions=[], sandbox_executor=e2e_executor)

        ar = await hybrid.forward("ShellAction", {"command": "echo hybrid_ok"})
        assert ar.state == ActionStatusCode.SUCCESS
        assert "hybrid_ok" in ar.result[0]["content"]

        ar = await hybrid.forward("DoesNotExist", {})
        assert ar.valid == ActionValidCode.INVALID

    @pytest.mark.asyncio
    async def test_unknown_action(self, e2e_executor):
        ar = await e2e_executor.forward("FakeAction", {})
        assert ar.valid == ActionValidCode.INVALID


# =====================================================================
# Direct invocation entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sandbox E2E tests")
    parser.add_argument("--gateway", default=GATEWAY_URL, help="Gateway URL")
    parser.add_argument("--image", default=IMAGE_TAG, help="Sandbox image tag")
    args = parser.parse_args()

    GATEWAY_URL = args.gateway
    IMAGE_TAG = args.image

    async def main():
        # --- Setup ---
        print("=" * 60)
        print("  Sandbox E2E Test")
        print("=" * 60)

        gw = requests.Session()
        gw.headers.update({"Content-Type": "application/json"})

        print(f"\n[1/5] Creating sandbox (image={IMAGE_TAG})...")
        resp = gw.post(
            f"{GATEWAY_URL}/envs",
            json={"image_tag": IMAGE_TAG, "ttl_seconds": 600},
            timeout=120,
        )
        ret = resp.json()
        assert ret["ok"], f"Failed: {ret}"
        sandbox_url = ret["env"]["url"]
        env_id = ret["env"]["env_id"]
        print(f"       url={sandbox_url}")

        session = requests.Session()
        session.headers.update({"Connection": "keep-alive", "Content-Type": "application/json"})

        def _exec(cmd, timeout=120):
            r = session.post(
                f"{sandbox_url}/exec",
                json={"command": cmd, "cwd": "/root", "timeout_sec": timeout},
            )
            return r.json()

        try:
            # Upload lagent
            print("[2/5] Uploading lagent source...")
            buf = io.BytesIO()
            lagent_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lagent_pkg = os.path.join(lagent_root, "lagent")
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for root, dirs, files in os.walk(lagent_pkg):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for fname in files:
                        if fname.endswith(".pyc"):
                            continue
                        full_path = os.path.join(root, fname)
                        tar.add(full_path, arcname=os.path.relpath(full_path, lagent_root))
            content_b64 = base64.b64encode(buf.getvalue()).decode()
            resp = session.post(
                f"{sandbox_url}/upload",
                json={"target_path": "/tmp/lagent_src.tar.gz", "content_b64": content_b64},
            )
            assert resp.json()["ok"]
            _exec("cd /tmp && tar xzf lagent_src.tar.gz")
            _exec(f"cat > /tmp/lagent/actions/__init__.py << 'EOF'\n{_MINIMAL_ACTIONS_INIT}EOF")
            _exec(f"cat > /tmp/lagent/hooks/__init__.py << 'EOF'\n{_MINIMAL_HOOKS_INIT}EOF")

            # Install deps
            print("[3/5] Installing dependencies...")
            _exec(f"pip install {_REQUIRED_PIPS} 2>&1 | tail -3", timeout=120)

            # Start daemon
            print("[4/5] Starting daemon...")
            config_json = json.dumps(E2E_ACTIONS_CONFIG)
            _exec(f"echo '{config_json}' > /tmp/lagent_actions_config.json")
            _exec(
                "PYTHONPATH=/tmp:$PYTHONPATH nohup python -m lagent.serving.sandbox.daemon start "
                "--sock /tmp/lagent_action.sock "
                "--actions-config /tmp/lagent_actions_config.json "
                "> /tmp/lagent_daemon.log 2>&1 &"
            )
            time.sleep(3)
            r = _exec("test -S /tmp/lagent_action.sock && echo 'ready'")
            assert "ready" in r.get("stdout", ""), f"Daemon failed: {_exec('cat /tmp/lagent_daemon.log')}"

            # Run tests
            print("[5/5] Running tests...\n")
            client = RealSandboxClient(sandbox_url)
            executor = SandboxActionExecutor(
                sandbox_client=client,
                actions_config=E2E_ACTIONS_CONFIG,
                sock_path="/tmp/lagent_action.sock",
            )
            executor._connected = True
            result = await executor._daemon_call({"cmd": "list_tools"})
            executor._tool_descriptions = {t["name"]: t for t in result["tools"]}
            hybrid = HybridActionExecutor(local_actions=[], sandbox_executor=executor)

            tools = executor.keys()
            print(f"  Tools: {tools}")

            passed = 0
            failed = 0

            async def check(name, coro, assertion):
                nonlocal passed, failed
                try:
                    ar = await coro
                    assertion(ar)
                    print(f"  PASS  {name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {name}: {e}")
                    failed += 1

            await check(
                "shell(echo)",
                executor.forward("ShellAction", {"command": "echo hello"}),
                lambda ar: (
                    assert_eq(ar.state, ActionStatusCode.SUCCESS),
                    assert_in("hello", ar.result[0]["content"]),
                ),
            )
            await check(
                "shell(ls /)",
                executor.forward("ShellAction", {"command": "ls /"}),
                lambda ar: assert_eq(ar.state, ActionStatusCode.SUCCESS),
            )
            await check(
                "write_file",
                executor.forward("WriteFileAction", {"path": "e2e.txt", "content": "HELLO\nWORLD\n"}),
                lambda ar: assert_eq(ar.state, ActionStatusCode.SUCCESS),
            )
            await check(
                "read_file",
                executor.forward("ReadFileAction", {"path": "e2e.txt"}),
                lambda ar: (
                    assert_eq(ar.state, ActionStatusCode.SUCCESS),
                    assert_in("HELLO", ar.result[0]["content"]),
                ),
            )
            await check(
                "edit_file",
                executor.forward("EditFileAction", {"path": "e2e.txt", "search": "WORLD", "replace": "EDITED"}),
                lambda ar: assert_eq(ar.state, ActionStatusCode.SUCCESS),
            )

            # Verify edit
            ar = await executor.forward("ReadFileAction", {"path": "e2e.txt"})
            await check(
                "verify_edit",
                asyncio.coroutine(lambda: ar)(),
                lambda ar: assert_in("EDITED", ar.result[0]["content"]),
            ) if False else None  # skip the coroutine trick
            assert "EDITED" in ar.result[0]["content"], "Edit verification failed"
            print("  PASS  verify_edit")
            passed += 1

            await check(
                "python(6*7)",
                executor.forward("PythonInterpreter", {"command": "def solution():\n    return 6*7"}),
                lambda ar: (
                    assert_eq(ar.state, ActionStatusCode.SUCCESS),
                    assert_eq(ar.result[0]["content"], "42"),
                ),
            )
            await check(
                "cross_action_state",
                hybrid.forward("ShellAction", {"command": "cat /root/e2e.txt"}),
                lambda ar: assert_in("EDITED", ar.result[0]["content"]),
            )
            await check(
                "hybrid_routing",
                hybrid.forward("ShellAction", {"command": "echo ok"}),
                lambda ar: assert_in("ok", ar.result[0]["content"]),
            )
            await check(
                "unknown_action",
                hybrid.forward("FakeAction", {}),
                lambda ar: assert_eq(ar.valid, ActionValidCode.INVALID),
            )

            print(f"\n{'=' * 60}")
            print(f"  Results: {passed} passed, {failed} failed")
            print(f"{'=' * 60}")
            if failed:
                exit(1)

        finally:
            print("\nCleaning up sandbox...")
            try:
                gw.delete(f"{GATEWAY_URL}/envs/{env_id}", timeout=30)
                print("  Done.")
            except Exception as e:
                print(f"  Cleanup failed: {e}")

    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    def assert_in(needle, haystack):
        assert needle in haystack, f"{needle!r} not in {haystack!r}"

    asyncio.run(main())


# =====================================================================
# Part 3: AgentDaemon E2E tests
#
# Tests the full InternClawAgent running inside a sandbox via AgentDaemon.
# Requires: RUN_AGENT_E2E=1 + LLM server accessible from sandbox.
#
# Run with:
#   RUN_AGENT_E2E=1 python tests/test_actions/test_sandbox_executor.py --agent-e2e
#   or
#   RUN_AGENT_E2E=1 pytest tests/test_actions/test_sandbox_executor.py -v -k "AgentDaemon"
# =====================================================================

agent_e2e = pytest.mark.skipif(
    not os.environ.get("RUN_AGENT_E2E"),
    reason="Set RUN_AGENT_E2E=1 to run AgentDaemon E2E tests",
)

# ClusterX config (shared storage, deps pre-installed)
CLUSTERX_PARTITION = "llmit_proxy"
CLUSTERX_CONDA_ENV = "xtuner_dev"
CLUSTERX_CONDA_ACTIVATE = "/mnt/shared-storage-user/liukuikun/miniconda3/bin/activate"
LAGENT_PATH = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"


@pytest.fixture(scope="module")
def agent_sandbox_env():
    """Create a ClusterX sandbox with SandboxServer for AgentDaemon tests.

    Uses ClusterX because the shared storage already has lagent + deps.
    """
    import sys
    sys.path.insert(0, LAGENT_PATH)

    try:
        from lagent.serving.sandbox.providers.clusterx import ClusterXProvider
    except ImportError:
        pytest.skip("clusterx not available")

    provider = ClusterXProvider(
        partition=CLUSTERX_PARTITION,
        conda_env=CLUSTERX_CONDA_ENV,
        conda_activate_path=CLUSTERX_CONDA_ACTIVATE,
        python_path=LAGENT_PATH,
        port=19876,
        extra_run_kwargs={
            "cpus_per_task": 4,
            "memory_per_task": 10,
            "no_env": True,
        },
    )

    client, job_id = provider.create(timeout=300)

    yield client, job_id, provider

    try:
        provider.delete(job_id)
    except Exception:
        pass


@pytest.fixture()
def agent_daemon_client(agent_sandbox_env):
    """Start AgentDaemon in the sandbox and return a connected SandboxAgent."""
    import json as _json
    from lagent.serving.sandbox.agent import SandboxAgent

    client, job_id, provider = agent_sandbox_env

    # Patch client for PYTHONPATH + conda
    original_exec = client.execute
    prefix = (
        f"source {CLUSTERX_CONDA_ACTIVATE} {CLUSTERX_CONDA_ENV} && "
        f"PYTHONPATH={LAGENT_PATH}:$PYTHONPATH "
    )

    def patched_execute(command, **kw):
        return original_exec(f"{prefix}{command}", **kw)

    client.execute = patched_execute

    # Write agent config
    from workspace.agents.default_agent.config import agent_config
    config_json = _json.dumps(agent_config, ensure_ascii=False)
    escaped = config_json.replace("'", "'\\''")
    client.execute(f"echo '{escaped}' > /tmp/agent_config.json")
    client.execute("mkdir -p /root/workspace/memory /root/workspace/skills")

    # Start daemon
    sock_path = "/tmp/lagent_agent_e2e.sock"
    client.execute(
        f"nohup python -m lagent.serving.sandbox.daemon start "
        f"--mode agent --config /tmp/agent_config.json "
        f"--sock {sock_path} "
        f"> /tmp/lagent_agent_e2e.log 2>&1 &"
    )

    import time
    time.sleep(8)

    r = client.execute(f"test -S {sock_path} && echo 'ready' || echo 'not ready'")
    if "ready" not in r.get("stdout", ""):
        r = client.execute(f"tail -30 /tmp/lagent_agent_e2e.log")
        pytest.fail(f"AgentDaemon failed to start:\n{r.get('stdout', '')}")

    agent = SandboxAgent(
        sandbox_client=client,
        agent_config=agent_config,
        sock_path=sock_path,
    )
    agent._connected = True
    return agent


@agent_e2e
class TestAgentDaemonE2E:
    """Full InternClawAgent running inside a sandbox via AgentDaemon."""

    @pytest.mark.asyncio
    async def test_ping(self, agent_daemon_client):
        r = await agent_daemon_client._daemon_call({"cmd": "ping"})
        assert r["status"] == "ok"
        assert r["type"] == "agent"

    @pytest.mark.asyncio
    async def test_list_tools(self, agent_daemon_client):
        r = await agent_daemon_client._daemon_call({"cmd": "list_tools"})
        tools = r.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert len(tools) > 0
        # Should have at least shell and file actions
        assert any("Shell" in n or "shell" in n for n in tool_names)

    @pytest.mark.asyncio
    async def test_chat(self, agent_daemon_client):
        response = await agent_daemon_client("请执行 echo hello world 并告诉我结果")
        assert response.content is not None
        assert len(str(response.content)) > 0

    @pytest.mark.asyncio
    async def test_state_dict(self, agent_daemon_client):
        # Chat first to have some state
        await agent_daemon_client("执行 echo test")
        state = await agent_daemon_client.get_state_dict()
        assert isinstance(state, dict)

    @pytest.mark.asyncio
    async def test_reset(self, agent_daemon_client):
        await agent_daemon_client.reset()
        # After reset, should still be able to chat
        r = await agent_daemon_client._daemon_call({"cmd": "ping"})
        assert r["status"] == "ok"
