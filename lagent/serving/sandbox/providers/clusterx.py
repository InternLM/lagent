"""ClusterX sandboxes using an existing, optionally shared CLI runtime.

The host never imports clusterx. A subprocess in the CLI's Python environment
uses validated SDK models and returns structured job data. The code checkout and
private state directory must be mounted on compute nodes.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import secrets
import shlex
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from .base import SandboxClient

_SHARED_CLI = "/hypervolve/share/env/clusterx/bin/clusterx"
_RESULT_PREFIX = "LAGENT_CLUSTERX_RESULT="
_TERMINAL = {"failed", "stopped", "succeeded", "terminated", "deleted"}
_RESOURCE_KEYS = {"gpus_per_task", "cpus_per_task", "memory_per_task", "shared_memory"}
_PROTECTED = {"cmd", "no_env", "image", "num_nodes", "tasks_per_node", "job_name"}

# Executed only inside the existing clusterx environment, not the host interpreter.
_ADAPTER = r'''
import json, sys
request = json.load(sys.stdin)
try:
    from clusterx import CLUSTER, CLUSTER_MAPPING, CLUSTERX_CONFIG
    mapping = CLUSTER_MAPPING[CLUSTER]
    params_type = mapping["params"]
    action = request["action"]
    if action == "defaults":
        config = getattr(CLUSTERX_CONFIG, CLUSTER) or {}
        result = {"cluster": CLUSTER, "tmpdir": config.get("tmpdir"),
                  "fields": list(params_type.model_fields)}
    else:
        if action in ("run", "validate"):
            values = request["params"]
            unknown = set(values) - set(params_type.model_fields)
            if unknown:
                raise ValueError("Unsupported clusterx parameters: " + ", ".join(sorted(unknown)))
            if values.get("no_env") is not True:
                raise ValueError("Sandbox submission requires no_env=True")
            params = params_type.model_validate(values)
            if action == "validate":
                info = None
                result = {"valid": True}
            else:
                info = mapping["type"]().run(params)
        elif action == "get":
            info = mapping["type"]().get_job_info(request["job_id"], verbose=False)
        elif action == "stop":
            mapping["type"]().stop(job_id=request["job_id"])
            info = None
        else:
            raise ValueError("Unknown adapter action")
        if action != "validate":
            result = ({"job_id": info.job_id, "status": str(getattr(info.status, "value", info.status)),
                       "nodes_ip": info.nodes_ip or []}
                      if info is not None else {"stopped": request["job_id"]})
    print("LAGENT_CLUSTERX_RESULT=" + json.dumps({"ok": True, "result": result}), flush=True)
except Exception as exc:
    print("LAGENT_CLUSTERX_RESULT=" + json.dumps({"ok": False, "error": str(exc),
          "error_type": type(exc).__name__}), flush=True)
    sys.exit(1)
'''


def _positive(value, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def _private_json(path: Path, data: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


class ClusterXProvider:
    """Create one cluster job per authenticated HTTP sandbox, connecting directly to its node IP.

    Args:
        partition: Clusterx partition; omitted to use the configured default.
        image: Default image; ``create(image_tag=...)`` takes precedence.
        clusterx_path: Existing CLI, defaulting to the installed or shared CLI.
        clusterx_python: Python containing clusterx; otherwise read the CLI shebang.
        state_dir: Private shared state root, defaulting to clusterx's tmpdir.
        python_path: Compute-visible lagent checkout or extra PYTHONPATH entries.
        python_executable: Python inside the user's image.
        startup_timeout: Maximum queue plus endpoint readiness wait, in seconds.
        poll_interval: Scheduler/readiness polling interval, in seconds.
        extra_run_kwargs: Additional validated scheduler parameters.
        port: HTTP port; zero lets the compute node choose a free port.
    """

    def __init__(
        self,
        partition: str | None = None,
        image: str | None = None,
        *,
        clusterx_path: str | None = None,
        clusterx_python: str | None = None,
        state_dir: str | Path | None = None,
        python_path: str | None = None,
        python_executable: str = "python3",
        startup_timeout: float = 900,
        poll_interval: float = 5,
        extra_run_kwargs: dict | None = None,
        port: int = 0,
        server_module: str = "lagent.serving.sandbox.server",
        conda_env: str | None = None,
        conda_activate_path: str | None = None,
    ):
        self.partition = partition
        self.image = image
        self.clusterx_path = (
            clusterx_path or os.environ.get("LAGENT_CLUSTERX_BIN") or shutil.which("clusterx") or _SHARED_CLI
        )
        self.clusterx_python = clusterx_python
        state = state_dir or os.environ.get("LAGENT_CLUSTERX_STATE_DIR")
        self.state_dir = Path(state) if state else None
        self.python_path = python_path or str(Path(__file__).resolve().parents[4])
        self.python_executable = python_executable
        self.startup_timeout = _positive(startup_timeout, "startup_timeout")
        self.poll_interval = _positive(poll_interval, "poll_interval")
        self.extra_run_kwargs = dict(extra_run_kwargs or {})
        if _PROTECTED & self.extra_run_kwargs.keys():
            raise ValueError("Provider-managed scheduler parameters cannot be overridden")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in [0, 65535]")
        self.port = port
        self.server_module = server_module
        self.conda_env = conda_env
        self.conda_activate_path = conda_activate_path
        if conda_env and not conda_activate_path:
            raise ValueError("conda_activate_path is required with conda_env")
        self._defaults: dict[str | None, dict] = {}
        self._jobs: dict[str, dict] = {}
        self._submission_uncertain: str | None = None

    async def _process(
        self, argv: list[str], *, data: dict | None = None, cluster_name=None, timeout=120
    ) -> tuple[int, str]:
        env = os.environ.copy()
        if cluster_name:
            env["CLUSTERX_CONFIG_DEFAULT"] = cluster_name
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd="/tmp",
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(json.dumps(data).encode() if data is not None else None),
                timeout=timeout,
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return process.returncode, stdout.decode("utf-8", errors="replace")

    async def _rpc(self, action: str, *, cluster_name=None, **data) -> dict:
        rc, output = await self._process(
            [self.clusterx_python, "-c", _ADAPTER],
            data={"action": action, **data},
            cluster_name=cluster_name,
        )
        for line in reversed(output.splitlines()):
            if line.startswith(_RESULT_PREFIX):
                reply = json.loads(line[len(_RESULT_PREFIX) :])
                if rc != 0 or reply.get("ok") is not True:
                    raise RuntimeError(f"clusterx {action}: {reply.get('error_type')}: {reply.get('error')}")
                return reply["result"]
        raise RuntimeError(f"clusterx {action} returned no structured response (exit {rc})")

    async def _ensure_runtime(self, cluster_name=None) -> dict:
        if cluster_name in self._defaults:
            return self._defaults[cluster_name]
        config = Path(os.environ.get("CLUSTERX_CFG_PATH", str(Path.home() / ".config/clusterx.yaml")))
        if not config.is_file():
            raise RuntimeError("Missing clusterx configuration; configure it before creating a sandbox")
        binary = Path(shutil.which(self.clusterx_path) or self.clusterx_path).resolve(strict=True)
        if self.clusterx_python is None:
            with binary.open(encoding="utf-8") as stream:
                shebang = stream.readline().strip()
            if shebang.startswith("#!/") and " " not in shebang[2:]:
                self.clusterx_python = shebang[2:]
            else:
                raise ValueError("Set clusterx_python to the interpreter containing this CLI's clusterx installation")
        rc, help_text = await self._process([str(binary), "run", "--help"], cluster_name=cluster_name)
        if rc != 0:
            raise RuntimeError("clusterx run --help failed; verify the selected runtime and configuration")
        for flag in ("--no-env", "--image", "--gpus-per-task", "--cpus-per-task", "--memory-per-task"):
            if flag not in help_text:
                raise ValueError(f"Current clusterx runtime does not advertise {flag}")
        defaults = await self._rpc("defaults", cluster_name=cluster_name)
        self._defaults[cluster_name] = defaults
        return defaults

    def _build_cmd(self, env_file: Path, ready_file: Path, workspace_path: str, ttl_seconds: float) -> str:
        commands = ["set -euo pipefail", f"source {shlex.quote(str(env_file))}"]
        for path in self.python_path.split(os.pathsep):
            if path:
                commands.append("git config --global --add safe.directory " + shlex.quote(path))
        commands += [
            "export PYTHONPATH=" + shlex.quote(self.python_path) + '${PYTHONPATH:+:$PYTHONPATH}',
            "mkdir -p -- " + shlex.quote(workspace_path),
            "cd -- " + shlex.quote(workspace_path),
        ]
        if self.conda_env:
            commands.append("source " + shlex.quote(self.conda_activate_path) + " " + shlex.quote(self.conda_env))
        server = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{ttl_seconds:g}s",
            self.python_executable,
            "-m",
            self.server_module,
            "--port",
            str(self.port),
            "--backend",
            "stdlib",
            "--ready-file",
            str(ready_file),
        ]
        commands.append("exec " + shlex.join(server))
        return "bash -c " + shlex.quote("\n".join(commands))

    async def create(
        self,
        job_name: str | None = None,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        image_tag: str | None = None,
        ttl_seconds: float = 3600,
        resources: dict | None = None,
        env_vars: dict[str, str] | None = None,
        key: str | None = None,
        cluster_name: str | None = None,
        workspace_path: str | None = None,
        **run_overrides,
    ) -> tuple[SandboxClient, str]:
        if self._submission_uncertain:
            raise RuntimeError(
                f"Resolve the previous uncertain submission before retrying: {self._submission_uncertain}"
            )
        image = image_tag if image_tag is not None else self.image
        if not isinstance(image, str) or not image.strip():
            raise ValueError("A nonempty sandbox image_tag or provider image is required")
        ttl = _positive(ttl_seconds, "ttl_seconds")
        wait_seconds = _positive(timeout if timeout is not None else self.startup_timeout, "timeout")
        interval = _positive(poll_interval if poll_interval is not None else self.poll_interval, "poll_interval")
        environment = dict(env_vars or {})
        for name, value in environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not isinstance(value, str) or "\0" in value:
                raise ValueError("env_vars require valid names and string values without NUL")
        workspace_path = workspace_path or environment.get("TASK_WORKSPACE", "/workspace")
        if not isinstance(workspace_path, str) or not Path(workspace_path).is_absolute():
            raise ValueError("workspace_path must be an absolute container path")
        token = key if key is not None else secrets.token_urlsafe(32)
        if not isinstance(token, str) or not token or "\0" in token:
            raise ValueError("key must be a nonempty string without NUL")
        environment["LAGENT_SANDBOX_TOKEN"] = token
        extras = {**self.extra_run_kwargs, **run_overrides}
        if _PROTECTED & extras.keys():
            raise ValueError("Provider-managed scheduler parameters cannot be overridden")
        resource_values = dict(resources or {})
        if set(resource_values) - _RESOURCE_KEYS:
            raise ValueError(f"Unsupported resources: {sorted(set(resource_values) - _RESOURCE_KEYS)}")
        extras.update(resource_values)
        gpu = extras.setdefault("gpus_per_task", 0)
        _positive(gpu, "gpus_per_task", zero=True)
        extras.setdefault("cpus_per_task", 12 * gpu if gpu else 4)
        extras.setdefault("memory_per_task", 150 * gpu if gpu else 10)
        for name in _RESOURCE_KEYS & extras.keys():
            value = extras[name]
            _positive(value, name, zero=name == "gpus_per_task")
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer (memory units are GB)")
        defaults = await self._ensure_runtime(cluster_name)
        params = {**extras, "image": image, "num_nodes": 1, "tasks_per_node": 1, "no_env": True}
        if self.partition:
            params.setdefault("partition", self.partition)
        unsupported = set(params) - set(defaults["fields"])
        if unsupported:
            raise ValueError(f"Current clusterx backend does not support parameters: {sorted(unsupported)}")
        base = self.state_dir
        if base is None:
            if not defaults.get("tmpdir"):
                raise ValueError("Set state_dir to a private directory on shared storage")
            base = Path(defaults["tmpdir"]) / ".lagent_sandboxes"
        if base.is_symlink():
            raise ValueError("state_dir cannot be a symlink")
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = base.resolve() / uuid.uuid4().hex
        state.mkdir(mode=0o700)
        env_file, ready_file = state / "environment.sh", state / "ready.json"
        descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for name, value in environment.items():
                stream.write(f"export {name}={shlex.quote(value)}\n")
        params["job_name"] = job_name or f"lagent-sandbox-{state.name[:12]}"
        params["cmd"] = self._build_cmd(env_file, ready_file, workspace_path, ttl)
        try:
            await self._rpc("validate", cluster_name=cluster_name, params=params)
        except BaseException:
            shutil.rmtree(state)
            raise
        _private_json(
            state / "state.json", {"status": "submitting", "job_name": params["job_name"], "cluster": cluster_name}
        )
        job_id = None
        client = None
        submission = asyncio.create_task(self._rpc("run", cluster_name=cluster_name, params=params))
        cancelled = False
        try:
            try:
                info = await asyncio.shield(submission)
            except asyncio.CancelledError:
                # Wait for the bounded submission RPC to obtain an exact id for cleanup.
                cancelled = True
                info = await submission
            job_id = info.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError("clusterx submission did not return a job_id")
            self._jobs[job_id] = {"state_dir": str(state), "cluster": cluster_name, "client": None}
            _private_json(
                state / "state.json", {"status": "submitted", "job_id": job_id, "job_name": params["job_name"]}
            )
            if cancelled:
                raise asyncio.CancelledError()
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                info = await self._rpc("get", cluster_name=cluster_name, job_id=job_id)
                status = str(info.get("status", "")).lower()
                if status in _TERMINAL:
                    raise RuntimeError(f"ClusterX job {job_id} exited before readiness: {status}")
                if status == "running" and ready_file.is_file():
                    ready = json.loads(ready_file.read_text(encoding="utf-8"))
                    port = ready.get("port")
                    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                        raise ValueError("Sandbox ready file contains an invalid port")
                    for address in info.get("nodes_ip") or []:
                        try:
                            node_ip = ipaddress.ip_address(address)
                        except ValueError:
                            continue
                        host = f"[{node_ip}]" if node_ip.version == 6 else str(node_ip)
                        url = f"http://{host}:{port}"
                        if client is None or self._jobs[job_id].get("url") != url:
                            if client is not None:
                                await client.aclose()
                            client = SandboxClient(url + "?" + urlencode({"token": token}), trust_env=False)
                            self._jobs[job_id].update(url=url, node_ip=str(node_ip), client=client)
                        if (await client.health_check()).get("ok"):
                            _private_json(state / "state.json", {"status": "ready", "job_id": job_id, "url": url})
                            return client, job_id
                        break
                await asyncio.sleep(interval)
            raise TimeoutError(f"ClusterX job {job_id} did not become ready within {wait_seconds:g}s")
        except BaseException as exc:
            if job_id is None:
                self._submission_uncertain = f"job_name={params['job_name']}, state_dir={state}"
                _private_json(state / "state.json", {"status": "submission_uncertain", "job_name": params["job_name"]})
                raise RuntimeError(
                    f"ClusterX submission outcome is unknown; inspect {self._submission_uncertain}. "
                    "Private launch files were retained and automatic resubmission is blocked."
                ) from exc
            try:
                await self.delete(job_id)
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"Sandbox creation failed and stopping job {job_id} also failed: {cleanup_error}"
                ) from exc
            if client is not None:
                await client.aclose()
            raise

    async def delete(self, job_id: str) -> None:
        """Stop exactly this job; retain private state and raise on cleanup failure."""
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("A nonempty job_id is required for precise deletion")
        job = self._jobs.get(job_id, {})
        await self._ensure_runtime(job.get("cluster"))
        try:
            await self._rpc("stop", cluster_name=job.get("cluster"), job_id=job_id)
        finally:
            if job.get("client") is not None:
                await job["client"].aclose()
        if job.get("state_dir"):
            shutil.rmtree(job["state_dir"])
        self._jobs.pop(job_id, None)

    async def get(self, job_id: str) -> dict:
        cluster_name = self._jobs.get(job_id, {}).get("cluster")
        await self._ensure_runtime(cluster_name)
        return await self._rpc("get", cluster_name=cluster_name, job_id=job_id)

    def list(self) -> list[dict]:
        return [
            {"job_id": job_id, **{key: value for key, value in record.items() if key != "client"}}
            for job_id, record in self._jobs.items()
        ]

    async def aclose(self) -> None:
        failures = []
        for job_id, record in list(self._jobs.items()):
            try:
                await self.delete(job_id)
            except Exception as exc:
                failures.append(f"{job_id}: {exc}")
            if record.get("client") is not None:
                try:
                    await record["client"].aclose()
                except Exception as exc:
                    failures.append(f"{job_id} client: {exc}")
        if failures:
            raise RuntimeError("ClusterX cleanup failed: " + "; ".join(failures))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
