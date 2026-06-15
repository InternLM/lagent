"""Local sandbox provider — subprocess-based, no network.

Used for testing and local development.  Implements the same interface
as SandboxClient but executes commands via ``subprocess.run()`` and
copies files directly on the local filesystem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Tuple


class LocalClient:
    """Sandbox client that runs commands locally via subprocess.

    Implements the same interface as :class:`SandboxClient` so it can
    be used as a drop-in replacement.
    """

    def __init__(self, working_dir: str = "/tmp/lagent_sandbox"):
        self.working_dir = working_dir
        self._processes = {}
        os.makedirs(working_dir, exist_ok=True)

    def execute(
        self,
        command: str,
        cwd: str = None,
        timeout_sec: int = 60,
        detach: bool = False,
    ) -> dict:
        timeout_sec = int(timeout_sec)
        cwd = cwd or self.working_dir
        try:
            if detach:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
                self._processes[proc.pid] = proc
                return {
                    "ok": True,
                    "stdout": "",
                    "stderr": "",
                    "return_code": 0,
                    "pid": proc.pid,
                }
            # Background commands (ending with &): use Popen so the
            # background process survives after the shell exits.
            if command.rstrip().endswith("&"):
                subprocess.Popen(
                    ["bash", "-c", command],
                    cwd=cwd,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
                return {
                    "ok": True,
                    "stdout": "",
                    "stderr": "",
                    "return_code": 0,
                }
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout_sec,
            )
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec}s",
                "return_code": 124,
            }

    def upload_file(self, local_path: str, remote_path: str) -> dict:
        os.makedirs(os.path.dirname(remote_path) or ".", exist_ok=True)
        shutil.copy2(local_path, remote_path)
        return {"ok": True, "target_path": remote_path}

    def download_file(self, remote_path: str) -> bytes:
        with open(remote_path, "rb") as f:
            return f.read()

    def health_check(self) -> dict:
        return {"ok": True}

    def is_pid_running(self, pid: int) -> bool:
        pid = int(pid)
        proc = self._processes.get(pid)
        if proc is not None and proc.poll() is not None:
            return False
        result = self.execute(f"kill -0 {pid} >/dev/null 2>&1", cwd="/tmp", timeout_sec=5)
        return result.get("return_code") == 0

    def wait_pid_exit(
        self,
        pid: int,
        timeout: int | float = 600,
        interval: int | float = 1.0,
    ) -> dict:
        pid = int(pid)
        timeout_value = float(timeout)
        interval_value = float(interval)
        proc = self._processes.get(pid)
        if proc is not None:
            try:
                return_code = proc.wait(timeout=timeout_value)
                return {"ok": True, "pid": pid, "exited": True, "return_code": return_code}
            except subprocess.TimeoutExpired:
                return {"ok": False, "pid": pid, "exited": False, "timeout": timeout_value}

        deadline = time.monotonic() + timeout_value
        poll = max(0.05, interval_value)
        while time.monotonic() <= deadline:
            if not self.is_pid_running(pid):
                return {"ok": True, "pid": pid, "exited": True}
            time.sleep(poll)
        return {"ok": False, "pid": pid, "exited": False, "timeout": timeout_value}

    def close(self):
        pass

    def __repr__(self):
        return f"LocalClient(working_dir={self.working_dir!r})"


class LocalProvider:
    """Creates local sandbox environments (just a working directory).

    Usage::

        provider = LocalProvider()
        client, sandbox_id = provider.create()
        # client is a LocalClient — same interface as SandboxClient
    """

    def __init__(self, base_dir: str = "/tmp/lagent_sandboxes"):
        self.base_dir = base_dir
        self._sandboxes = {}
        self._counter = 0

    def create(self, working_dir: str = None, **kwargs) -> Tuple[LocalClient, str]:
        self._counter += 1
        sandbox_id = f"local-{self._counter}"
        working_dir = working_dir or os.path.join(self.base_dir, sandbox_id)
        client = LocalClient(working_dir=working_dir)
        self._sandboxes[sandbox_id] = client
        return client, sandbox_id

    def delete(self, sandbox_id: str) -> None:
        client = self._sandboxes.pop(sandbox_id, None)
        if client and os.path.exists(client.working_dir):
            shutil.rmtree(client.working_dir, ignore_errors=True)

    def list(self):
        return [{"sandbox_id": sid, "working_dir": c.working_dir} for sid, c in self._sandboxes.items()]
