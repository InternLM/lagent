"""Sandbox client and provider base definitions.

SandboxClient — async HTTP client for interacting with any sandbox.
SandboxProvider — protocol for sandbox lifecycle management.

All HTTP I/O is native async via ``httpx.AsyncClient``.  The heartbeat
background thread that earlier versions ran was removed: the gateway no
longer GCs idle sandboxes, and with async I/O keeping thousands of
concurrent coroutines alive is cheap, so callers poll ``/health``
themselves when they need readiness.
"""

from __future__ import annotations
import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Protocol, Tuple, runtime_checkable
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)


class SandboxClient:
    """Async HTTP client for a sandbox's ``/exec``, ``/upload``, ``/download``,
    ``/health`` API.

    Connection pooling is handled by ``httpx.AsyncClient``.  Use as an async
    context manager, or explicitly ``await client.aclose()`` when done.

    Args:
        url (str): Base URL of the sandbox HTTP API. May carry query params
            (e.g. ``http://host?token=...``); they are stripped off and applied
            as default query params on every request.
        timeout (float): Default request timeout in seconds. Defaults to ``60``.
        max_connections (int): Max concurrent HTTP connections to the sandbox.
            Per-sandbox concurrency is normally 1-2, so the default of 32 is
            generous. Defaults to ``32``.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        max_connections: int = 32,
    ):
        parts = urlsplit(url.rstrip("/"))
        self.url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        self._client = httpx.AsyncClient(
            base_url=self.url,
            params=dict(parse_qsl(parts.query)),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max(max_connections // 2, 4),
            ),
        )

    async def execute(
        self,
        command: str,
        cwd: str = "/root",
        timeout_sec: int = 60,
        detach: bool = False,
    ) -> dict:
        """Execute a bash command inside the sandbox.

        Args:
            command (str): Shell command to run.
            cwd (str): Working directory. Defaults to ``"/root"``.
            timeout_sec (int): Command timeout inside the sandbox. Defaults to ``60``.
            detach (bool): Whether to launch the command in background.

        Returns:
            dict: Server response with ``stdout`` / ``stderr`` / ``return_code``.
        """
        timeout_sec = int(timeout_sec)
        resp = await self._client.post(
            "/exec",
            json={"command": command, "cwd": cwd, "timeout_sec": timeout_sec, "detach": detach},
            timeout=timeout_sec + 10,
        )
        resp.raise_for_status()
        return resp.json()

    async def upload_file(self, local_path: str, remote_path: str) -> dict:
        """Upload a local file to the sandbox.

        Args:
            local_path (str): Path to a file on the caller's filesystem.
            remote_path (str): Destination path inside the sandbox.

        Returns:
            dict: Server response.
        """
        blob = Path(local_path).read_bytes()
        return await self.upload_bytes(remote_path, blob)

    async def upload_bytes(self, remote_path: str, content: bytes) -> dict:
        """Upload in-memory bytes to the sandbox.

        Args:
            remote_path (str): Destination path inside the sandbox.
            content (bytes): Raw bytes to write.

        Returns:
            dict: Server response.
        """
        content_b64 = base64.b64encode(content).decode("utf-8")
        resp = await self._client.post(
            "/upload",
            json={"target_path": remote_path, "content_b64": content_b64},
        )
        resp.raise_for_status()
        return resp.json()

    async def download_file(self, remote_path: str) -> bytes:
        """Download a file from the sandbox.

        Args:
            remote_path (str): Source path inside the sandbox.

        Returns:
            bytes: File contents.
        """
        resp = await self._client.post(
            "/download",
            json={"source_path": remote_path},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return base64.b64decode(data["content_b64"])
        raise RuntimeError(data.get("error", "Download failed"))

    async def health_check(self) -> dict:
        """Single ``/health`` ping. Returns ``{"ok": False, "error": ...}`` on
        transport failure instead of raising.

        Returns:
            dict: Server response, or ``{"ok": False, "error": <str>}`` on
                transport failure.
        """
        try:
            resp = await self._client.get("/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def is_pid_running(self, pid: int) -> bool:
        """Return whether ``pid`` is currently alive inside the sandbox."""
        pid = int(pid)
        result = await self.execute(f"kill -0 {pid} >/dev/null 2>&1", cwd="/tmp", timeout_sec=5)
        return result.get("return_code") == 0

    async def wait_pid_exit(
        self,
        pid: int,
        timeout: int | float = 600,
        interval: int | float = 1.0,
    ) -> dict:
        """Wait until ``pid`` exits inside the sandbox."""
        timeout_value = float(timeout)
        interval_value = float(interval)
        pid = int(pid)
        deadline = time.monotonic() + timeout_value
        poll = max(0.05, interval_value)
        while time.monotonic() <= deadline:
            if not await self.is_pid_running(pid):
                return {"ok": True, "pid": pid, "exited": True}
            await asyncio.sleep(poll)
        return {"ok": False, "pid": pid, "exited": False, "timeout": timeout_value}

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "SandboxClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"SandboxClient(url={self.url!r})"


@runtime_checkable
class SandboxProvider(Protocol):
    """Protocol for sandbox lifecycle management.

    Different implementations manage different infrastructure: k8s Gateway,
    ClusterX, Docker, local subprocess, etc.
    """

    async def create(self, **kwargs) -> Tuple[SandboxClient, str]:
        """Create a new sandbox.

        Returns:
            tuple[SandboxClient, str]: Client for the new sandbox and an
                identifier for lifecycle management (delete, status).
        """
        ...

    async def delete(self, sandbox_id: str) -> None:
        """Delete a sandbox."""
        ...
