"""Sandbox client and provider base definitions.

SandboxClient — unified client for interacting with any sandbox.
SandboxProvider — protocol for sandbox lifecycle management.
"""

from __future__ import annotations

import base64
import logging
import shutil
from typing import List, Protocol, Tuple, runtime_checkable

import requests

logger = logging.getLogger(__name__)


class SandboxClient:
    """Unified HTTP client for sandbox interaction.

    Every provider returns a ``SandboxClient`` pointing to a sandbox's
    HTTP API.  The API contract (``/exec``, ``/upload``, ``/download``,
    ``/health``) is the same regardless of the underlying infrastructure.

    Parameters
    ----------
    url : str
        Base URL of the sandbox HTTP API.
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        })

    def execute(self, command: str, cwd: str = "/root", timeout_sec: int = 60) -> dict:
        """Execute a bash command inside the sandbox."""
        resp = self.session.post(
            f"{self.url}/exec",
            json={"command": command, "cwd": cwd, "timeout_sec": timeout_sec},
        )
        resp.raise_for_status()
        return resp.json()

    def upload_file(self, local_path: str, remote_path: str) -> dict:
        """Upload a local file to the sandbox."""
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = self.session.post(
            f"{self.url}/upload",
            json={"target_path": remote_path, "content_b64": content_b64},
        )
        resp.raise_for_status()
        return resp.json()

    def download_file(self, remote_path: str) -> bytes:
        """Download a file from the sandbox."""
        resp = self.session.post(
            f"{self.url}/download",
            json={"source_path": remote_path},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return base64.b64decode(data["content_b64"])
        raise RuntimeError(data.get("error", "Download failed"))

    def health_check(self) -> dict:
        """Check if the sandbox is alive."""
        try:
            resp = self.session.get(f"{self.url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __repr__(self):
        return f"SandboxClient(url={self.url!r})"


@runtime_checkable
class SandboxProvider(Protocol):
    """Protocol for sandbox lifecycle management.

    Different implementations manage different infrastructure:
    k8s Gateway, ClusterX, Docker, local subprocess, etc.
    """

    def create(self, **kwargs) -> Tuple[SandboxClient, str]:
        """Create a new sandbox.

        Returns
        -------
        client : SandboxClient
            Client connected to the new sandbox.
        sandbox_id : str
            Identifier for lifecycle management (delete, status).
        """
        ...

    def delete(self, sandbox_id: str) -> None:
        """Delete a sandbox."""
        ...
