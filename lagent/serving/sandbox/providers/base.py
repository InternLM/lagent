"""Sandbox client and provider base definitions.

SandboxClient — unified client for interacting with any sandbox.
SandboxProvider — protocol for sandbox lifecycle management.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List, Protocol, Tuple, runtime_checkable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Heartbeat log — dedicated file, never propagates to the root logger
# so 400 clients × 30s pings don't flood the application's stderr.
# Path can be overridden with ``LAGENT_SANDBOX_HEARTBEAT_LOG`` env.
# ─────────────────────────────────────────────────────────────────

_HEARTBEAT_LOG_PATH = os.environ.get(
    "LAGENT_SANDBOX_HEARTBEAT_LOG",
    "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/xtuner/work_dir/sandbox_heartbeat.log",
)

_hb_logger_lock = threading.Lock()
_hb_logger: logging.Logger | None = None


def _heartbeat_logger() -> logging.Logger:
    """Return a module-private file logger for heartbeat events.

    Uses ``propagate=False`` so records land only in the file, never on
    the root logger / stderr.  Lazy, thread-safe, idempotent.
    """
    global _hb_logger
    if _hb_logger is not None:
        return _hb_logger
    with _hb_logger_lock:
        if _hb_logger is not None:
            return _hb_logger
        lg = logging.getLogger("lagent.sandbox.heartbeat")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not lg.handlers:
            try:
                Path(_HEARTBEAT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(_HEARTBEAT_LOG_PATH, encoding="utf-8")
                fh.setFormatter(logging.Formatter(
                    "%(asctime)s %(levelname)s %(threadName)s %(message)s"
                ))
                lg.addHandler(fh)
            except Exception:
                # Fall back to a null handler if the file dir is unwritable;
                # better to silently drop heartbeat logs than crash the app.
                lg.addHandler(logging.NullHandler())
        _hb_logger = lg
        return lg


class SandboxClient:
    """Unified HTTP client for sandbox interaction.

    Every provider returns a ``SandboxClient`` pointing to a sandbox's
    HTTP API.  The API contract (``/exec``, ``/upload``, ``/download``,
    ``/health``) is the same regardless of the underlying infrastructure.

    A background heartbeat thread pings ``/health`` on a fixed interval —
    some gateways GC idle sandboxes, and periodic pings both keep them
    alive and detect death early.  Heartbeat events (both ok and fail)
    are written to a dedicated log file (``LAGENT_SANDBOX_HEARTBEAT_LOG``)
    to keep the application's stderr clean.  Pass ``heartbeat_interval=0``
    (or ``None``) to disable.

    Parameters
    ----------
    url : str
        Base URL of the sandbox HTTP API.
    heartbeat_interval : float, optional
        Seconds between background ``/health`` pings.  Defaults to ``30``.
        ``0`` or ``None`` disables the heartbeat.
    """

    _DEFAULT_HEARTBEAT_INTERVAL_SEC: float = 30.0

    def __init__(
        self,
        url: str,
        heartbeat_interval: float | None = _DEFAULT_HEARTBEAT_INTERVAL_SEC,
    ):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        })
        # Intentionally no HTTPAdapter/Retry customization: per-sandbox
        # concurrency is 1-2 (main task serial + heartbeat), default
        # ``requests.Session`` with keep-alive is plenty.  Retries at this
        # layer burn ephemeral ports (each retry opens a fresh socket);
        # transient 5xx / dropped connections surface to the caller and
        # get handled by the runner's task-level retry / categorization.

        self._hb_interval: float = 0.0
        self._hb_stop: threading.Event | None = None
        self._hb_thread: threading.Thread | None = None
        self._hb_last_ok: float = 0.0
        self._hb_last_err: str | None = None
        self._hb_tick: int = 0
        if heartbeat_interval:
            self.start_heartbeat(heartbeat_interval)

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
        """Single synchronous ping to ``/health``."""
        try:
            resp = self.session.get(f"{self.url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def start_heartbeat(self, interval_sec: float = 30.0) -> None:
        """Start a background thread pinging ``/health`` every ``interval_sec``.

        Idempotent: re-calling with a new interval restarts the thread.
        Failures are logged as warnings; ``last_heartbeat_ok()`` and
        ``last_heartbeat_error()`` expose the current state for callers
        that want to react to a dead sandbox.
        """
        self.stop_heartbeat()
        self._hb_interval = float(interval_sec)
        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"sandbox-heartbeat-{self.url.rsplit('/', 1)[-1]}",
            daemon=True,
        )
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        """Signal the heartbeat thread to exit and wait briefly for it."""
        if self._hb_stop is not None:
            self._hb_stop.set()
        if self._hb_thread is not None and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=2.0)
        self._hb_stop = None
        self._hb_thread = None

    def last_heartbeat_ok(self) -> float:
        """Monotonic timestamp of the last successful heartbeat (0 if never)."""
        return self._hb_last_ok

    def last_heartbeat_error(self) -> str | None:
        """String describing the most recent heartbeat failure, if any."""
        return self._hb_last_err

    def close(self):
        """Stop the heartbeat and close the HTTP session."""
        self.stop_heartbeat()
        self.session.close()

    def __repr__(self):
        return f"SandboxClient(url={self.url!r})"

    # -- private --

    def _heartbeat_loop(self) -> None:
        assert self._hb_stop is not None
        hb_log = _heartbeat_logger()
        # First ping fires right away so callers see a fresh ok-at promptly.
        while not self._hb_stop.is_set():
            self._hb_tick += 1
            t0 = time.monotonic()
            try:
                resp = self.session.get(f"{self.url}/health", timeout=5)
                latency_ms = (time.monotonic() - t0) * 1000
                if resp.ok and (resp.json() or {}).get("ok"):
                    self._hb_last_ok = time.monotonic()
                    self._hb_last_err = None
                    hb_log.info(
                        "OK  tick=%d latency=%.0fms url=%s",
                        self._hb_tick, latency_ms, self.url,
                    )
                else:
                    self._hb_last_err = (
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    hb_log.warning(
                        "BAD tick=%d latency=%.0fms url=%s err=%s",
                        self._hb_tick, latency_ms, self.url, self._hb_last_err,
                    )
            except Exception as exc:
                self._hb_last_err = f"{type(exc).__name__}: {exc}"
                hb_log.warning(
                    "ERR tick=%d latency=%.0fms url=%s err=%s",
                    self._hb_tick, (time.monotonic() - t0) * 1000,
                    self.url, self._hb_last_err,
                )
            # ``wait`` returns True on stop, so it cleanly exits without sleeping.
            if self._hb_stop.wait(self._hb_interval):
                return


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
