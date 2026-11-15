"""Authenticated execd transport to the loopback remote-exec protocol.

The original remote-exec server remains responsible for commands, detached
processes and file serialization. It listens only on loopback. Neither IAM nor
execd credentials are forwarded to commands, files or task runtime environments.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
import time

import httpx

from .base import SandboxClient


class DBoxTransportError(httpx.HTTPError, RuntimeError):
    """Sanitized data-plane failure compatible with existing PID polling retries."""

    def __init__(self, message, *, status_code=None, error_code=None, phase="execd_transport",
                 transport_error_type=None, elapsed_seconds=None, response_received=False):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.phase = phase
        self.transport_error_type = transport_error_type
        self.elapsed_seconds = elapsed_seconds
        self.response_received = response_received


async def _safe_error_code(response):
    """Classify a bounded error body without retaining free-form server text."""
    data = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=4096):
        data.extend(chunk[:4096 - len(data)])
        if len(data) >= 4096:
            break
    known = {"SANDBOX_NOT_RUNNING", "SANDBOX_NOT_FOUND", "INVALID_STATE", "RATE_LIMITED",
             "QUOTA_EXCEEDED", "UNAUTHORIZED", "FORBIDDEN", "PROXY_POOL_EXHAUSTED",
             "BACKEND_CONNECTION_FAILED", "BACKEND_TIMEOUT", "BACKEND_RESPONSE_FAILED",
             "SERVICE_UNAVAILABLE", "PORT_NOT_FOUND"}
    try:
        body = json.loads(data)
        error = body.get("detail") or body.get("error") or body
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and code.rsplit("::", 1)[-1] in known:
            return code.rsplit("::", 1)[-1]
    except (ValueError, AttributeError):
        pass
    if bytes(data).strip().lower().rstrip(b".") in {
        b"sandbox is not running", b"sandbox not running", b"sandbox is not ready",
    }:
        return "SANDBOX_NOT_RUNNING"
    return None


class DBoxSandboxClient(SandboxClient):
    """Preserve SandboxClient operations through an authenticated execd call."""

    process_probe_deadline_sec = 30.0

    def __init__(self, url: str, *, auth_headers, execd_token: str, sandbox_id: str, provider):
        super().__init__(url, sandbox_id=sandbox_id, lifecycle_provider=provider)
        self._auth_headers = auth_headers
        self._execd_token = execd_token
        self.sandbox_id = sandbox_id
        self.provider = provider

    async def _remote_request(self, path: str, payload: dict | None = None, timeout: float = 60) -> dict:
        encoded = base64.b64encode(json.dumps(payload).encode()).decode() if payload is not None else ""
        code = (
            "import base64,json,urllib.request; "
            "payload=base64.b64decode(" + repr(encoded) + ") if " + repr(encoded) + " else None; "
            "request=urllib.request.Request(" + repr("http://127.0.0.1:18000/" + path)
            + ",data=payload,headers={'Content-Type':'application/json'}); "
            "response=urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout="
            + repr(timeout) + "); "
            "print('DBOX_RESPONSE:'+base64.b64encode(response.read()).decode())"
        )
        text = await self._execd_stdout("python3 -c " + shlex.quote(code), timeout + 15)
        if not text.startswith("DBOX_RESPONSE:"):
            raise DBoxTransportError("DBox execution response incomplete or failed", status_code=200,
                                     phase="execd_response_stream", response_received=True)
        try:
            result = json.loads(base64.b64decode(text[len("DBOX_RESPONSE:"):], validate=True))
            if not isinstance(result, dict):
                raise ValueError("not a response object")
            return result
        except Exception:
            raise RuntimeError("DBox execution returned invalid response data") from None

    async def _run_process_probe(self, command: str) -> int | None:
        # Probe through execd directly: the loopback remote-exec server also
        # serves the agent's tool commands, so a busy queue must not delay or
        # fail process monitoring. Report the probe's status on stdout because
        # execd's own completion is the printf below.
        try:
            async with asyncio.timeout(self.process_probe_deadline_sec):
                text = await self._execd_stdout(command + "; printf 'DBOX_PID:%s\\n' \"$?\"", 20)
        except TimeoutError:
            raise DBoxTransportError("DBox process probe deadline exceeded", phase="process_probe",
                                     transport_error_type="TimeoutError") from None
        marker, _, value = text.partition(":")
        return int(value) if marker == "DBOX_PID" and value in {"0", "1", "2"} else None

    async def _execd_stdout(self, command: str, timeout: float) -> str:
        """Read one execd command stream without exposing commands or credentials."""
        headers = self._auth_headers()
        headers["X-EXECD-ACCESS-TOKEN"] = self._execd_token
        stdout = []
        complete = False
        failed = False
        started = time.monotonic()
        response_status = None
        phase = "execd_response_headers"
        def failure(message, **details):
            return DBoxTransportError(message, status_code=response_status,
                elapsed_seconds=round(time.monotonic() - started, 6),
                response_received=response_status is not None, **details)
        try:
            async with self._client.stream(
                "POST", "command", headers=headers, json={"command": command}, timeout=timeout,
            ) as response:
                response_status = response.status_code
                if response.status_code != 200:
                    try:
                        code = await asyncio.wait_for(_safe_error_code(response), timeout=2.0)
                    except (httpx.HTTPError, TimeoutError):
                        # An incomplete diagnostic body must not erase the HTTP
                        # status already observed from the control endpoint.
                        code = None
                    raise failure(
                        f"DBox execd returned HTTP {response.status_code}" + (f" ({code})" if code else ""),
                        error_code=code, phase="execd_proxy",
                    )
                phase = "execd_response_stream"
                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].lstrip()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "stdout":
                        stdout.append(event.get("text", ""))
                    elif event.get("type") in {"stderr", "error", "execution_error"}:
                        failed = True
                    elif event.get("type") == "execution_complete":
                        complete = True
                        # This event terminates the execution protocol. Waiting
                        # for HTTP EOF can time out after a completed command and
                        # holds a proxy pool connection for no useful work.
                        break
        except DBoxTransportError:
            raise
        except Exception as exc:
            raise failure(f"DBox authenticated transport failed ({type(exc).__name__})",
                          phase=phase, transport_error_type=type(exc).__name__) from None
        finally:
            headers.clear()
        if failed or not complete:
            raise failure("DBox execution response incomplete or failed", phase="execd_response_stream")
        return "".join(stdout).strip()

    async def execute(self, command: str, cwd: str = "/root", timeout_sec: int = 60, detach: bool = False) -> dict:
        return await self._remote_request(
            "exec", {"command": command, "cwd": cwd, "timeout_sec": int(timeout_sec), "detach": detach},
            timeout=int(timeout_sec) + 10,
        )

    def _process_probe_error(self) -> Exception:
        return DBoxTransportError("DBox process probe returned no valid process state", phase="process_probe")

    async def upload_bytes(self, remote_path: str, content: bytes) -> dict:
        # execd invokes a shell: keep each command well below ARG_MAX. Assemble
        # chunks in a private staging path and rename only after all succeed.
        if len(content) <= 24000:
            return await self._remote_request(
                "upload", {"target_path": remote_path, "content_b64": base64.b64encode(content).decode()},
            )
        from uuid import uuid4
        staging = "/tmp/dbox-transfer-" + uuid4().hex
        try:
            await self.execute("umask 077; : > " + shlex.quote(staging), cwd="/tmp")
            for offset in range(0, len(content), 24000):
                chunk = base64.b64encode(content[offset:offset + 24000]).decode()
                code = "import base64;open(" + repr(staging) + ",'ab').write(base64.b64decode(" + repr(chunk) + "))"
                result = await self.execute("python3 -c " + shlex.quote(code), cwd="/tmp")
                if result.get("return_code") != 0:
                    raise RuntimeError("DBox chunk upload failed")
            result = await self.execute("mv -- " + shlex.quote(staging) + " " + shlex.quote(remote_path), cwd="/tmp")
            if result.get("return_code") != 0:
                raise RuntimeError("DBox upload commit failed")
            return {"ok": True}
        finally:
            await self.execute("rm -f -- " + shlex.quote(staging), cwd="/tmp")

    async def download_file(self, remote_path: str) -> bytes:
        result = await self._remote_request("download", {"source_path": remote_path})
        if result.get("ok") is not True:
            raise RuntimeError("DBox file download failed")
        return base64.b64decode(result["content_b64"], validate=True)

    async def health_check(self) -> dict:
        try:
            return await self._remote_request("health", timeout=5)
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__}
