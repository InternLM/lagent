"""SandboxServer — lightweight HTTP API that turns any machine into a sandbox.

Provides ``/exec``, ``/upload``, ``/download``, ``/health`` endpoints.
Supports two backends: FastAPI (if available) or stdlib http.server (fallback).

Usage::

    # Start server (auto-detects backend)
    python -m lagent.serving.sandbox.server --port 8080

    # Force stdlib backend (zero deps)
    python -m lagent.serving.sandbox.server --port 8080 --backend stdlib

    # Or run the file directly (no package imports needed)
    python /path/to/lagent/serving/sandbox/server.py --port 8080

Set ``LAGENT_SANDBOX_TOKEN`` to require a matching ``?token=...`` query
parameter or ``Authorization: Bearer ...`` header on every endpoint.
"""

import argparse
import base64
import hmac
import json
import logging
import math
import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger(__name__)


def _authorized(path, authorization, token):
    if not token:
        return True
    candidates = parse_qs(urlsplit(path).query).get("token", [])
    scheme, _, value = (authorization or "").partition(" ")
    if scheme.lower() == "bearer":
        candidates.append(value)
    return any(hmac.compare_digest(candidate.encode(), token.encode()) for candidate in candidates)


def _signal_process_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _execute_command(command, cwd="/root", timeout_sec=60, detach=False):
    """Run Bash, keeping detached processes separate from HTTP request lifetime."""
    try:
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_sec must be finite and positive")
        if not isinstance(detach, bool):
            raise ValueError("detach must be a boolean")
        process = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if detach else subprocess.PIPE,
            stderr=subprocess.DEVNULL if detach else subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if detach:
            # Reap the direct child without holding a request thread or output pipe.
            threading.Thread(target=process.wait, daemon=True).start()
            return {"ok": True, "stdout": "", "stderr": "", "return_code": 0, "pid": process.pid}
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Also kill descendants which ignored TERM and closed their output pipes.
                _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return {
                "ok": False,
                "stdout": stdout,
                "stderr": stderr + f"\nCommand timed out after {timeout_sec}s",
                "return_code": 124,
            }
        return {
            "ok": process.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": process.returncode,
        }
    except Exception as error:
        return {"ok": False, "stdout": "", "stderr": str(error), "return_code": 1}


def _upload_file(target_path, content_b64):
    try:
        content = base64.b64decode(content_b64, validate=True)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"ok": True, "target_path": str(target), "size": len(content)}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def _download_file(source_path):
    try:
        data = Path(source_path).read_bytes()
        return {"ok": True, "content_b64": base64.b64encode(data).decode("utf-8")}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def _write_ready_file(path, host, port):
    """Atomically publish the bound address using a private file."""
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sandbox-ready-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"host": host, "port": port}, stream)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ---------------------------------------------------------------------------
# FastAPI backend
# ---------------------------------------------------------------------------


def create_fastapi_app():
    """Create a FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Lagent SandboxServer")
    token = os.environ.get("LAGENT_SANDBOX_TOKEN", "")

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if not _authorized(str(request.url), request.headers.get("authorization"), token):
            return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    class ExecRequest(BaseModel):
        command: str
        cwd: str = "/root"
        timeout_sec: float = 60
        detach: bool = False

    class UploadRequest(BaseModel):
        target_path: str
        content_b64: str

    class DownloadRequest(BaseModel):
        source_path: str

    @app.post("/exec")
    def execute(req: ExecRequest):
        return _execute_command(req.command, req.cwd, req.timeout_sec, req.detach)

    @app.post("/upload")
    def upload(req: UploadRequest):
        return _upload_file(req.target_path, req.content_b64)

    @app.post("/download")
    def download(req: DownloadRequest):
        return _download_file(req.source_path)

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Stdlib backend (zero deps fallback)
# ---------------------------------------------------------------------------


def create_stdlib_server(host: str, port: int):
    """Create an http.server based server (no third-party deps)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    token = os.environ.get("LAGENT_SANDBOX_TOKEN", "")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not self._authenticate():
                return
            if urlsplit(self.path).path == "/health":
                self._respond({"ok": True})
            else:
                self._respond({"error": "Not found"}, 404)

        def do_POST(self):
            if not self._authenticate():
                return
            body = self._read_body()
            if body is None:
                return
            handlers = {
                "/exec": self._handle_exec,
                "/upload": self._handle_upload,
                "/download": self._handle_download,
            }
            handler = handlers.get(urlsplit(self.path).path)
            if handler:
                handler(body)
            else:
                self._respond({"error": "Not found"}, 404)

        def _handle_exec(self, body):
            command = body.get("command", "")
            cwd = body.get("cwd", "/root")
            timeout_sec = body.get("timeout_sec", 60)
            self._respond(_execute_command(command, cwd, timeout_sec, body.get("detach", False)))

        def _handle_upload(self, body):
            self._respond(_upload_file(body.get("target_path"), body.get("content_b64")))

        def _handle_download(self, body):
            self._respond(_download_file(body.get("source_path")))

        def _authenticate(self):
            if _authorized(self.path, self.headers.get("Authorization"), token):
                return True
            self._respond({"ok": False, "error": "Unauthorized"}, 401)
            return False

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                return body
            except Exception as e:
                self._respond({"error": f"Bad request: {e}"}, 400)
                return None

        def _respond(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            logger.debug("%s %s %s", self.address_string(), self.command, urlsplit(self.path).path)

    return ThreadingHTTPServer((host, port), Handler)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="lagent.serving.sandbox.server",
        description="SandboxServer: HTTP API for sandbox interaction",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ready-file", help="Write the bound host/port as JSON (mode 0600); supports --port 0.")
    parser.add_argument(
        "--backend",
        choices=["auto", "fastapi", "stdlib"],
        default="auto",
        help="Server backend: fastapi (uvicorn), stdlib (http.server), or auto-detect",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    backend = args.backend
    if backend == "auto":
        try:
            import fastapi, uvicorn  # noqa: F401

            backend = "fastapi"
        except ImportError:
            backend = "stdlib"

    if backend == "fastapi":
        import uvicorn

        logger.info("Starting SandboxServer (fastapi) on %s:%d", args.host, args.port)
        # Query tokens must not be emitted in uvicorn access logs.
        config = uvicorn.Config(create_fastapi_app(), host=args.host, port=args.port, access_log=False)
        sock = config.bind_socket()
        try:
            _write_ready_file(args.ready_file, *sock.getsockname()[:2])
            uvicorn.Server(config).run(sockets=[sock])
        finally:
            sock.close()
    else:
        server = create_stdlib_server(args.host, args.port)
        _write_ready_file(args.ready_file, *server.server_address[:2])
        logger.info("Starting SandboxServer (stdlib) on %s:%d", *server.server_address[:2])
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
