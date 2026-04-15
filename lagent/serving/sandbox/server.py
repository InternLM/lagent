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
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI backend
# ---------------------------------------------------------------------------

def create_fastapi_app():
    """Create a FastAPI application."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Lagent SandboxServer")

    class ExecRequest(BaseModel):
        command: str
        cwd: str = "/root"
        timeout_sec: int = 60

    class UploadRequest(BaseModel):
        target_path: str
        content_b64: str

    class DownloadRequest(BaseModel):
        source_path: str

    @app.post("/exec")
    def execute(req: ExecRequest):
        try:
            result = subprocess.run(
                req.command, shell=True, capture_output=True, text=True,
                cwd=req.cwd, timeout=req.timeout_sec,
            )
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False, "stdout": "",
                "stderr": f"Command timed out after {req.timeout_sec} seconds",
                "return_code": 124,
            }
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "return_code": 1}

    @app.post("/upload")
    def upload(req: UploadRequest):
        try:
            target = Path(req.target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(req.content_b64))
            return {"ok": True, "target_path": req.target_path, "size": target.stat().st_size}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/download")
    def download(req: DownloadRequest):
        try:
            data = Path(req.source_path).read_bytes()
            return {"ok": True, "content_b64": base64.b64encode(data).decode("utf-8")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Stdlib backend (zero deps fallback)
# ---------------------------------------------------------------------------

def create_stdlib_server(host: str, port: int):
    """Create an http.server based server (no third-party deps)."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._respond({"ok": True})
            else:
                self._respond({"error": "Not found"}, 404)

        def do_POST(self):
            body = self._read_body()
            if body is None:
                return
            handlers = {
                "/exec": self._handle_exec,
                "/upload": self._handle_upload,
                "/download": self._handle_download,
            }
            handler = handlers.get(self.path)
            if handler:
                handler(body)
            else:
                self._respond({"error": "Not found"}, 404)

        def _handle_exec(self, body):
            command = body.get("command", "")
            cwd = body.get("cwd", "/root")
            timeout_sec = body.get("timeout_sec", 60)
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=cwd, timeout=timeout_sec,
                )
                self._respond({
                    "ok": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                })
            except subprocess.TimeoutExpired:
                self._respond({
                    "ok": False, "stdout": "",
                    "stderr": f"Command timed out after {timeout_sec}s",
                    "return_code": 124,
                })
            except Exception as e:
                self._respond({"ok": False, "stdout": "", "stderr": str(e), "return_code": 1})

        def _handle_upload(self, body):
            try:
                target = Path(body["target_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(body["content_b64"]))
                self._respond({"ok": True, "target_path": str(target), "size": target.stat().st_size})
            except Exception as e:
                self._respond({"ok": False, "error": str(e)})

        def _handle_download(self, body):
            try:
                data = Path(body["source_path"]).read_bytes()
                self._respond({"ok": True, "content_b64": base64.b64encode(data).decode("utf-8")})
            except Exception as e:
                self._respond({"ok": False, "error": str(e)})

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                return json.loads(raw) if raw else {}
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
            logger.debug("%s %s", self.address_string(), format % args)

    return HTTPServer((host, port), Handler)


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
    parser.add_argument(
        "--backend", choices=["auto", "fastapi", "stdlib"], default="auto",
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
        uvicorn.run(create_fastapi_app(), host=args.host, port=args.port)
    else:
        server = create_stdlib_server(args.host, args.port)
        logger.info("Starting SandboxServer (stdlib) on %s:%d", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    main()
