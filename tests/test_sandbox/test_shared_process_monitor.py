"""Gateway and direct clients share lifecycle and process-state monitoring."""

import asyncio
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from lagent.serving.sandbox.providers.base import SandboxClient, SandboxProcessError
from lagent.serving.sandbox.providers.gateway import GatewayProvider


class TestSharedProcessMonitor:
    @pytest.mark.asyncio
    async def test_gateway_client_can_query_independent_lifecycle(self) -> None:
        provider = GatewayProvider("http://gateway.test", user="test")
        await provider._client.aclose()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"ok": True, "env": {"env_id": "owned", "url": "http://sandbox.test"}})
            return httpx.Response(200, json={"env": {"status": "Running"}})

        provider._client = httpx.AsyncClient(base_url=provider.gateway_url, transport=httpx.MockTransport(handler))
        try:
            client, sid = await provider.create("test")
            try:
                assert sid == "owned"
                assert await client.get_lifecycle_state() == "running"
            finally:
                await client.aclose()
        finally:
            await provider.aclose()

    @pytest.mark.asyncio
    async def test_direct_client_without_provider_reports_unknown(self) -> None:
        async with SandboxClient("http://sandbox.test") as client:
            assert await client.get_lifecycle_state() is None

    @pytest.mark.asyncio
    async def test_probe_failure_is_not_process_exit(self) -> None:
        async with SandboxClient("http://sandbox.test") as client:
            client.execute = AsyncMock(
                side_effect=[
                    {"return_code": None},
                    {"return_code": 2},
                    {"return_code": 1},
                    {"return_code": 0},
                ]
            )
            for _ in range(2):
                with pytest.raises(SandboxProcessError):
                    await client.is_pid_running(123)
            assert not await client.is_pid_running(123)
            assert await client.is_pid_running(123)
            assert "/proc/123/stat" in client.execute.call_args.args[0]

    @pytest.mark.asyncio
    async def test_zombie_is_exited_on_base_client(self) -> None:
        # Run the real generated probe against a zombie before reaping it.
        import os

        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            from pathlib import Path

            for _ in range(100):
                if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    break
                await asyncio.sleep(0.01)
            async with SandboxClient("http://sandbox.test") as client:

                async def execute(command: str, **kwargs) -> dict:
                    process = subprocess.run(shlex.split(command), capture_output=True, timeout=5)
                    return {"return_code": process.returncode}

                client.execute = execute
                assert not await client.is_pid_running(pid)
        finally:
            os.waitpid(pid, 0)

    @pytest.mark.asyncio
    async def test_lifecycle_errors_remain_errors(self) -> None:
        provider = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("unavailable")))
        async with SandboxClient("http://sandbox.test", sandbox_id="owned", lifecycle_provider=provider) as client:
            with pytest.raises(RuntimeError):
                await client.get_lifecycle_state()
