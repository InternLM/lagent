"""Public endpoint pinning works without importing any task recipe."""

import asyncio
import json
from pathlib import Path
import shlex
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lagent.serving.sandbox.network import SandboxNetwork


class TestSandboxNetwork:
    @pytest.mark.asyncio
    async def test_pin_replaces_stale_aliases_and_policy_uses_same_snapshot(
        self, tmp_path: Path
    ) -> None:
        hosts = tmp_path / "hosts"
        hosts.write_text("127.0.0.1 localhost\n10.1.1.1 model.test other.test # keep\n")
        events = []
        provider = SimpleNamespace(
            configure_network=AsyncMock(return_value={"status": "ok"})
        )

        def resolve(host, port, *args):
            addresses = [
                line.split()[0]
                for line in hosts.read_text().splitlines()
                if host in line.partition("#")[0].split()[1:]
            ]
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in addresses
            ]

        async def execute(command, **kwargs):
            code = shlex.split(command)[-1]
            events.append(code)
            if "endpoints=" in code:
                return {
                    "return_code": 0,
                    "stdout": json.dumps({"model.test": ["10.2.3.4"]}),
                }
            code = code.replace("Path('/etc/hosts')", "Path(" + repr(str(hosts)) + ")")
            with patch("socket.getaddrinfo", side_effect=resolve):
                exec(compile(code, "<pin-hosts>", "exec"), {})
            return {"return_code": 0}

        client = SimpleNamespace(
            provider=provider, sandbox_id="source", execute=execute
        )
        network = SandboxNetwork(urls=["http://model.test/v1"], peer_ids=["peer"])
        result = await network.configure(client)
        assert result["addresses"] == {"model.test": ["10.2.3.4"]}
        provider.configure_network.assert_awaited_once_with(
            "source", peer_ids=["peer"], external_targets=["10.2.3.4/32"]
        )
        assert "10.1.1.1 other.test # keep" in hosts.read_text()
        assert "127.0.0.1 localhost" in hosts.read_text()
        await network.configure(client, resolved=result["addresses"])
        assert hosts.read_text().count("10.2.3.4 model.test") == 1
        assert len([code for code in events if "endpoints=" in code]) == 1

    @pytest.mark.asyncio
    async def test_rejected_addresses_do_not_change_policy_or_hosts(self) -> None:
        client = SimpleNamespace(
            provider=SimpleNamespace(configure_network=AsyncMock()),
            sandbox_id="source",
            execute=AsyncMock(),
        )

        def reject(ip):
            raise ValueError("denied by task")

        network = SandboxNetwork(urls=["http://model.test"], address_validator=reject)
        with pytest.raises(ValueError):
            await network.configure(client, resolved={"model.test": ["8.8.8.8"]})
        client.execute.assert_not_awaited()
        client.provider.configure_network.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_timeout_and_empty_peer_only_configuration(self) -> None:
        provider = SimpleNamespace(
            configure_network=AsyncMock(return_value={"status": "ok"})
        )
        client = SimpleNamespace(
            provider=provider, sandbox_id="source", execute=AsyncMock()
        )
        await SandboxNetwork(peer_ids=["peer"]).configure(client)
        client.execute.assert_not_awaited()

        async def stalled(*args, **kwargs):
            await asyncio.sleep(10)

        client.execute = stalled
        with pytest.raises(asyncio.TimeoutError):
            await SandboxNetwork(urls=["http://model.test"], timeout=0.01).configure(
                client
            )

    @pytest.mark.asyncio
    async def test_snapshot_cannot_remap_literal_endpoint(self) -> None:
        with pytest.raises(ValueError, match="literal"):
            await SandboxNetwork(urls=["http://10.1.2.3"]).configure(
                SimpleNamespace(
                    provider=SimpleNamespace(configure_network=AsyncMock())
                ),
                resolved={"10.1.2.3": ["10.2.3.4"]},
            )
