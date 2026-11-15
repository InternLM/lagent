"""Resolve, pin and authorize sandbox HTTP endpoints from one address snapshot.

Call configure before starting a workload. Tasks declare endpoints/peers and
optionally reject addresses; they do not implement DNS or hosts-file updates.
IPv4 is explicit because the sandbox platform adapter currently supports /32.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import shlex
from typing import Callable
from urllib.parse import urlsplit


def http_endpoint(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Expected an HTTP(S) URL without credentials")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", host):
        raise ValueError("Expected an IPv4 address or DNS hostname")
    return host, parsed.port or (443 if parsed.scheme == "https" else 80)


def ipv4_egress_targets(
    host: str,
    addresses: list[str] | None = None,
    address_validator: Callable[[str], str] | None = None,
) -> list[str]:
    """Return a bootstrap hostname or validated exact-IP platform targets."""
    if addresses is None:
        try:
            addresses = [str(ipaddress.IPv4Address(host))]
        except ipaddress.AddressValueError:
            return [host]
    if not addresses:
        raise ValueError("Resolved endpoint must have IPv4 addresses")
    canonical = sorted({str(ipaddress.IPv4Address(ip)) for ip in addresses})
    if address_validator is not None:
        for ip in canonical:
            address_validator(ip)
    return [ip + "/32" for ip in canonical]


class SandboxNetwork:
    """Shared network setup for a client with an owned lifecycle provider.

    urls identifies HTTP endpoints; peer_ids identifies other owned sandboxes.
    address_validator may reject any resolved IPv4 before any policy/hosts edit.
    Setup is bounded and does not retry mutation or workload execution.
    """

    def __init__(
        self,
        *,
        urls: list[str] | None = None,
        peer_ids: list[str] | None = None,
        python: str = "python3",
        address_validator: Callable[[str], str] | None = None,
        timeout: float = 60,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Network setup timeout must be finite and positive")
        self.endpoints = dict(http_endpoint(url) for url in (urls or []))
        self.peer_ids = list(peer_ids or [])
        self.python, self.address_validator, self.timeout = (
            python,
            address_validator,
            timeout,
        )

    async def resolve(self, client) -> dict[str, list[str]]:
        """Resolve inside the sandbox, without changing its hosts or policy."""
        if not self.endpoints:
            return {}
        code = (
            "import json,socket; endpoints=" + repr(self.endpoints) + "; "
            "print(json.dumps({h:sorted({r[4][0] for r in socket.getaddrinfo(h,p,"
            "socket.AF_INET,socket.SOCK_STREAM)}) for h,p in endpoints.items()}))"
        )
        result = await asyncio.wait_for(
            client.execute(
                shlex.join([self.python, "-c", code]),
                cwd="/tmp",
                timeout_sec=self.timeout,
            ),
            timeout=self.timeout,
        )
        if result.get("return_code") != 0:
            raise RuntimeError("Sandbox endpoint resolution failed")
        return self._validate(json.loads(result["stdout"]))

    async def configure(
        self,
        client,
        *,
        bootstrap: bool = False,
        resolved: dict[str, list[str]] | None = None,
    ) -> dict:
        """Apply and verify exact-IP egress and matching host bindings.

        bootstrap=True only authorizes names during trusted setup. Final setup
        resolves once unless supplied an earlier snapshot; it never re-resolves
        between installing the platform policy and pinning connection addresses.
        """

        async def configure() -> dict:
            provider = getattr(client, "provider", None)
            if provider is None or not hasattr(provider, "configure_network"):
                raise RuntimeError(
                    "Sandbox provider does not support network configuration"
                )
            if bootstrap:
                if resolved is not None:
                    raise ValueError("Bootstrap must not accept a resolved snapshot")
                targets = []
                for host in self.endpoints:
                    targets.extend(
                        ipv4_egress_targets(
                            host, address_validator=self.address_validator
                        )
                    )
                addresses = None
            else:
                addresses = (
                    await self.resolve(client)
                    if resolved is None
                    else self._validate(resolved)
                )
                targets = sorted(
                    {ip + "/32" for values in addresses.values() for ip in values}
                )
            policy = await provider.configure_network(
                client.sandbox_id, peer_ids=self.peer_ids, external_targets=targets
            )
            if addresses:
                await self._pin(client, addresses)
            return {"policy": policy, "addresses": addresses, "bootstrap": bootstrap}

        return await asyncio.wait_for(configure(), timeout=self.timeout)

    def _validate(self, value: dict) -> dict[str, list[str]]:
        if not isinstance(value, dict) or set(value) != set(self.endpoints):
            raise ValueError(
                "Resolved endpoint snapshot does not match configured hosts"
            )
        result = {}
        for host, ips in value.items():
            if not isinstance(ips, list) or not ips:
                raise ValueError("Resolved endpoint must have IPv4 addresses")
            canonical = sorted({str(ipaddress.IPv4Address(ip)) for ip in ips})
            try:
                literal = str(ipaddress.IPv4Address(host))
            except ipaddress.AddressValueError:
                literal = None
            if literal is not None and canonical != [literal]:
                raise ValueError("A literal endpoint cannot be remapped")
            if self.address_validator is not None:
                for ip in canonical:
                    self.address_validator(ip)
            result[host] = canonical
        return result

    async def _pin(self, client, addresses: dict[str, list[str]]) -> None:
        # Rewrite only configured aliases; leave unrelated aliases and comments.
        # /etc/hosts is commonly a bind mount, so replace its content in place.
        code = """import json,socket
from pathlib import Path
bindings = json.loads(%r)
p = Path('/etc/hosts')
lines = []
for line in p.read_text().splitlines():
    body, sep, comment = line.partition('#')
    parts = body.split()
    if len(parts) >= 2:
        aliases = [h for h in parts[1:] if h.lower() not in bindings]
        if aliases:
            lines.append(' '.join([parts[0], *aliases]) + ((' #' + comment) if sep else ''))
        elif sep:
            lines.append('#' + comment)
    else:
        lines.append(line)
for host, ips in bindings.items():
    lines.extend(ip + ' ' + host for ip in ips)
p.write_text('\\n'.join(lines) + '\\n')
for host, ips in bindings.items():
    actual = {r[4][0] for r in socket.getaddrinfo(host,None,socket.AF_INET,socket.SOCK_STREAM)}
    if actual != set(ips):
        raise RuntimeError('Pinned endpoint resolution differs from egress addresses')
""" % json.dumps(addresses)
        result = await client.execute(
            shlex.join([self.python, "-c", code]), cwd="/tmp", timeout_sec=self.timeout
        )
        if result.get("return_code") != 0:
            raise RuntimeError(
                "Sandbox endpoint pinning or resolution verification failed"
            )
