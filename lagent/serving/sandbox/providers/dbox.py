"""Direct DBox REST lifecycle adapter for Lagent's remote-exec client.

Instantiate one provider per Trial. Its server and main allocations share a
``sandboxGroup`` so DBox schedules them in the same cluster. IAM credentials
are read only by this trusted lifecycle adapter; they never enter a sandbox,
its environment, or its remote-exec HTTP client.
"""

from __future__ import annotations

import asyncio
import json
import ipaddress
import math
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from .base import SandboxClient
from .dbox_client import DBoxSandboxClient


class DBoxProvider:
    """Manage original task images through DBox's native REST API.

    The mounted remote-exec binary implements the existing ``SandboxClient``
    protocol. No SDK, dependency installation, task image conversion, or IAM
    token forwarding to the data plane is required. ``create`` waits for DBox
    scheduling; the caller still verifies remote-exec and task-service health.

    ``sandbox_group`` defaults to a fresh UUID, scoped to this provider. Reuse
    this instance for dependencies of one Trial, never for unrelated Trials.
    ``credential_file`` accepts a plain AK:SK or JSON with ``aksk``/``api_key``.
    """

    _AUTH_HEADER = "OPEN-SANDBOX-API-KEY"
    _ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
    _TERMINAL_STATES = frozenset({
        "error", "failed", "deleted", "deleting", "terminated", "terminating",
        "expired", "stopped", "stopping", "paused", "succeeded", "completed",
    })

    def __init__(
        self,
        api_url: str = "http://deepenv.sandbox.pjlab.org.cn",
        *,
        project: str,
        credential_file: str,
        sandbox_group: str | None = None,
        ready_timeout: float = 600.0,
        create_timeout: float = 120.0,
        delete_timeout: float = 30.0,
        get_timeout: float = 15.0,
        poll_interval: float = 2.0,
        pool_size: int = 32,
    ):
        parts = urlsplit(api_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path.rstrip("/")):
            raise ValueError("DBox api_url must be an HTTP(S) origin without credentials")
        if not isinstance(project, str) or not self._ID_PATTERN.fullmatch(project):
            raise ValueError("DBox requires an explicit project name")
        if not credential_file:
            raise ValueError("DBox requires a trusted credential file")
        for value in (ready_timeout, create_timeout, delete_timeout, get_timeout, poll_interval):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("DBox timeouts and poll_interval must be finite positive numbers")
        try:
            self.sandbox_group = str(UUID(sandbox_group)) if sandbox_group is not None else str(uuid4())
        except (ValueError, TypeError, AttributeError):
            raise ValueError("DBox sandbox_group must be a UUID") from None
        self.api_url = api_url.rstrip("/")
        self.project = project
        self.credential_file = str(credential_file)
        self.ready_timeout = float(ready_timeout)
        self._create_timeout = float(create_timeout)
        self._delete_timeout = float(delete_timeout)
        self._get_timeout = float(get_timeout)
        self._poll_interval = float(poll_interval)
        self.owned_env_ids: set[str] = set()
        self._execd_tokens: dict[str, str] = {}
        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers={"Content-Type": "application/json"},
            timeout=create_timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=pool_size, max_keepalive_connections=max(pool_size // 2, 4)),
        )

    def _auth_headers(self) -> dict[str, str]:
        try:
            raw = Path(self.credential_file).read_text().strip()
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw
            key = parsed.get("aksk") or parsed.get("api_key") if isinstance(parsed, dict) else parsed
            if (not isinstance(key, str) or key.count(":") != 1
                    or any(c.isspace() for c in key) or not all(key.split(":"))):
                raise ValueError("invalid credential")
        except Exception:
            # A JSON parse exception or filesystem exception can include input.
            raise ValueError("DBox requires a readable trusted credential file containing complete AK:SK") from None
        return {self._AUTH_HEADER: key}

    async def _request(self, method: str, path: str, *, timeout: float, payload: dict | None = None) -> httpx.Response:
        headers = self._auth_headers()
        try:
            # POST is deliberately not retried: a lost response may already
            # have allocated a sandbox and no idempotency contract is known.
            response = await self._client.request(method, path, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:
            suffix = "; allocation may be ambiguous" if method == "POST" else ""
            raise RuntimeError(f"DBox {method} request failed ({type(exc).__name__}){suffix}") from None
        finally:
            headers.clear()
        # Do not expose response bodies, request headers, or an exception chain
        # from the upstream service; errors may echo IAM credentials.
        if response.status_code >= 300 and response.status_code != 404:
            classification = ""
            if response.status_code == 429:
                try:
                    error = response.json().get("error", {})
                    code = error.get("code") if isinstance(error, dict) else None
                    if code in {"RATE_LIMITED", "QUOTA_EXCEEDED"}:
                        classification = f" ({code})"
                except (ValueError, AttributeError):
                    pass
            raise RuntimeError(f"DBox {method} request returned HTTP {response.status_code}{classification}")
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("not an object")
            return value
        except Exception:
            raise RuntimeError("DBox returned an invalid JSON object") from None

    @classmethod
    def _check_id(cls, env_id: str) -> str:
        if not isinstance(env_id, str) or not cls._ID_PATTERN.fullmatch(env_id):
            raise ValueError("DBox returned an invalid sandbox identifier")
        return env_id

    def _normalize(self, value: dict, env_id: str) -> dict:
        # DBox has published both {state: 'running'} and
        # {status: {state: 'Running'}} representations.
        state = value.get("state") or value.get("status") or "pending"
        if isinstance(state, dict):
            state = state.get("state", "pending")
        if not isinstance(state, str):
            raise RuntimeError("DBox returned an invalid sandbox state")
        state = state.lower()
        status = "ready" if state in {"running", "ready"} else state
        info = {
            "env_id": env_id,
            "status": status,
            "sbx_name": env_id,
            "url": f"{self.api_url}/v1/sandboxes/{env_id}/proxy/44772/",
            "sandbox_group": self.sandbox_group,
        }
        extensions = value.get("extensions") or {}
        if isinstance(extensions, dict):
            cluster = extensions.get("clusterID")
            if isinstance(cluster, str) and self._ID_PATTERN.fullmatch(cluster):
                info["cluster"] = cluster
        return {"ok": True, "env": info}

    def _payload(self, image_tag: str, ttl_seconds: int, fields: dict) -> dict:
        if not isinstance(image_tag, str) or not image_tag or any(c.isspace() for c in image_tag):
            raise ValueError("DBox requires a nonempty task image URI")
        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= 86400:
            raise ValueError("DBox ttl_seconds must be an integer between 60 and 86400")
        cluster = fields.pop("cluster_name", "dbox")
        if cluster != "dbox":
            raise ValueError("Direct DBox provider only supports cluster_name='dbox'")
        resources = fields.pop("resources", None) or {"cpu": 2, "memory": 4}
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory"}
                or any(type(v) is not int or v <= 0 for v in resources.values())):
            raise ValueError("DBox resources require positive integer cpu and memory (GiB)")
        if fields:
            raise ValueError("Unsupported direct DBox creation fields; inject task runtime through a trusted provisioner")
        return {
            "image": {"uri": image_tag},
            "entrypoint": ["/mnt/llm-ai-infra/tools/remote-exec-server", "--host", "127.0.0.1", "--port", "18000"],
            "timeout": ttl_seconds,
            "resourceRequests": {"cpu": str(resources["cpu"]), "memory": f"{resources['memory']}Gi"},
            "ports": [{"containerPort": 44772, "name": "execd", "protocol": "tcp", "purpose": "system service"}],
            "volumes": [{"name": "shared", "mountPath": "/mnt/llm-ai-infra/", "subPath": "/mnt/llm-ai-infra", "readOnly": True}],
            # The trusted provisioner needs namespaced NET_ADMIN for the agent
            # UID firewall. Default runc lacks it; no Docker daemon is required.
            "extensions": {"project": self.project, "sandboxGroup": self.sandbox_group,
                           "runtimeClassName": "sysbox-runc", "disableProjectDataMount": "true"},
            "networkPolicy": {"defaultAction": "deny", "egress": [], "ingress": {
                "defaultAction": "allow", "rules": [{"action": "deny", "from": "sandbox:any"}],
            }},
        }

    async def create(self, image_tag: str, ttl_seconds: int = 1800, timeout: float | None = None,
                     user: str | None = None, **kwargs) -> tuple[SandboxClient, str]:
        """Allocate once, then wait for platform readiness and return a client."""
        # XTuner's generic provider interface requires a user identifier.
        # DBox authorization comes from the trusted project/credential pair;
        # this bookkeeping field is not a sandbox environment or API override.
        payload = self._payload(image_tag, ttl_seconds, kwargs)
        token = secrets.token_urlsafe(40)
        payload["env"] = {"EXECD_ACCESS_TOKEN": token}
        response = await self._request("POST", "/v1/sandboxes", payload=payload,
                                       timeout=self._create_timeout if timeout is None else timeout)
        if response.status_code not in {200, 201, 202}:
            raise RuntimeError("DBox creation was not accepted; allocation may be ambiguous")
        value = self._json(response)
        env_id = self._check_id(value.get("id") or value.get("sandboxID"))
        self.owned_env_ids.add(env_id)
        self._execd_tokens[env_id] = token
        try:
            info = self._normalize(value, env_id)["env"]
            deadline = time.monotonic() + self.ready_timeout
            while info["status"] != "ready":
                if info["status"] in self._TERMINAL_STATES:
                    raise RuntimeError("DBox reached a terminal state before ready")
                if time.monotonic() >= deadline:
                    raise TimeoutError("DBox platform readiness timed out")
                await asyncio.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
                info = (await self.get(env_id))["env"]
            client = DBoxSandboxClient(info["url"], auth_headers=self._auth_headers, execd_token=token,
                                       sandbox_id=env_id, provider=self)
        except BaseException:
            # The Pool has not received a lease yet. Retain ownership if
            # rollback fails so its final cleanup can try this known ID again.
            cleanup = asyncio.create_task(self.delete(env_id))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                try:
                    await cleanup
                except Exception:
                    pass
            except Exception:
                pass
            raise
        return client, env_id

    async def get(self, env_id: str) -> dict:
        """Return sanitized Gateway-shaped status, compatible with provisioners."""
        self._check_id(env_id)
        response = await self._request("GET", f"/v1/sandboxes/{env_id}", timeout=self._get_timeout)
        if response.status_code == 404:
            return self._normalize({"state": "deleted"}, env_id)
        return self._normalize(self._json(response), env_id)

    async def configure_peer(self, env_id: str, peer_id: str, *, external_targets: list[str] | None = None) -> dict:
        """Compatibility entry point for a single peer."""
        return await self.configure_network(env_id, peer_ids=[peer_id], external_targets=external_targets)

    async def configure_network(self, env_id: str, *, peer_ids: list[str] | None = None,
                                external_targets: list[str] | None = None) -> dict:
        """Set and verify deny-by-default egress for any owned sandbox."""
        peers = list(dict.fromkeys(peer_ids or []))
        for value in [env_id, *peers]:
            self._check_id(value)
            if value not in self.owned_env_ids:
                raise ValueError("Network policy may reference only this provider's owned sandboxes")
        if env_id in peers:
            raise ValueError("Sandbox cannot reference itself")
        targets = list(dict.fromkeys(external_targets or []))
        for target in targets:
            if not isinstance(target, str) or not target or any(c.isspace() for c in target):
                raise ValueError("External targets must be IPv4 CIDRs or hostnames")
            if "/" in target:
                # Only exact hosts: a wider CIDR (up to 0.0.0.0/0) would bypass
                # the deny-by-default egress this policy exists to enforce.
                if ipaddress.IPv4Network(target).prefixlen != 32:
                    raise ValueError("External CIDR targets must be exact IPv4 /32 hosts")
            elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", target):
                raise ValueError("External targets must be IPv4 CIDRs or hostnames")
        exceptions = ["sandbox:" + peer for peer in peers]
        ingress = {"action": "deny", "from": "sandbox:any", "protocol": "tcp"}
        if exceptions:
            ingress["except"] = exceptions
        policy = {"defaultAction": "deny", "egress": [
            *[{"action": "allow", "target": peer} for peer in exceptions],
            *[{"action": "allow", "target": target} for target in targets],
        ], "ingress": {"defaultAction": "allow", "rules": [ingress]}}
        response = await self._request("PUT", f"/v1/sandboxes/{env_id}/networkpolicy",
                                       payload=policy, timeout=self._get_timeout)
        result = self._json(response)
        actual = result.get("policy", {})
        if (response.status_code != 200 or result.get("status") != "ok"
                or actual.get("defaultAction") != "deny" or actual.get("egress") != policy["egress"]
                or actual.get("ingress") != policy["ingress"]):
            raise RuntimeError("DBox did not confirm the requested network policy")
        return result

    async def delete(self, env_id: str) -> None:
        """Delete and confirm deleted/404; keep ownership if confirmation fails."""
        self._check_id(env_id)
        response = await self._request("DELETE", f"/v1/sandboxes/{env_id}", timeout=self._delete_timeout)
        if response.status_code != 404:
            deadline = time.monotonic() + self._delete_timeout
            while True:
                current = (await self.get(env_id))["env"]
                if current["status"] == "deleted":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("DBox deletion was not confirmed by platform status")
                await asyncio.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
        self.owned_env_ids.discard(env_id)
        self._execd_tokens.pop(env_id, None)

    async def aclose(self) -> None:
        """Close HTTP connections; owning Pool is responsible for allocations."""
        await self._client.aclose()

    async def __aenter__(self) -> "DBoxProvider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
