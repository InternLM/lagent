"""k8s Gateway sandbox provider (async).

Wraps the EnvGateway HTTP API for sandbox lifecycle management.  Returns a
:class:`SandboxClient` pointing to the allocated sandbox URL as soon as the
gateway has scheduled it — readiness polling (``/health``) and create-rate
control belong to callers such as the agent loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .base import AllocationCallback, SandboxClient, SandboxEnvironment, SandboxStatus
from .credentials import read_trusted_aksk
from .lifecycle import allocated_sandbox, allocation_preflight, wait_ready

logger = logging.getLogger(__name__)


class GatewayProvider:
    """Manages sandboxes via the EnvGateway HTTP API.

    Usage::

        async with GatewayProvider("http://env-gateway.ailab.ailab.ai") as provider:
            client, env_id = await provider.create(image_tag="hb_3d-scan-calc")
            async with client:
                result = await client.execute("echo hello")
            await provider.delete(env_id)

    Args:
        gateway_url (str): Base URL of the EnvGateway service.
        pool_size (int): Max concurrent HTTP connections in the gateway pool.
            Defaults to ``1024``.
        create_timeout (float): HTTP timeout for ``create``. Defaults to ``120``.
        delete_timeout (float): HTTP timeout for ``delete``. Defaults to ``30``.
        get_timeout (float): HTTP timeout for ``get``. Defaults to ``15``.
    """

    supports_allocation_callback = True
    supports_deletion_confirmation = True

    def __init__(
        self,
        gateway_url: str,
        *,
        pool_size: int = 1024,
        create_timeout: float = 120.0,
        delete_timeout: float = 30.0,
        get_timeout: float = 15.0,
        user: str | None = None,
        project: str | None = None,
        credential_file: str | None = None,
        ready_timeout: float = 600.0,
        verify_delete: bool = False,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.user, self.project = user, project
        self.credential_file = credential_file
        self.ready_timeout = ready_timeout
        self.verify_delete = verify_delete
        # IDs are owned by this instance; credentials are never retained.
        self.owned_env_ids: set[str] = set()
        self._dbox_env_ids: set[str] = set()
        self._create_timeout = create_timeout
        self._delete_timeout = delete_timeout
        self._get_timeout = get_timeout
        self._client = httpx.AsyncClient(
            base_url=self.gateway_url,
            headers={"Content-Type": "application/json"},
            timeout=create_timeout,
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=max(pool_size // 4, 32),
            ),
        )

    async def create(
        self,
        image_tag: str,
        ttl_seconds: int = 1800,
        timeout: float | None = None,
        *,
        on_allocated: AllocationCallback | None = None,
        **kwargs: Any,
    ) -> tuple[SandboxClient, str]:
        """Create a new sandbox environment.

        Publish the ID before waiting for DBox platform readiness. Legacy
        Gateway allocation returns immediately. Callers still poll application
        ``/health`` for both transports.

        Args:
            image_tag (str): Docker image tag for the sandbox.
            ttl_seconds (int): Time-to-live in seconds. Defaults to ``1800``.
            timeout (float | None): HTTP timeout for this create request.
            **kwargs: Extra fields forwarded verbatim to the gateway.

        Returns:
            tuple[SandboxClient, str]: Client for the sandbox and its env_id.
        """
        request_timeout = self._create_timeout if timeout is None else timeout
        is_dbox = kwargs.get("cluster_name") == "dbox"
        with allocation_preflight():
            if is_dbox:
                payload = self._dbox_payload(image_tag, ttl_seconds, kwargs)
            else:
                # Preserve the legacy gateway contract for existing recipes.
                payload = {"image_tag": image_tag, "ttl_seconds": ttl_seconds, **kwargs}
        info = await self._allocate(payload, is_dbox=is_dbox, timeout=request_timeout)
        env_id = info["env_id"]
        if is_dbox:
            self._dbox_env_ids.add(env_id)
        async with allocated_sandbox(env_id, self.owned_env_ids, on_allocated, self._rollback_allocated):
            if is_dbox:
                info = await wait_ready(env_id, info, self.get, timeout=self.ready_timeout, poll_interval=2)
            url = info.get("url")
            if not url:
                raise RuntimeError("Gateway has no remote-exec URL")
            client = SandboxClient(url, sandbox_id=env_id, lifecycle_provider=self)
        logger.info("Created sandbox: env_id=%s", env_id)
        return client, env_id

    async def _allocate(self, payload: dict, *, is_dbox: bool, timeout: float) -> SandboxEnvironment:
        """Send exactly one POST and validate its allocated ID without leaking credentials."""
        try:
            resp = await self._client.post("/envs", json=payload, timeout=timeout)
            resp.raise_for_status()
            ret = resp.json()
        except Exception as exc:
            # HTTP bodies/exception chains may echo extra_params.aksk.
            raise RuntimeError(f"Gateway create failed ({type(exc).__name__}); allocation may be ambiguous") from None
        finally:
            if is_dbox:
                payload.get("extra_params", {}).pop("aksk", None)
        if not ret.get("ok"):
            raise RuntimeError("Gateway rejected sandbox creation")
        info = ret.get("env") or {}
        env_id = info.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            raise RuntimeError("Gateway creation response has no environment ID")
        return info

    async def _rollback_allocated(self, env_id: str) -> None:
        """Shield the Gateway delete; allocation ownership survives failure."""
        await asyncio.shield(self.delete(env_id))

    def _dbox_payload(self, image_tag: str, ttl_seconds: int, kwargs: dict) -> dict:
        fields = dict(kwargs)
        fields.pop("cluster_name", None)
        user = fields.pop("user", None) or self.user
        extra = dict(fields.pop("extra_params", {}))
        project = extra.get("project") or fields.pop("project", None) or self.project
        credential_file = fields.pop("credential_file", None) or self.credential_file
        key = fields.pop("key", None)
        if credential_file:
            key = read_trusted_aksk(credential_file)
        key = key or extra.get("aksk")
        if not isinstance(key, str) or key.count(":") != 1 or any(c.isspace() for c in key) or not all(key.split(":")):
            raise ValueError("DBox requires a complete AK:SK credential in its trusted credential file")
        if not user or not project:
            raise ValueError("DBox requires explicit user and project")
        if not 60 <= ttl_seconds <= 86400:
            raise ValueError("DBox ttl_seconds must be between 60 and 86400")
        resources = fields.pop("resources", None)
        if resources:
            extra["resources"] = dict(resources)
        for name, value in extra.get("resources", {}).items():
            if name not in {"cpu", "memory"} or type(value) is not int or value <= 0:
                raise ValueError("DBox resources require positive integer cpu and memory (GiB)")
        if fields.pop("env_vars", None):
            raise ValueError(
                "DBox gateway does not document env_vars; inject task runtime through a trusted provisioner"
            )
        if fields:
            raise ValueError("Unsupported DBox creation fields")
        extra.update(project=project, aksk=key)
        return dict(env=image_tag, user=user, cluster_name="dbox", ttl_seconds=ttl_seconds, extra_params=extra)

    async def delete(self, env_id: str) -> None:
        """Delete a sandbox environment.

        Args:
            env_id (str): Environment ID returned by :meth:`create`.
        """
        try:
            resp = await self._client.delete(f"/envs/{env_id}", timeout=self._delete_timeout)
            if resp.status_code != 404:
                resp.raise_for_status()
                result = resp.json() if resp.content else {}
                if result.get("ok") is False or result.get("deleted") is False:
                    raise RuntimeError("Gateway did not confirm deletion")
            if self.verify_delete or env_id in self._dbox_env_ids:
                deadline = time.monotonic() + self._delete_timeout
                while True:
                    response = await self._client.get(f"/envs/{env_id}", timeout=self._get_timeout)
                    if response.status_code == 404:
                        break
                    response.raise_for_status()
                    current = response.json()
                    if current.get("env", current).get("status") == "deleted":
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("DBox deletion was not confirmed by platform status")
                    await asyncio.sleep(1)
        except Exception as exc:
            raise RuntimeError(f"Gateway deletion failed ({type(exc).__name__})") from None
        self.owned_env_ids.discard(env_id)
        self._dbox_env_ids.discard(env_id)
        logger.info("Deleted sandbox: env_id=%s", env_id)

    async def get(self, env_id: str) -> SandboxStatus:
        """Get sandbox status.

        Args:
            env_id (str): Environment ID returned by :meth:`create`.

        Returns:
            dict: Gateway's JSON response.
        """
        try:
            resp = await self._client.get(f"/envs/{env_id}", timeout=self._get_timeout)
            if resp.status_code == 404:
                return {"ok": True, "env": {"env_id": env_id, "status": "deleted"}}
            resp.raise_for_status()
            result = resp.json()
            return {**result, "env": result.get("env", result), "ok": result.get("ok", True)}
        except Exception as exc:
            raise RuntimeError(f"Gateway status request failed ({type(exc).__name__})") from None

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "GatewayProvider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
