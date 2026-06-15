"""k8s Gateway sandbox provider (async).

Wraps the EnvGateway HTTP API for sandbox lifecycle management.  Returns a
:class:`SandboxClient` pointing to the allocated sandbox URL as soon as the
gateway has scheduled it — readiness polling (``/health``) and create-rate
control belong to callers such as the agent loop.
"""

from __future__ import annotations

import logging
from typing import Tuple

import httpx

from lagent.serving.sandbox.providers.base import SandboxClient

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

    def __init__(
        self,
        gateway_url: str,
        *,
        pool_size: int = 1024,
        create_timeout: float = 120.0,
        delete_timeout: float = 30.0,
        get_timeout: float = 15.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
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
        **kwargs,
    ) -> Tuple[SandboxClient, str]:
        """Create a new sandbox environment.

        Returns as soon as the gateway has allocated a URL + env_id.  Does NOT
        wait for the sandbox to become healthy — callers (e.g. runner
        ``_acquire_ready_sandbox``) poll ``/health`` themselves.

        Args:
            image_tag (str): Docker image tag for the sandbox.
            ttl_seconds (int): Time-to-live in seconds. Defaults to ``1800``.
            timeout (float | None): HTTP timeout for this create request.
            **kwargs: Extra fields forwarded verbatim to the gateway.

        Returns:
            tuple[SandboxClient, str]: Client for the sandbox and its env_id.
        """
        request_timeout = self._create_timeout if timeout is None else timeout
        resp = await self._client.post(
            "/envs",
            json={"image_tag": image_tag, "ttl_seconds": ttl_seconds, **kwargs},
            timeout=request_timeout,
        )
        resp.raise_for_status()
        ret = resp.json()
        if not ret.get("ok"):
            raise RuntimeError(f"Failed to create sandbox: {ret.get('error', ret)}")
        url = ret["env"]["url"]
        env_id = ret["env"]["env_id"]
        logger.info("Created sandbox: url=%s, env_id=%s", url, env_id)
        return SandboxClient(url), env_id

    async def delete(self, env_id: str) -> None:
        """Delete a sandbox environment.

        Args:
            env_id (str): Environment ID returned by :meth:`create`.
        """
        resp = await self._client.delete(f"/envs/{env_id}", timeout=self._delete_timeout)
        resp.raise_for_status()
        logger.info("Deleted sandbox: env_id=%s", env_id)

    async def get(self, env_id: str) -> dict:
        """Get sandbox status.

        Args:
            env_id (str): Environment ID returned by :meth:`create`.

        Returns:
            dict: Gateway's JSON response.
        """
        resp = await self._client.get(f"/envs/{env_id}", timeout=self._get_timeout)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "GatewayProvider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
