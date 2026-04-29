"""k8s Gateway sandbox provider (async).

Wraps the EnvGateway HTTP API for sandbox lifecycle management.  Returns a
:class:`SandboxClient` pointing to the allocated sandbox URL as soon as the
gateway has scheduled it — readiness polling (``/health``) is the caller's
responsibility so slow boots don't tie up this provider's concurrency.

Two layers protect the gateway from burst traffic:

- A token-bucket rate limiter paces ``create`` calls to a steady
  ``GATEWAY_CREATES_PER_SEC`` (default 4/s, with a small burst).  This
  matches the empirical sandbox-spawn rate; bursting past this saturates
  the gateway and most creates fail.
- An ``asyncio.Semaphore`` caps in-flight creates as a safety net for the
  case where the gateway slows down and would otherwise let inflight
  requests pile up unboundedly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Tuple

import httpx

from lagent.serving.sandbox.providers.base import SandboxClient

logger = logging.getLogger(__name__)


_DEFAULT_CREATES_PER_SEC = float(os.environ.get("GATEWAY_CREATES_PER_SEC", "4.0"))
_DEFAULT_CREATES_BURST = int(os.environ.get("GATEWAY_CREATES_BURST", "8"))
_DEFAULT_MAX_CONCURRENT_CREATES = int(os.environ.get("GATEWAY_MAX_CONCURRENT_CREATES", "64"))
_DEFAULT_POOL_SIZE = int(os.environ.get("LAGENT_GATEWAY_POOL_SIZE", "1024"))


class _TokenBucket:
    """Async token-bucket rate limiter.

    Refills at ``rate`` tokens per second up to ``capacity``; ``acquire()``
    consumes one token, sleeping the caller until one is available.  FIFO
    among waiters: tokens are reserved under the lock before the actual
    sleep, so two concurrent acquirers don't both race past a single token.
    """

    def __init__(self, rate_per_sec: float, capacity: int):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._rate = rate_per_sec
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            # Reserve the token: charge ourselves and bump _last forward so the
            # next acquirer queues behind us instead of double-spending.
            self._tokens = 0.0
            self._last = now + wait
        await asyncio.sleep(wait)


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
        creates_per_sec (float): Steady-state ``create`` rate. Defaults to
            ``GATEWAY_CREATES_PER_SEC`` env or ``4.0``.
        creates_burst (int): Token-bucket capacity. Defaults to
            ``GATEWAY_CREATES_BURST`` env or ``8``.
        max_concurrent_creates (int): Hard cap on in-flight ``create`` calls
            (safety net for slow gateways). Defaults to
            ``GATEWAY_MAX_CONCURRENT_CREATES`` env or ``64``.
        pool_size (int): Max concurrent HTTP connections in the gateway pool.
            Defaults to ``LAGENT_GATEWAY_POOL_SIZE`` env or ``1024``.
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        creates_per_sec: float = _DEFAULT_CREATES_PER_SEC,
        creates_burst: int = _DEFAULT_CREATES_BURST,
        max_concurrent_creates: int = _DEFAULT_MAX_CONCURRENT_CREATES,
        pool_size: int = _DEFAULT_POOL_SIZE,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.gateway_url,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=max(pool_size // 4, 32),
            ),
        )
        self._create_limiter = _TokenBucket(creates_per_sec, creates_burst)
        self._create_sem = asyncio.Semaphore(max_concurrent_creates)
        logger.info(
            "GatewayProvider rate limit: %.2f creates/s (burst=%d), max concurrent=%d",
            creates_per_sec, creates_burst, max_concurrent_creates,
        )

    async def create(
        self,
        image_tag: str,
        ttl_seconds: int = 1800,
        **kwargs,
    ) -> Tuple[SandboxClient, str]:
        """Create a new sandbox environment.

        Returns as soon as the gateway has allocated a URL + env_id.  Does NOT
        wait for the sandbox to become healthy — callers (e.g. runner
        ``_acquire_ready_sandbox``) poll ``/health`` themselves.

        Args:
            image_tag (str): Docker image tag for the sandbox.
            ttl_seconds (int): Time-to-live in seconds. Defaults to ``1800``.
            **kwargs: Extra fields forwarded verbatim to the gateway.

        Returns:
            tuple[SandboxClient, str]: Client for the sandbox and its env_id.
        """
        await self._create_limiter.acquire()
        async with self._create_sem:
            resp = await self._client.post(
                "/envs",
                json={"image_tag": image_tag, "ttl_seconds": ttl_seconds, **kwargs},
                timeout=120,
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
        resp = await self._client.delete(f"/envs/{env_id}", timeout=30)
        resp.raise_for_status()
        logger.info("Deleted sandbox: env_id=%s", env_id)

    async def get(self, env_id: str) -> dict:
        """Get sandbox status.

        Args:
            env_id (str): Environment ID returned by :meth:`create`.

        Returns:
            dict: Gateway's JSON response.
        """
        resp = await self._client.get(f"/envs/{env_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "GatewayProvider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
