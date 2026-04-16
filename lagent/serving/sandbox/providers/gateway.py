"""k8s Gateway sandbox provider.

Wraps the EnvGateway HTTP API for sandbox lifecycle management.
The Gateway already provides sandboxes with an HTTP API (``/exec``,
``/upload``, ``/download``), so we just return a standard
:class:`SandboxClient` pointing to the sandbox URL.
"""

from __future__ import annotations

import logging
from typing import Tuple

import requests

from lagent.serving.sandbox.providers.base import SandboxClient

logger = logging.getLogger(__name__)


class GatewayProvider:
    """Manages sandboxes via the EnvGateway HTTP API.

    Usage::

        provider = GatewayProvider("http://env-gateway.ailab.ailab.ai")
        client, env_id = provider.create(image_tag="hb_3d-scan-calc")
        # client is a SandboxClient pointing to the sandbox URL
        result = client.execute("echo hello")
        provider.delete(env_id)

    Parameters
    ----------
    gateway_url : str
        Base URL of the EnvGateway service.
    """

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def create(
        self,
        image_tag: str,
        ttl_seconds: int = 1800,
        **kwargs,
    ) -> Tuple[SandboxClient, str]:
        """Create a new sandbox environment.

        Parameters
        ----------
        image_tag : str
            Docker image tag for the sandbox.
        ttl_seconds : int
            Time-to-live in seconds (default 30 min).

        Returns
        -------
        client : SandboxClient
            Client connected to the sandbox.
        env_id : str
            Environment ID for lifecycle management.
        """
        resp = self.session.post(
            f"{self.gateway_url}/envs",
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
        client =  SandboxClient(url)
        for _ in range(300):
            health_json = client.health_check()
            if health_json['ok']:
                return client, env_id
            import time 
            time.sleep(2)
        raise Exception
        

    def delete(self, env_id: str) -> None:
        """Delete a sandbox environment."""
        resp = self.session.delete(
            f"{self.gateway_url}/envs/{env_id}",
            timeout=30,
        )
        resp.raise_for_status()
        logger.info("Deleted sandbox: env_id=%s", env_id)

    def get(self, env_id: str) -> dict:
        """Get sandbox status."""
        resp = self.session.get(
            f"{self.gateway_url}/envs/{env_id}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
