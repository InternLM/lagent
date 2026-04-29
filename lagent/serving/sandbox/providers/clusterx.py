"""ClusterX sandbox provider (async).

Submits a job to the ClusterX scheduler that runs a
:class:`~lagent.serving.sandbox.server.SandboxServer` inside the allocated
node.  The shared filesystem means lagent code is already accessible — no
uploading needed.

Usage::

    from lagent.serving.sandbox.providers.clusterx import ClusterXProvider

    provider = ClusterXProvider(partition="puyu-gpu")
    client, job_id = await provider.create()
    async with client:
        result = await client.execute("echo hello")
    await provider.delete(job_id)

Requires the ``clusterx`` package and a valid cluster configuration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Tuple

from .base import SandboxClient

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080


class ClusterXProvider:
    """Manages sandboxes via ClusterX job scheduler.

    Each ``create()`` submits a job that starts a SandboxServer.  The server
    is accessible via the node IP on the shared network.

    Args:
        partition (str): ClusterX partition to submit jobs to.
        image (str): Docker image for the job. Optional.
        port (int): Port for the SandboxServer. Defaults to ``8080``.
        server_module (str): Python module to run as the sandbox server.
            Defaults to ``"lagent.serving.sandbox.server"``.
        conda_env (str): Conda environment to activate before starting server.
            Optional.
        python_path (str): Extra PYTHONPATH to prepend. Optional.
        extra_run_kwargs (dict): Extra kwargs passed to the ClusterX run
            params (cpus_per_task, memory_per_task, etc.). Optional.
    """

    def __init__(
        self,
        partition: Optional[str] = None,
        image: Optional[str] = None,
        port: int = DEFAULT_PORT,
        server_module: str = "lagent.serving.sandbox.server",
        conda_env: Optional[str] = None,
        conda_activate_path: str = "/mnt/shared-storage-user/liukuikun/miniconda3/bin/activate",
        python_path: Optional[str] = None,
        extra_run_kwargs: Optional[dict] = None,
    ):
        self.port = port
        self.server_module = server_module
        self.conda_env = conda_env
        self.conda_activate_path = conda_activate_path
        self.python_path = python_path
        self.extra_run_kwargs = extra_run_kwargs or {}
        self._jobs: dict[str, dict] = {}

        from clusterx import CLUSTER, CLUSTER_MAPPING
        self._cluster_type = CLUSTER_MAPPING[CLUSTER]["type"]
        self._params_type = CLUSTER_MAPPING[CLUSTER]["params"]
        self._client = self._cluster_type()

        self._cluster_defaults = self._load_cluster_defaults(CLUSTER)
        self.partition = partition or self._cluster_defaults.get("partition", "")
        self.image = image or self._cluster_defaults.get("image")

    async def create(
        self,
        job_name: Optional[str] = None,
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs,
    ) -> Tuple[SandboxClient, str]:
        """Submit a job and wait for the SandboxServer to be ready.

        Args:
            job_name (str): Job name. Auto-generated if not provided. Optional.
            timeout (int): Max seconds to wait for the server to be ready.
                Defaults to ``300``.
            poll_interval (int): Seconds between status checks. Defaults to ``5``.
            **kwargs: Override any ClusterX run params.

        Returns:
            tuple[SandboxClient, str]: Connected sandbox client and job
                identifier for lifecycle management.
        """
        from clusterx.launcher.base import JobStatus

        run_kwargs = {
            "partition": self.partition,
            "cmd": self._build_cmd(),
            "num_nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_task": 0,
            **self.extra_run_kwargs,
            **kwargs,
        }
        if job_name:
            run_kwargs["job_name"] = job_name
        if self.image:
            run_kwargs["image"] = self.image
        if "mount" not in run_kwargs:
            mounts = self._cluster_defaults.get("mount")
            if mounts:
                run_kwargs["data_sources"] = ",".join(mounts) if isinstance(mounts, list) else mounts
            run_kwargs["image"] = self.image

        params = self._params_type.model_construct(**run_kwargs)

        status = await asyncio.to_thread(self._client.run, params)
        job_id = status.job_id
        logger.info("Submitted ClusterX job: %s", job_id)

        deadline = time.monotonic() + timeout
        node_ip = None
        while time.monotonic() < deadline:
            info = await asyncio.to_thread(self._client.get_job_info, job_id)
            if info.status == JobStatus.RUNNING:
                if info.nodes_ip:
                    node_ip = info.nodes_ip[0]
                    break
            elif info.status in (JobStatus.FAILED, JobStatus.STOPPED):
                raise RuntimeError(f"ClusterX job {job_id} failed with status: {info.status}")
            await asyncio.sleep(poll_interval)

        if not node_ip:
            await asyncio.to_thread(self._client.stop, job_id=job_id)
            raise TimeoutError(f"ClusterX job {job_id} did not start within {timeout}s")

        url = f"http://{node_ip}:{self.port}"
        logger.info("Job %s running on %s", job_id, url)

        client = SandboxClient(url)
        server_deadline = time.monotonic() + 60
        healthy = False
        while time.monotonic() < server_deadline:
            health = await client.health_check()
            if health.get("ok"):
                healthy = True
                break
            await asyncio.sleep(2)
        if not healthy:
            await client.aclose()
            await asyncio.to_thread(self._client.stop, job_id=job_id)
            raise TimeoutError(f"SandboxServer at {url} did not become healthy within 60s")

        self._jobs[job_id] = {"url": url, "node_ip": node_ip}
        logger.info("ClusterX sandbox ready: job_id=%s, url=%s", job_id, url)
        return client, job_id

    async def delete(self, job_id: str) -> None:
        """Stop a ClusterX job and release resources.

        Args:
            job_id (str): Job identifier returned by :meth:`create`.
        """
        try:
            await asyncio.to_thread(self._client.stop, job_id=job_id)
            logger.info("Stopped ClusterX job: %s", job_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
        self._jobs.pop(job_id, None)

    async def get(self, job_id: str) -> dict:
        """Get job status.

        Args:
            job_id (str): Job identifier returned by :meth:`create`.

        Returns:
            dict: Job status dict with ``job_id``, ``status``, ``nodes_ip``.
        """
        info = await asyncio.to_thread(self._client.get_job_info, job_id)
        return {
            "job_id": info.job_id,
            "status": str(info.status),
            "nodes_ip": info.nodes_ip,
        }

    def list(self) -> list[dict]:
        """List tracked sandbox jobs.

        Returns:
            list[dict]: One entry per job, with ``job_id`` plus url / node_ip.
        """
        return [{"job_id": jid, **info} for jid, info in self._jobs.items()]

    async def aclose(self) -> None:
        """No-op for API symmetry; ClusterX client has no async resources to release."""
        return None

    async def __aenter__(self) -> "ClusterXProvider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    # -- private --

    @staticmethod
    def _load_cluster_defaults(cluster_name: str) -> dict:
        import os
        try:
            import yaml
            config_path = os.path.expanduser("~/.config/clusterx.yaml")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                return config.get(cluster_name, {}) or {}
        except Exception:
            pass
        return {}

    def _build_cmd(self) -> str:
        parts = []
        if self.conda_env:
            parts.append(f"source {self.conda_activate_path} {self.conda_env} &&")
        import lagent.serving.sandbox.server as _mod
        server_path = _mod.__file__
        parts.append(f"python {server_path} --port {self.port} --backend stdlib")
        return " ".join(parts)
