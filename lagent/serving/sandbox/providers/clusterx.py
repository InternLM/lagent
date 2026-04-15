"""ClusterX sandbox provider.

Submits a job to the ClusterX scheduler that runs a
:class:`~lagent.serving.sandbox.server.SandboxServer` inside the
allocated node.  The shared filesystem means lagent code is already
accessible — no uploading needed.

Usage::

    from lagent.serving.sandbox.providers.clusterx import ClusterXProvider

    provider = ClusterXProvider(partition="puyu-gpu")
    client, job_id = provider.create()
    # client is a SandboxClient pointing to http://<node_ip>:8080
    result = client.execute("echo hello")
    provider.delete(job_id)

Requires the ``clusterx`` package and a valid cluster configuration.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from .base import SandboxClient

logger = logging.getLogger(__name__)

# Default port for SandboxServer inside the job
DEFAULT_PORT = 8080


class ClusterXProvider:
    """Manages sandboxes via ClusterX job scheduler.

    Each ``create()`` submits a job that starts a SandboxServer.
    The server is accessible via the node IP on the shared network.

    Parameters
    ----------
    partition : str
        ClusterX partition to submit jobs to.
    image : str, optional
        Docker image for the job.
    port : int
        Port for the SandboxServer (default 8080).
    server_module : str
        Python module to run as the sandbox server.
    conda_env : str, optional
        Conda environment to activate before starting server.
    python_path : str, optional
        Extra PYTHONPATH to prepend.
    extra_run_kwargs : dict, optional
        Extra kwargs passed to the ClusterX run params (cpus_per_task,
        memory_per_task, etc.).
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

        # Lazy import clusterx
        from clusterx import CLUSTER, CLUSTER_MAPPING
        self._cluster_type = CLUSTER_MAPPING[CLUSTER]['type']
        self._params_type = CLUSTER_MAPPING[CLUSTER]['params']
        self._client = self._cluster_type()

        # Read defaults from clusterx config (~/.config/clusterx.yaml)
        self._cluster_defaults = self._load_cluster_defaults(CLUSTER)
        self.partition = partition or self._cluster_defaults.get('partition', '')
        self.image = image or self._cluster_defaults.get('image')

    @staticmethod
    def _load_cluster_defaults(cluster_name: str) -> dict:
        """Read defaults from ~/.config/clusterx.yaml."""
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
        """Build the shell command that starts SandboxServer.

        Uses direct file execution (``python /path/to/server.py``)
        instead of ``python -m`` to avoid triggering full lagent package
        imports which may fail if optional dependencies are missing.
        """
        parts = []
        if self.conda_env:
            parts.append(f"source {self.conda_activate_path} {self.conda_env} &&")
        # Resolve server.py path relative to this file
        import lagent.serving.sandbox.server as _mod
        server_path = _mod.__file__
        # Use stdlib backend to avoid fastapi/pydantic version issues
        parts.append(f"python {server_path} --port {self.port} --backend stdlib")
        return " ".join(parts)

    def create(
        self,
        job_name: Optional[str] = None,
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs,
    ) -> Tuple[SandboxClient, str]:
        """Submit a job and wait for the SandboxServer to be ready.

        Parameters
        ----------
        job_name : str, optional
            Job name. Auto-generated if not provided.
        timeout : int
            Max seconds to wait for the server to be ready.
        poll_interval : int
            Seconds between status checks.
        **kwargs
            Override any ClusterX run params.

        Returns
        -------
        client : SandboxClient
            Connected to the SandboxServer.
        job_id : str
            Job identifier for lifecycle management.
        """
        from clusterx.launcher.base import JobStatus

        # Build run params — merge config defaults + provider settings + overrides
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
        # Apply mount from clusterx config if not explicitly set
        if "mount" not in run_kwargs:
            mounts = self._cluster_defaults.get("mount")
            if mounts:
                run_kwargs["data_sources"] = ",".join(mounts) if isinstance(mounts, list) else mounts
            run_kwargs["image"] = self.image

        params = self._params_type.model_construct(**run_kwargs)

        # Submit job
        status = self._client.run(params)
        job_id = status.job_id
        logger.info("Submitted ClusterX job: %s", job_id)

        # Wait for job to be RUNNING and get node IP
        deadline = time.monotonic() + timeout
        node_ip = None
        while time.monotonic() < deadline:
            info = self._client.get_job_info(job_id)
            if info.status == JobStatus.RUNNING:
                if info.nodes_ip:
                    node_ip = info.nodes_ip[0]
                    break
            elif info.status in (JobStatus.FAILED, JobStatus.STOPPED):
                raise RuntimeError(
                    f"ClusterX job {job_id} failed with status: {info.status}"
                )
            time.sleep(poll_interval)

        if not node_ip:
            self._client.stop(job_id=job_id)
            raise TimeoutError(
                f"ClusterX job {job_id} did not start within {timeout}s"
            )

        url = f"http://{node_ip}:{self.port}"
        logger.info("Job %s running on %s", job_id, url)

        # Wait for SandboxServer health check
        client = SandboxClient(url)
        server_deadline = time.monotonic() + 60  # extra 60s for server startup
        while time.monotonic() < server_deadline:
            health = client.health_check()
            if health.get("ok"):
                break
            time.sleep(2)
        else:
            self._client.stop(job_id=job_id)
            raise TimeoutError(
                f"SandboxServer at {url} did not become healthy within 60s"
            )

        self._jobs[job_id] = {"url": url, "node_ip": node_ip}
        logger.info("ClusterX sandbox ready: job_id=%s, url=%s", job_id, url)
        return client, job_id

    def delete(self, job_id: str) -> None:
        """Stop a ClusterX job and release resources."""
        try:
            self._client.stop(job_id=job_id)
            logger.info("Stopped ClusterX job: %s", job_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
        self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> dict:
        """Get job status."""
        info = self._client.get_job_info(job_id)
        return {
            "job_id": info.job_id,
            "status": str(info.status),
            "nodes_ip": info.nodes_ip,
        }

    def list(self):
        """List tracked sandbox jobs."""
        return [
            {"job_id": jid, **info}
            for jid, info in self._jobs.items()
        ]
