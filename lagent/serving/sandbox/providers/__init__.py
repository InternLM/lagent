"""Sandbox providers — create, manage, and connect to sandbox environments."""

from .base import AllocationCallback, SandboxClient, SandboxEnvironment, SandboxProvider, SandboxStatus
from .dbox import DBoxProvider
from .gateway import GatewayProvider
from .local import LocalClient, LocalProvider

__all__ = [
    "SandboxClient",
    "SandboxProvider",
    "AllocationCallback",
    "SandboxEnvironment",
    "SandboxStatus",
    "DBoxProvider",
    "GatewayProvider",
    "LocalProvider",
    "LocalClient",
    "ClusterXProvider",
]


def __getattr__(name):
    # Lazy import ClusterXProvider — clusterx is an optional dependency
    if name == "ClusterXProvider":
        from .clusterx import ClusterXProvider

        return ClusterXProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
