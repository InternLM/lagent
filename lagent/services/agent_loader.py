"""AgentLoader -- discover and load agent projects.

An agent project is a directory containing a ``config.py`` that
exports ``agent_config`` (a dict for ``create_object()``).

Optionally, a project may also export:

* ``build``   -- custom factory ``(config_dict) -> Agent``.
                 When absent the default ``create_object(agent_config)`` is used.
* ``name``    -- agent type name (defaults to directory name).
* ``description`` -- one-line description.
* ``background``  -- whether to run async (default False).

Usage::

    loader = AgentLoader(Path("workspace"))
    spec = await loader.load("my-agent")
    agent = spec.create()          # uses build or create_object

.. note::

    TODO: Support AGENT.md (markdown + YAML frontmatter) as a simplified
    format for declaring sub-agents.  This requires an ActionRegistry or
    similar mechanism to resolve tool names to Action instances, which is
    not yet implemented.  For now, only pyconfig (config.py) is supported.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from lagent.agents.agent import Agent
from lagent.utils import create_object

logger = logging.getLogger("lagent.interclaw.agent_loader")

BUILTIN_AGENTS_DIR = Path(__file__).parent.parent / "agents_builtin"


# ── AgentSpec ─────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    """Everything needed to instantiate an agent.

    The only truly essential field is ``agent_config``.  Everything else
    is metadata or optional override.
    """

    name: str = ""
    description: str = ""
    background: bool = False

    # Core: the PyConfig dict for create_object().
    agent_config: dict[str, Any] | None = None

    # Optional custom build function: (config_dict) -> Agent.
    # When set, create() calls build(agent_config) instead of create_object().
    build: Callable | None = None

    # Where this spec was loaded from (informational).
    project_dir: str | None = None

    # Pass-through metadata (AGENT.md fields like tools, model, etc.).
    extra: dict[str, Any] = field(default_factory=dict)

    def create(self) -> Agent:
        """Instantiate the agent from this spec.

        Uses ``build(agent_config)`` if a custom build function is set,
        otherwise falls back to ``create_object(agent_config)``.
        """
        if self.agent_config is None:
            raise ValueError(
                f"AgentSpec {self.name!r} has no agent_config"
            )
        if self.build is not None and callable(self.build):
            return self.build(self.agent_config)
        return create_object(self.agent_config)

    async def acreate(self, **kwargs) -> Agent:
        """Async version of :meth:`create`.

        Handles both sync and async build functions.
        Extra *kwargs* (e.g. ``llm=``, ``actions=``) are forwarded
        to the build function if it accepts them.
        """
        import inspect

        if self.agent_config is None:
            raise ValueError(
                f"AgentSpec {self.name!r} has no agent_config"
            )
        if self.build is not None and callable(self.build):
            result = self.build(self.agent_config, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        return create_object(self.agent_config)

    def to_dict(self) -> dict:
        """Serialize to plain dict (build is excluded)."""
        d = {
            "name": self.name,
            "description": self.description,
            "background": self.background,
        }
        if self.project_dir is not None:
            d["project_dir"] = self.project_dir
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict) -> AgentSpec:
        """Deserialize from plain dict."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            background=data.get("background", False),
            agent_config=data.get("agent_config"),
            project_dir=data.get("project_dir"),
            extra=data.get("extra", {}),
        )


# ── Module loader ─────────────────────────────────────────────────────

def _import_module_from_path(
    module_name: str,
    file_path: Path,
    package_dir: Path,
) -> dict[str, Any]:
    """Import a Python module and return its public attributes.

    The module is temporarily registered in ``sys.modules`` during
    execution (required for relative imports inside config.py) and
    removed afterwards to avoid polluting the module namespace.
    """
    parent = str(package_dir.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    prev_module = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, str(file_path),
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
    finally:
        # Restore or clean up sys.modules
        if prev_module is not None:
            sys.modules[module_name] = prev_module
        else:
            sys.modules.pop(module_name, None)
        if added and parent in sys.path:
            sys.path.remove(parent)


def _spec_from_module(
    attrs: dict[str, Any], name: str, path: str,
) -> AgentSpec:
    """Build AgentSpec from a config.py module's exported attributes."""
    agent_config = attrs.get("agent_config")
    if agent_config is None:
        raise ValueError(f"Agent {name!r}: config.py must export 'agent_config'")

    return AgentSpec(
        name=attrs.get("name", name),
        description=attrs.get("description", ""),
        background=bool(attrs.get("background", False)),
        agent_config=agent_config,
        build=attrs.get("build"),  # None if not defined
        project_dir=path,
        extra=attrs.get("extra", {}),
    )



# ── AgentLoader ───────────────────────────────────────────────────────


class AgentLoader:
    """Discover and load agent projects from the filesystem.

    Scans ``workspace/agents/`` (and optional user/builtin dirs) for
    agent project directories containing a ``config.py``.

    Parameters
    ----------
    workspace : Path
        Workspace root. Agents are at ``workspace/agents/<name>/``.
    user_agents_dir : Path, optional
        User-level agents directory.
    builtin_agents_dir : Path, optional
        Built-in agents (defaults to ``lagent/agents_builtin/``).
    """

    def __init__(
        self,
        workspace: Path,
        user_agents_dir: Path | None = None,
        builtin_agents_dir: Path | None = None,
    ):
        self.workspace = workspace
        self._dirs = [
            d for d in [
                workspace / "agents",
                user_agents_dir,
                builtin_agents_dir or BUILTIN_AGENTS_DIR,
            ] if d is not None
        ]

    async def list(self) -> list[dict[str, str]]:
        """List available agent projects: [{name, path}, ...]."""
        agents = []
        seen: set[str] = set()
        for directory in self._dirs:
            if not directory.exists():
                continue
            for agent_dir in sorted(directory.iterdir()):
                if not agent_dir.is_dir() or agent_dir.name in seen:
                    continue
                if (agent_dir / "config.py").exists():
                    agents.append({
                        "name": agent_dir.name,
                        "path": str(agent_dir),
                    })
                    seen.add(agent_dir.name)
        return agents

    async def load(self, name: str) -> AgentSpec | None:
        """Load a single agent by name.

        Returns AgentSpec or None if not found.
        """
        for directory in self._dirs:
            if directory is None or not directory.exists():
                continue
            agent_dir = directory / name
            if not agent_dir.is_dir():
                continue

            config_file = agent_dir / "config.py"
            if not config_file.exists():
                continue

            try:
                attrs = await asyncio.to_thread(
                    _import_module_from_path,
                    f"_agentproject_{name}_config",
                    config_file,
                    agent_dir,
                )
                return _spec_from_module(attrs, name, str(agent_dir))
            except Exception:
                logger.exception("Failed to load agent %r", name)
                return None

        return None

    async def load_all(self) -> dict[str, AgentSpec]:
        """Load all available agents. Returns name -> AgentSpec mapping."""
        entries = await self.list()
        specs = {}
        for entry in entries:
            spec = await self.load(entry["name"])
            if spec is not None:
                specs[spec.name] = spec
        return specs

    async def build_agents_summary(self) -> str:
        """Build XML summary of available agents for prompt injection."""
        specs = await self.load_all()
        if not specs:
            return ""

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<agents>"]
        for spec in specs.values():
            lines.append(f'  <agent name="{esc(spec.name)}">')
            if spec.description:
                lines.append(f"    <description>{esc(spec.description)}</description>")
            if spec.background:
                lines.append("    <background>true</background>")
            lines.append("  </agent>")
        lines.append("</agents>")
        return "\n".join(lines)
