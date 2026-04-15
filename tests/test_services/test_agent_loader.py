"""Unit tests for AgentLoader (pyconfig only).

Run:
    pytest tests/test_agent_loader.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagent.services.agent_loader import AgentLoader, AgentSpec, _import_module_from_path


# ── Fixtures ─────────────────────────────────────────────────────────


CONFIG_STANDARD = """\
from lagent.agents.internclaw_agent import (
    AsyncEnvAgent,
    AsyncPolicyAgent,
    InternClawAgent,
)
from lagent.llms.model import AsyncAPIClient
from lagent.agents.aggregator.context import InternClawContextBuilder

name = "code-reviewer"
description = "Reviews code for quality and best practices"
background = False

llm = dict(
    type=AsyncAPIClient,
    model=dict(
        model="gpt-4o",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    ),
    sample_params=dict(temperature=0.7),
    timeout=60,
    max_retry=3,
)

agent_config = dict(
    type=InternClawAgent,
    policy_agent=dict(
        type=AsyncPolicyAgent,
        llm=llm,
        aggregator=dict(type=InternClawContextBuilder),
        name="policy",
    ),
    env_agent=dict(
        type=AsyncEnvAgent,
        actions=[],
        name="env",
    ),
    max_turn=50,
)
"""

CONFIG_WITH_BUILD = """\
from lagent.agents.internclaw_agent import (
    AsyncEnvAgent,
    AsyncPolicyAgent,
    InternClawAgent,
)
from lagent.llms.model import AsyncAPIClient
from lagent.agents.aggregator.context import InternClawContextBuilder

name = "translator"
description = "Translates text between languages"
background = True

llm = dict(
    type=AsyncAPIClient,
    model=dict(
        model="gpt-4o-mini",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    ),
    timeout=60,
)

agent_config = dict(
    type=InternClawAgent,
    policy_agent=dict(
        type=AsyncPolicyAgent,
        llm=llm,
        aggregator=dict(type=InternClawContextBuilder),
        name="policy",
    ),
    env_agent=dict(
        type=AsyncEnvAgent,
        actions=[],
        name="env",
    ),
    max_turn=10,
)

def build(config):
    from lagent.utils import create_object
    return create_object(config)
"""


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with various agent project dirs."""
    agents = tmp_path / "agents"

    # 1. Standard InternClawAgent config
    reviewer = agents / "code-reviewer"
    reviewer.mkdir(parents=True)
    (reviewer / "config.py").write_text(CONFIG_STANDARD, encoding="utf-8")

    # 2. Config with custom build()
    translator = agents / "translator"
    translator.mkdir(parents=True)
    (translator / "config.py").write_text(CONFIG_WITH_BUILD, encoding="utf-8")

    # 3. Empty dir (should be skipped)
    (agents / "empty-dir").mkdir(parents=True)

    # 4. Dir with only AGENT.md (should be skipped)
    md_only = agents / "md-only"
    md_only.mkdir(parents=True)
    (md_only / "AGENT.md").write_text(
        "---\nname: md-only\n---\nHello\n",
        encoding="utf-8",
    )

    return tmp_path


# ── AgentLoader.list() ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_discovers_pyconfig_only(workspace):
    loader = AgentLoader(workspace)
    entries = await loader.list()
    names = {e["name"] for e in entries}

    assert names == {"code-reviewer", "translator"}
    assert "empty-dir" not in names
    assert "md-only" not in names


@pytest.mark.asyncio
async def test_list_returns_path(workspace):
    loader = AgentLoader(workspace)
    entries = await loader.list()
    for entry in entries:
        assert "path" in entry
        assert Path(entry["path"]).is_dir()


# ── AgentLoader.load() ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_standard_config(workspace):
    loader = AgentLoader(workspace)
    spec = await loader.load("code-reviewer")

    assert spec is not None
    assert spec.name == "code-reviewer"
    assert spec.description == "Reviews code for quality and best practices"
    assert spec.background is False
    assert spec.build is None
    assert spec.project_dir is not None

    # agent_config structure matches InternClawAgent pattern
    cfg = spec.agent_config
    assert cfg is not None
    assert "policy_agent" in cfg
    assert "env_agent" in cfg
    assert cfg.get("max_turn") == 50
    assert cfg["policy_agent"]["name"] == "policy"
    assert cfg["env_agent"]["name"] == "env"


@pytest.mark.asyncio
async def test_load_config_with_build(workspace):
    loader = AgentLoader(workspace)
    spec = await loader.load("translator")

    assert spec is not None
    assert spec.name == "translator"
    assert spec.description == "Translates text between languages"
    assert spec.background is True
    assert callable(spec.build)

    cfg = spec.agent_config
    assert cfg is not None
    assert cfg.get("max_turn") == 10


@pytest.mark.asyncio
async def test_load_nonexistent_returns_none(workspace):
    loader = AgentLoader(workspace)
    assert await loader.load("does-not-exist") is None


@pytest.mark.asyncio
async def test_load_empty_dir_returns_none(workspace):
    loader = AgentLoader(workspace)
    assert await loader.load("empty-dir") is None


@pytest.mark.asyncio
async def test_load_md_only_returns_none(workspace):
    loader = AgentLoader(workspace)
    assert await loader.load("md-only") is None


# ── AgentLoader.load_all() ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_all(workspace):
    loader = AgentLoader(workspace)
    specs = await loader.load_all()

    assert len(specs) == 2
    assert "code-reviewer" in specs
    assert "translator" in specs


# ── AgentLoader.build_agents_summary() ──────────────────────────────


@pytest.mark.asyncio
async def test_build_agents_summary(workspace):
    loader = AgentLoader(workspace)
    summary = await loader.build_agents_summary()

    assert "<agents>" in summary
    assert "code-reviewer" in summary
    assert "translator" in summary
    assert "<background>true</background>" in summary  # translator is background


@pytest.mark.asyncio
async def test_build_agents_summary_empty(tmp_path):
    loader = AgentLoader(tmp_path)
    assert await loader.build_agents_summary() == ""


# ── AgentSpec.create() / acreate() ──────────────────────────────────
# NOTE: Real InternClawAgent instantiation requires live dependencies
# (LLM connections, workspace paths, etc.).  These tests verify the
# config structure is correct.  Full instantiation is covered by
# integration tests (test_agent_service_e2e.py).


@pytest.mark.asyncio
async def test_spec_config_has_internclaw_structure(workspace):
    """Loaded config should have the InternClawAgent nested structure."""
    loader = AgentLoader(workspace)
    spec = await loader.load("code-reviewer")
    assert spec is not None

    cfg = spec.agent_config
    assert cfg is not None

    # Top-level: InternClawAgent
    from lagent.agents.internclaw_agent import InternClawAgent
    assert cfg["type"] is InternClawAgent

    # Nested: policy_agent with LLM
    policy = cfg["policy_agent"]
    assert policy["name"] == "policy"
    assert "llm" in policy

    # Nested: env_agent with actions
    env = cfg["env_agent"]
    assert env["name"] == "env"
    assert "actions" in env

    # max_turn
    assert cfg["max_turn"] == 50


@pytest.mark.asyncio
async def test_spec_with_build_is_callable(workspace):
    loader = AgentLoader(workspace)
    spec = await loader.load("translator")
    assert spec is not None
    assert callable(spec.build)
    assert spec.agent_config["max_turn"] == 10


@pytest.mark.asyncio
async def test_spec_create_uses_build_when_provided():
    """When build is set, create() delegates to it."""
    created = []

    def mock_build(config):
        agent = type("MockAgent", (), {"name": config.get("name", "mock")})()
        created.append(agent)
        return agent

    spec = AgentSpec(
        name="test",
        agent_config={"name": "test-agent"},
        build=mock_build,
    )
    agent = spec.create()
    assert len(created) == 1
    assert agent.name == "test-agent"


@pytest.mark.asyncio
async def test_spec_acreate_uses_async_build():
    """acreate() handles async build functions."""
    async def async_build(config):
        return type("MockAgent", (), {"name": "async-built"})()

    spec = AgentSpec(
        name="test",
        agent_config={"name": "test"},
        build=async_build,
    )
    agent = await spec.acreate()
    assert agent.name == "async-built"


def test_spec_create_no_config_raises():
    spec = AgentSpec(name="broken")
    with pytest.raises(ValueError, match="has no agent_config"):
        spec.create()


@pytest.mark.asyncio
async def test_spec_acreate_no_config_raises():
    spec = AgentSpec(name="broken")
    with pytest.raises(ValueError, match="has no agent_config"):
        await spec.acreate()


# ── AgentSpec serialization ─────────────────────────────────────────


def test_spec_to_dict_from_dict_roundtrip():
    spec = AgentSpec(
        name="test",
        description="A test agent",
        background=True,
        project_dir="/tmp/agents/test",
        agent_config=dict(type="lagent.agents.Agent"),
        extra={"model": "gpt-4"},
    )
    d = spec.to_dict()
    assert d["name"] == "test"
    assert d["project_dir"] == "/tmp/agents/test"

    restored = AgentSpec.from_dict(d)
    assert restored.name == spec.name
    assert restored.description == spec.description
    assert restored.background == spec.background
    assert restored.project_dir == spec.project_dir
    assert restored.extra == spec.extra
    assert restored.build is None


# ── sys.modules cleanup ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_does_not_pollute_sys_modules(workspace):
    module_name = "_agentproject_code-reviewer_config"
    assert module_name not in sys.modules

    loader = AgentLoader(workspace)
    await loader.load("code-reviewer")

    assert module_name not in sys.modules


@pytest.mark.asyncio
async def test_load_twice_gives_fresh_spec(workspace):
    loader = AgentLoader(workspace)
    spec1 = await loader.load("code-reviewer")
    spec2 = await loader.load("code-reviewer")

    assert spec1 is not None and spec2 is not None
    assert spec1.name == spec2.name
    assert spec1 is not spec2


# ── Multiple search dirs ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_agents_dir(workspace, tmp_path):
    user_dir = tmp_path / "user_agents"
    custom = user_dir / "custom-agent"
    custom.mkdir(parents=True)
    (custom / "config.py").write_text(
        CONFIG_STANDARD.replace("code-reviewer", "custom-agent"),
        encoding="utf-8",
    )

    loader = AgentLoader(workspace, user_agents_dir=user_dir)
    entries = await loader.list()
    names = {e["name"] for e in entries}

    assert "custom-agent" in names
    assert "code-reviewer" in names


@pytest.mark.asyncio
async def test_workspace_agents_take_priority(workspace, tmp_path):
    """If same name exists in workspace and user dir, workspace wins."""
    user_dir = tmp_path / "user_agents"
    dupe = user_dir / "code-reviewer"
    dupe.mkdir(parents=True)
    (dupe / "config.py").write_text(
        CONFIG_STANDARD.replace(
            'description = "Reviews code for quality and best practices"',
            'description = "From user dir"',
        ),
        encoding="utf-8",
    )

    loader = AgentLoader(workspace, user_agents_dir=user_dir)
    spec = await loader.load("code-reviewer")

    assert spec is not None
    assert spec.description == "Reviews code for quality and best practices"


# ── Error handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_broken_config_returns_none(workspace):
    broken = workspace / "agents" / "broken"
    broken.mkdir(parents=True)
    (broken / "config.py").write_text("@@@ not python", encoding="utf-8")

    loader = AgentLoader(workspace)
    assert await loader.load("broken") is None


@pytest.mark.asyncio
async def test_load_config_missing_agent_config_returns_none(workspace):
    bad = workspace / "agents" / "no-config"
    bad.mkdir(parents=True)
    (bad / "config.py").write_text(
        "name = 'no-config'\ndescription = 'missing agent_config'\n",
        encoding="utf-8",
    )

    loader = AgentLoader(workspace)
    assert await loader.load("no-config") is None


# ── create() / acreate() fallback to create_object ──────────────────


def test_spec_create_without_build_calls_create_object():
    """create() without build delegates to create_object()."""
    from lagent.agents.agent import Agent

    spec = AgentSpec(
        name="simple",
        agent_config=dict(type=Agent, name="simple-agent"),
    )
    agent = spec.create()
    assert agent.name == "simple-agent"


@pytest.mark.asyncio
async def test_spec_acreate_without_build_calls_create_object():
    """acreate() without build delegates to create_object()."""
    from lagent.agents.agent import Agent

    spec = AgentSpec(
        name="simple",
        agent_config=dict(type=Agent, name="simple-agent"),
    )
    agent = await spec.acreate()
    assert agent.name == "simple-agent"


# ── _import_module_from_path: spec_from_file_location returns None ──


def test_import_module_from_unloadable_file(tmp_path):
    """If spec_from_file_location returns None, ImportError is raised."""
    # A file without .py extension causes spec_from_file_location to
    # return None, triggering the ImportError guard on L156.
    bad_file = tmp_path / "not_a_module"
    bad_file.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ImportError, match="Cannot load spec"):
        _import_module_from_path("_test_mod", bad_file, tmp_path)


# ── _import_module_from_path: prev_module restoration ───────────────


def test_import_restores_prev_module(tmp_path):
    """If a module with the same name already exists, it's restored after load."""
    module_name = "_test_prev_module_restore"
    sentinel = object()
    sys.modules[module_name] = sentinel

    try:
        agent_dir = tmp_path / "dummy"
        agent_dir.mkdir()
        config_file = agent_dir / "config.py"
        config_file.write_text("value = 42\n", encoding="utf-8")

        attrs = _import_module_from_path(module_name, config_file, agent_dir)
        assert attrs["value"] == 42
        # Previous module should be restored
        assert sys.modules.get(module_name) is sentinel
    finally:
        sys.modules.pop(module_name, None)


# ── list() deduplication across multiple dirs ────────────────────────


@pytest.mark.asyncio
async def test_list_deduplicates_across_dirs(workspace, tmp_path):
    """Same agent name in workspace and user dir: only listed once."""
    user_dir = tmp_path / "user_agents"
    dupe = user_dir / "code-reviewer"
    dupe.mkdir(parents=True)
    (dupe / "config.py").write_text(
        CONFIG_STANDARD.replace(
            'description = "Reviews code for quality and best practices"',
            'description = "From user dir"',
        ),
        encoding="utf-8",
    )

    loader = AgentLoader(workspace, user_agents_dir=user_dir)
    entries = await loader.list()
    names = [e["name"] for e in entries]

    # code-reviewer should appear exactly once (workspace wins)
    assert names.count("code-reviewer") == 1


# ── builtin_agents_dir ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_builtin_agents_dir(tmp_path):
    """Agents in builtin_agents_dir are discovered."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    builtin = tmp_path / "builtin"
    agent = builtin / "builtin-agent"
    agent.mkdir(parents=True)
    (agent / "config.py").write_text(
        CONFIG_STANDARD.replace("code-reviewer", "builtin-agent"),
        encoding="utf-8",
    )

    loader = AgentLoader(workspace, builtin_agents_dir=builtin)
    entries = await loader.list()
    names = {e["name"] for e in entries}

    assert "builtin-agent" in names
