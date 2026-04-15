"""Tests for agent project discovery (config.py / AGENT.md).

Two discovery modes:
  1. config.py  → "pyconfig"  (agent_config dict)
  2. AGENT.md   → "markdown"  (YAML frontmatter)

Run:
    python tests/test_agent_project.py
"""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

# ── Ensure lagent is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagent.services.agent_loader import (
    AgentLoader,
    AgentSpec,
    _detect_kind,
)
from lagent.services.agent import (
    AgentService,
)

# ── Helpers ──────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


# ── Fixtures: create temp agent projects ─────────────────────────────

def create_test_workspace(tmp: Path) -> Path:
    """Create a workspace with agent projects."""
    agents = tmp / "agents"

    # ── 1. PyConfig agent (config.py) ─────────────────────────────
    reviewer = agents / "code-reviewer"
    reviewer.mkdir(parents=True)

    (reviewer / "config.py").write_text(
        """\
# Agent project config — discovered by AgentLoader
from lagent.agents.agent import Agent

agent_config = dict(
    type=Agent,
    name="code-reviewer",
)

name = "code-reviewer"
description = "Reviews code for quality and best practices"
system_prompt = "You are a code reviewer."
max_turns = 50
background = False
extra = {"tools": ["Read", "Grep"], "model": "gpt-4"}
""",
        encoding="utf-8",
    )

    # ── 2. PyConfig agent with build() ────────────────────────────
    translator = agents / "translator"
    translator.mkdir(parents=True)

    (translator / "config.py").write_text(
        """\
from lagent.agents.agent import Agent

agent_config = dict(
    type=Agent,
    name="translator",
)

name = "translator"
description = "Translates text between languages"
max_turns = 10

def build(config):
    \"\"\"Custom build function.\"\"\"
    from lagent.utils import create_object
    agent = create_object(config)
    return agent
""",
        encoding="utf-8",
    )

    # ── 3. Markdown agent (AGENT.md) ──────────────────────────────
    summarizer = agents / "summarizer"
    summarizer.mkdir(parents=True)

    (summarizer / "AGENT.md").write_text(
        """\
---
name: summarizer
description: Summarizes documents concisely
tools:
  - Read
  - Glob
model: gpt-4o-mini
max_turns: 30
background: false
---

You are a document summarizer. When given a document, provide a concise
summary highlighting the key points.
""",
        encoding="utf-8",
    )

    # ── 4. Empty dir (should be skipped) ──────────────────────────
    empty = agents / "empty-dir"
    empty.mkdir(parents=True)

    return tmp


# ── Test: _detect_kind ───────────────────────────────────────────────

def test_detect_kind(workspace: Path):
    print("\n── Test: _detect_kind ──")
    agents = workspace / "agents"

    check(
        "config.py → pyconfig",
        _detect_kind(agents / "code-reviewer") == "pyconfig",
    )
    check(
        "AGENT.md → markdown",
        _detect_kind(agents / "summarizer") == "markdown",
    )
    check(
        "empty dir → None",
        _detect_kind(agents / "empty-dir") is None,
    )


# ── Test: list ───────────────────────────────────────────────────────

async def test_list(workspace: Path):
    print("\n── Test: AgentLoader.list() ──")
    loader = AgentLoader(workspace)
    entries = await loader.list()

    names = {e["name"] for e in entries}
    check("discovers 3 agents", len(entries) == 3, f"got {len(entries)}: {names}")
    check("code-reviewer found", "code-reviewer" in names)
    check("translator found", "translator" in names)
    check("summarizer found", "summarizer" in names)
    check("empty-dir skipped", "empty-dir" not in names)

    # Check kinds
    kinds = {e["name"]: e["kind"] for e in entries}
    check("code-reviewer is pyconfig", kinds.get("code-reviewer") == "pyconfig")
    check("translator is pyconfig", kinds.get("translator") == "pyconfig")
    check("summarizer is markdown", kinds.get("summarizer") == "markdown")


# ── Test: load (pyconfig) ────────────────────────────────────────────

async def test_load_pyconfig(workspace: Path):
    print("\n── Test: load (pyconfig) ──")
    loader = AgentLoader(workspace)
    spec = await loader.load("code-reviewer")

    check("spec loaded", spec is not None)
    if spec is None:
        return

    check("name", spec.name == "code-reviewer", spec.name)
    check("description", "Reviews code" in spec.description, spec.description)
    check("system_prompt", "code reviewer" in spec.system_prompt, spec.system_prompt)
    check("max_turns", spec.max_turns == 50, str(spec.max_turns))
    check("agent_config is dict", isinstance(spec.agent_config, dict))
    check("agent_config has type", "type" in (spec.agent_config or {}))
    check("extra has tools", spec.extra.get("tools") == ["Read", "Grep"])
    check("extra has model", spec.extra.get("model") == "gpt-4")
    check("build is None", spec.build is None)
    check("project_dir set", spec.project_dir is not None)


# ── Test: load (pyconfig with build) ─────────────────────────────────

async def test_load_pyconfig_build(workspace: Path):
    print("\n── Test: load (pyconfig with build) ──")
    loader = AgentLoader(workspace)
    spec = await loader.load("translator")

    check("spec loaded", spec is not None)
    if spec is None:
        return

    check("name", spec.name == "translator", spec.name)
    check("description", "Translates" in spec.description, spec.description)
    check("max_turns", spec.max_turns == 10, str(spec.max_turns))
    check("build is callable", callable(spec.build))
    check("agent_config is dict", isinstance(spec.agent_config, dict))


# ── Test: load (markdown) ────────────────────────────────────────────

async def test_load_markdown(workspace: Path):
    print("\n── Test: load (markdown) ──")
    loader = AgentLoader(workspace)
    spec = await loader.load("summarizer")

    check("spec loaded", spec is not None)
    if spec is None:
        return

    check("name", spec.name == "summarizer", spec.name)
    check("description", "Summarizes" in spec.description, spec.description)
    check("system_prompt", "document summarizer" in spec.system_prompt, spec.system_prompt)
    check("max_turns", spec.max_turns == 30, str(spec.max_turns))
    check("tools in extra", spec.extra.get("tools") == ["Read", "Glob"])
    check("model in extra", spec.extra.get("model") == "gpt-4o-mini")
    check("no agent_config", spec.agent_config is None)


# ── Test: load_all ───────────────────────────────────────────────────

async def test_load_all(workspace: Path):
    print("\n── Test: load_all ──")
    loader = AgentLoader(workspace)
    specs = await loader.load_all()

    check("3 specs loaded", len(specs) == 3, f"got {len(specs)}")
    check("code-reviewer in specs", "code-reviewer" in specs)
    check("translator in specs", "translator" in specs)
    check("summarizer in specs", "summarizer" in specs)


# ── Test: build_agents_summary ───────────────────────────────────────

async def test_summary(workspace: Path):
    print("\n── Test: build_agents_summary ──")
    loader = AgentLoader(workspace)
    summary = await loader.build_agents_summary()

    check("not empty", len(summary) > 0)
    check("contains <agents>", "<agents>" in summary)
    check("contains code-reviewer", "code-reviewer" in summary)
    check("contains translator", "translator" in summary)
    check("contains summarizer", "summarizer" in summary)


# ── Test: AgentSpec.create() (pyconfig) ──────────────────────────────

async def test_spec_create_pyconfig(workspace: Path):
    print("\n── Test: AgentSpec.create() (pyconfig) ──")
    loader = AgentLoader(workspace)
    spec = await loader.load("code-reviewer")
    assert spec is not None

    agent = spec.create()
    check("agent created", agent is not None)
    check("agent has memory", hasattr(agent, "memory"))
    check("agent name", agent.name == "code-reviewer", agent.name)


# ── Test: AgentSpec.create() (pyconfig with build) ───────────────────

async def test_spec_create_build(workspace: Path):
    print("\n── Test: AgentSpec.create() (pyconfig with build) ──")
    loader = AgentLoader(workspace)
    spec = await loader.load("translator")
    assert spec is not None

    agent = spec.create()
    check("agent created", agent is not None)
    check("agent has memory", hasattr(agent, "memory"))
    check("agent name", agent.name == "translator", agent.name)


# ── Test: AgentService with mixed project types ─────────────────────

async def test_agent_service_mixed(workspace: Path):
    print("\n── Test: AgentService with mixed project types ──")
    loader = AgentLoader(workspace)
    service = AgentService(agent_loader=loader)
    await service.load_specs()

    check("3 types available", len(service.available_types) == 3,
          str(service.available_types))
    check("code-reviewer registered", "code-reviewer" in service.available_types)
    check("translator registered", "translator" in service.available_types)
    check("summarizer registered", "summarizer" in service.available_types)

    # Check that pyconfig spec has agent_config
    cr_spec = service.get_spec("code-reviewer")
    check("pyconfig spec has agent_config",
          cr_spec is not None and cr_spec.agent_config is not None)

    # Check that translator spec has build
    tr_spec = service.get_spec("translator")
    check("translator spec has build",
          tr_spec is not None and callable(tr_spec.build))


# ── Test: AgentSpec serialization roundtrip ──────────────────────────

def test_spec_serialization():
    print("\n── Test: AgentSpec serialization roundtrip ──")
    spec = AgentSpec(
        name="test",
        description="A test agent",
        project_dir="/tmp/agents/test",
        agent_config=dict(type="lagent.agents.Agent"),
        extra={"tools": ["Read"]},
    )
    d = spec.to_dict()
    check("to_dict has project_dir", d["project_dir"] == "/tmp/agents/test")

    restored = AgentSpec.from_dict(d)
    check("roundtrip name", restored.name == spec.name)
    check("roundtrip project_dir", restored.project_dir == spec.project_dir)
    check("roundtrip extra", restored.extra == spec.extra)
    check("build is None after roundtrip", restored.build is None)


# ── Test: priority (config.py wins over AGENT.md) ───────────────────

async def test_priority(workspace: Path):
    """If a project has both config.py AND AGENT.md, config.py wins."""
    print("\n── Test: config.py priority over AGENT.md ──")
    agents = workspace / "agents"
    hybrid = agents / "hybrid"
    hybrid.mkdir(parents=True, exist_ok=True)

    # Create both config.py and AGENT.md
    (hybrid / "config.py").write_text(
        """\
from lagent.agents.agent import Agent
agent_config = dict(type=Agent, name="hybrid-from-config")
name = "hybrid"
description = "From config.py"
""",
        encoding="utf-8",
    )
    (hybrid / "AGENT.md").write_text(
        """\
---
name: hybrid
description: From AGENT.md
---
System prompt from markdown.
""",
        encoding="utf-8",
    )

    kind = _detect_kind(hybrid)
    check("config.py wins", kind == "pyconfig")

    loader = AgentLoader(workspace)
    spec = await loader.load("hybrid")
    check("spec loaded", spec is not None)
    if spec:
        check("description from config.py", spec.description == "From config.py",
              spec.description)


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    tmp = Path(tempfile.mkdtemp(prefix="test_agent_project_"))
    try:
        workspace = create_test_workspace(tmp)
        print(f"Workspace: {workspace}")

        test_detect_kind(workspace)
        await test_list(workspace)
        await test_load_pyconfig(workspace)
        await test_load_pyconfig_build(workspace)
        await test_load_markdown(workspace)
        await test_load_all(workspace)
        await test_summary(workspace)
        await test_spec_create_pyconfig(workspace)
        await test_spec_create_build(workspace)
        await test_agent_service_mixed(workspace)
        test_spec_serialization()
        await test_priority(workspace)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
