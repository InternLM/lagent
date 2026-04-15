---
name: task-synthesis
description: Synthesize complete agent evaluation task packages from an OpenClaw skill. Use when given a skill (SKILL.md path, skill name, or skill content) and asked to generate a benchmark task — including task.md (with frontmatter, Prompt, Expected Behavior, Assets Preparation, Toolset Preparation, Environment Specification, Grading Criteria, Automated Checks, LLM Judge Rubric), an assets/ folder, a tools/ folder with OpenClaw-loadable tool definitions, and a scripts/ folder containing env_spec.sh and any helper scripts. Triggers on phrases like "generate a task from this skill", "synthesize a task for skill X", "create a benchmark task", "draft a task spec".
---

# Task Synthesis

## Overview

This skill produces a complete, ready-to-run evaluation task package derived from an existing OpenClaw skill. The input is a skill (SKILL.md, skill name, or a path to the skill); the output is a self-contained task directory:

| Output | Description |
|---|---|
| `task.md` | Task definition following the standard template |
| `assets/` | All input files and directories the agent needs at runtime |
| `tools/` | OpenClaw-loadable tool definitions the agent can call |
| `scripts/` | Executable scripts: `env_spec.sh` (required) + any helpers |

**`tools/` vs `scripts/` distinction:**
- `tools/` — tool definitions that OpenClaw loads and exposes to the agent as callable tools (e.g., `get_video_meta`, `query_database`). These are invoked by the agent during the task.
- `scripts/` — shell or Python scripts that run in the execution environment, not called by the agent directly. Always includes `env_spec.sh` (even if empty). May include helper scripts like `init_db.py`, `mock_server.py`, etc.

For the canonical task.md schema, field rules, and blank template — see `references/task-schema.md` and `references/task-blank-template.md`.
See `references/synthesis-patterns.md` for asset, tool, script, and grading patterns.

---

## Workflow

### Step 1 — Analyze the Skill

Read the skill's SKILL.md (and any referenced files), then establish three things:

**1. Classify family and archetype**

**Skill Family** — where does this skill primarily operate? Choose one:

`Documents & Files` / `Filesystem & Workspace` / `Code / Repo / IDE` / `Browser & Web` / `SaaS / API Integrations` / `Local Apps / OS Automation` / `Media` / `DevOps / System / Cloud` / `Search & Research` / `Agent Infra / Orchestration` / `Hybrid`

**Skill Archetype** — what is the primary action this skill performs? Choose one:

`retrieve` / `extract` / `transform` / `organize` / `analyze` / `generate` / `edit` / `execute` / `validate` / `monitor` / `automate` / `hybrid`

Record both as `family` and `archetype` — they feed into the task.md frontmatter and inform query realism and grading focus in later steps.

**2. Map capability boundary (Provides / Requires)**

| Set | Question | Examples |
|---|---|---|
| **Provides** | What tools, APIs, or capabilities does this skill itself expose? | Custom OpenClaw tools it ships, helper scripts it installs, services it starts, APIs it wraps |
| **Requires** | What must already exist for this skill to function? | CLI binaries (`ffmpeg`, `git`), language runtimes, API keys, OS packages, specific file formats |

- **Provides** → comes with the skill; the task author does not need to prepare these
- **Requires** → what the task author must prepare: install in `env_spec.sh`, synthesize in `tools/`, declare in `## Toolset Preparation`

**3. Extract baseline information**

- **Trigger scenarios**: what kinds of user requests does this skill handle?
- **Required inputs**: what must the user provide for the skill to work?
- **Optional / defaultable inputs**: what can be inferred or has a sensible default?
- **Failure modes**: what can go wrong if inputs are missing or malformed?

With these three established, you have everything needed to write a realistic query and design the execution environment and evaluation rules.

### Step 2 — Write the Query

With classification and capability boundary established, write the `## Prompt` as a real user message. Two principles govern both the prompt and its relationship to conditions in Step 3:

1. **Natural first.** Write from the user's perspective — goal-oriented, at whatever level of detail feels realistic for the task. A simple request can be one line; a complex one may include steps or constraints. What it should never read like is a task spec or evaluation brief written by the task author.

2. **Query and conditions are a pair.** Write the prompt with awareness of what conditions will (and will not) be provided. What the conditions omit shapes what the agent must do — ask, infer, or explore — and that shapes grading just as much as what the query says.

Additional rules:
- **Language: Chinese.** Write the prompt in Chinese (Simplified).
- **No absolute paths, no `assets/`、`task/`、`workspace/` prefix.** Use only filenames or workspace-relative paths (e.g., `data.csv`, `reports/summary.md`). All assets are copied into the agent's workspace at runtime; `assets/` is invisible to the agent.

### Step 3 — Prepare Environment

Define the minimal runtime dependencies and write `scripts/env_spec.sh` to fulfill them.

**Design:** Record what the runtime needs —
- OS / base image assumptions (if any)
- Language runtimes and version constraints
- Required packages (system, pip, npm, etc.)
- Environment variables (API keys, config values)
- Background services and their initial state (e.g., "SQLite DB with schema X, seeded with N rows")

**Implement:** Refer to `references/synthesis-patterns.md` (scripts/ section) for the base template; adapt to the dependencies above. Add auxiliary scripts as needed (e.g., `init_db.py`, `seed_data.py`, `start_server.py`).

**Verify (static checklist):**
- Every dependency checks for existence before installing (e.g., `command -v` for CLI tools, `python3 -c "import X"` for pip packages)
- Language packages use user-local install (`pip install --user`, local `node_modules`)
- System CLI tools (`ffmpeg`, `jq`, etc.) check existence first, then fall back to `sudo apt-get` only if missing, and fail with a clear error if sudo is unavailable
- Every environment variable has a corresponding `export` line
- Every background service has a start command and a readiness check
- Every one-time setup step (DB init, data seeding) is present and runs after its dependency is ready
- Script uses only `$WORKSPACE`, `$TASK_PATH`, `$HOME` — no hardcoded absolute paths
- Script exits non-zero on any failure (`set -e` or explicit error handling)

Do not proceed to Step 4 until verification passes.

### Step 4 — Prepare Tools

Decide which tools the agent can call, then synthesize and verify them.

For each tool, record:
- **Name**, **description**, **input schema**, **output schema**
- **Type** — one of:
  - `built-in`: already provided by OpenClaw; check `assets/built_in_tools.json` first; no plugin needed
  - `static mock`: return value is hardcoded regardless of input; asset files for this tool do not need real content
  - `input-derived mock`: return value is computed from the actual input file; asset files **must** be real (synthesized in Step 5)

> **Format fidelity:** mock return values must look like what a real implementation would return — correct field names, envelope structure, data types. A bare string where structured JSON is expected misleads the agent.

**Implement:** For each `static mock` / `input-derived mock` tool, synthesize a plugin under `tools/{tool_name}/` following the **`openclaw-plugin-creator`** skill. `built-in` tools need no plugin.

**Verify (static checklist):**
- Every `static mock` / `input-derived mock` tool has a plugin directory under `tools/{tool_name}/`

Do not proceed to Step 5 until verification passes.

### Step 5 — Prepare Assets

Decide which input files the agent needs in its workspace, then create them.

Keep assets consistent with the environment (Step 3) and tools (Step 4): exported files should match seeded DB records; files consumed by `input-derived mock` tools must be real files whose content drives those tools' return values.

Create each file at `assets/{path}`. Generate realistic, self-consistent content. Fall back to a placeholder only when real content cannot be produced and the file will be consumed exclusively by `static mock` tools.

### Step 6 — Expected Behavior & Grading

**Expected behavior:** With query (Step 2) and conditions (Steps 3–5) fixed, reason through what a correct agent execution looks like. Document:
- **Actions**: tool calls, file operations, decisions the agent should make
- **Intermediate outputs**: transient artifacts needed for subsequent steps
- **Final outputs**: the concrete deliverable — file(s) written, message sent, state changed — and expected format/content

**Grading:** Convert expected behaviors into a complete grading spec (must sum to 100%).

For any criterion whose expected value depends on synthesized assets, compute it now from the actual files (run the calculation, compute the hash, etc.).

Classify each criterion:
- *Can a script check this without ambiguity?* → `automated`
- *Requires understanding intent, quality, or semantic correctness?* → `llm-judge`

When the conditions lock down the output to a deterministic value, grade the value. When the output is non-deterministic or the task evaluates process/reasoning, grade behavior or quality instead.

**Part 1 — Automated** *(omit if none)*
- List each criterion as a bullet with percentage weight
- Provide a fully implemented `grade()` function — no `...` or `pass`; criterion key strings must match bullet text exactly

**Part 2 — LLM Judge** *(omit if none)*
- List each criterion as a bullet with percentage weight
- Provide rubrics: full-score, zero-score, and any partial-credit gradations

### Step 7 — Write task.md

With all assets, tools, environment, and grading values known, produce `task.md` following the schema in `references/task-schema.md`. Use `references/task-blank-template.md` as the starting template — fill in every section; leave no placeholders.

Inputs:
- Frontmatter values: status, difficulty, timeout, grading_mode, family, archetype
- Prompt (Step 2)
- Expected Behavior and Grading Criteria (Step 6)
- Assets Preparation, Toolset Preparation, Environment Specification (Steps 3–5)

**Do not proceed to Step 8 until `task.md` is written to disk.**

### Step 8 — Validate

**Hard gate. Do not deliver until every item passes. For each failure: fix it, then re-check before continuing.**

- [ ] `task.md` exists at the task package root
- [ ] All frontmatter fields present; no `TODO` placeholders remain
- [ ] `workspace_files` lists every asset path correctly
- [ ] Grading percentages sum to 100
- [ ] Grading Criteria match query type (output-focused vs behavior-focused)
- [ ] `grade()` is fully implemented — no `...` or `pass`
- [ ] `grade()` criterion key strings match Grading Criteria bullets exactly
- [ ] Every asset in **Assets Preparation** exists in `assets/`
- [ ] Every tool in **Toolset Preparation** exists in `tools/`
- [ ] `scripts/env_spec.sh` exists and runs without errors
- [ ] All environment dependencies in **Environment Specification** are covered by `scripts/env_spec.sh`

---

## Output Directory Layout

```
{task-id}/
├── task.md
├── assets/
│   ├── {file-or-dir-1}
│   └── {file-or-dir-2}
├── tools/
│   └── {tool_name}          ← OpenClaw-loadable tool definitions
└── scripts/
    ├── env_spec.sh          ← always present
    └── {helper_script}.py   ← optional
```

---

## Quick Reference

**Difficulty heuristics:**
- `low` — single-step, unambiguous output, < 60 s timeout
- `medium` — multi-step, some judgment or environment inspection, 60–180 s timeout
- `high` — open-ended, complex reasoning or multi-turn clarification, > 180 s timeout

**Grading mode:**
- `automated` — all criteria deterministic (file check, regex, unit test, transcript pattern match)
- `llm-judge` — all criteria semantic (quality, reasoning, clarification behavior)
- `hybrid` — mix; default for most tasks

**Grading derives from query + conditions together:**
- If conditions provide everything needed → grading is output-correctness → prefer automated
- If conditions omit something the query implies → grading is agent behavior (did it ask? inspect? propose?) → prefer llm-judge + transcript checks
- If query leaves the approach open → grading is reasoning quality and option coverage → prefer llm-judge

**Step ordering rationale:**
- Step 1: understand skill → 1.1 classify (family + archetype) → 1.2 capability boundary (Provides / Requires)
- Step 2: write prompt (natural, goal-oriented; query and conditions designed as a pair)
- Step 3: prepare environment — design deps, write `env_spec.sh`, verify
- Step 4: prepare tools — design sufficient/actual sets, synthesize plugins, verify
- Step 5: prepare assets — design sufficient/actual sets, synthesize files
- Step 6: expected behavior + grading (precompute values, write `grade()`, LLM judge rubrics)
- Step 7: write `task.md` via task-markdown-editor skill
- Step 8: validate — hard gate, fix and re-check before delivering

**Status lifecycle:** `preparing` → `ready` → `deprecated`
