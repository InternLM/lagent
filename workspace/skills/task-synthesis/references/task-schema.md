# task.md Schema Reference

Complete field-by-field reference for the task.md format.

## Frontmatter Fields

```yaml
---
id: {task-id}                        # Provided in prompt — use as-is; do NOT modify
name: {task-name}                    # GENERATED: short, hyphen-linked, content-focused (see naming rules)
scene: {scene}                       # Provided in prompt — use as-is
category: {category}                 # Provided in prompt — use as-is
family: {family}                     # GENERATED: see allowed values below
archetype: {archetype}               # GENERATED: see allowed values below
skill_set:
  - {skill-id}                       # Provided in prompt — use as-is
grading_mode: automated|llm-judge|hybrid   # Determined during synthesis
timeout_seconds: 180                 # Adjust: low=60, medium=180, high=300+
workspace_files:                     # List of ALL files provided to the agent
  - assets/{filename-or-dir}
difficulty: low|medium|high
status: preparing|ready|deprecated
---
```

### Field Rules

**Fields provided in the prompt — use as-is, do NOT alter:**
- `id`, `scene`, `category`, `skill_set`

**Fields always generated during task synthesis:**

#### `name`
- Short, descriptive, hyphen-linked slug
- 3–6 words max; omit filler words ("the", "a", "task")
- Must reflect the concrete action and subject
- Examples: `extract-invoice-totals`, `rename-files-by-date`, `summarize-pr-diff`, `query-sqlite-schema`

#### `family`
Must be exactly one of:

`Documents & Files` / `Filesystem & Workspace` / `Code / Repo / IDE` / `Browser & Web` / `SaaS / API Integrations` / `Local Apps / OS Automation` / `Media` / `DevOps / System / Cloud` / `Search & Research` / `Agent Infra / Orchestration` / `Hybrid`

#### `archetype`
Must be exactly one of:

`retrieve` / `extract` / `transform` / `organize` / `analyze` / `generate` / `edit` / `execute` / `validate` / `monitor` / `automate` / `hybrid`

#### `grading_mode`
- `automated` — all criteria are deterministically checkable via `grade()`
- `llm-judge` — all criteria require semantic evaluation
- `hybrid` — mix of both; default for most tasks

#### `difficulty`
- `low` — single-step, unambiguous output, ≤60s
- `medium` — multi-step, some judgment or env inspection, 60–180s
- `high` — open-ended, complex reasoning or multi-turn clarification, >180s

---

## Section Reference

### `## Prompt`

Real user message — natural language, goal-oriented. NOT a spec or step list.
- Write as the user would type it: terse, target-focused
- When referring to files, use only the filename or a relative path (e.g., `data.csv`, `reports/summary.csv`) — never absolute paths like `/workspace/data.csv` or path placeholders like `{file}`; all assets are placed in the agent's current workspace at runtime

### `## Expected Behavior`

Checklist for evaluators. Covers: key actions → intermediate outputs → final deliverable.

Format: checkbox list (`- [ ]`), ordered by execution flow.

Each item must be concrete and verifiable (not "does a good job").
Each item should map to a Grading Criteria bullet.

```markdown
The agent should:

- [ ] Read `assets/input.csv` and parse its rows
- [ ] Compute the sum of the `amount` column
- [ ] Write a JSON file to `output.json` containing `{"total": <computed_sum>}`
```

### `## Assets Preparation`

All files the agent can read at runtime. Listed as bullet items.

Each entry must include:
- File path (relative to workspace root, e.g., `assets/data.csv`)
- Format / file type
- Content description: schema, size, topic, key values, any constraints

```markdown
- `assets/data.csv` — CSV, 50 rows, columns: `id` (int), `name` (str), `amount` (float); total of `amount` column = 1234.56
- `assets/config.json` — JSON, fields: `output_dir` (str), `format` (str, one of "json"|"csv")
```

### `## Toolset Preparation`

All custom plugins the agent may call. Listed as bullet items.

Each entry must include:
- Tool name (snake_case)
- One-line description
- Input parameters: `name: type — description`
- Return value: type and structure
- Mock type: deterministic mock or live stub

```markdown
- `get_file_meta(path: str) -> dict` — Returns file metadata (size, mtime, mime_type). Deterministic mock returning fixed values for known paths.
  - Input: `path` (str) — absolute or workspace-relative file path
  - Output: `{ "size": int, "mtime": str, "mime_type": str }`
```

Omit this section entirely if no custom tools are needed.

### `## Environment Specification`

Minimal runtime dependencies. Include only what is strictly required.

```markdown
- Platform: Ubuntu 22.04
- Python: 3.11
- pip: pandas==2.1.0, requests==2.31.0
- System: none
- Network: none
- Environment variables: none
```

### `## Grading Criteria`

Two subsections summing to exactly 100%.
- `automated` tasks: 100% Automated, remove LLM section
- `llm-judge` tasks: 100% LLM, remove Automated section
- `hybrid` tasks: split as appropriate

Each Automated bullet → key in `grade()` return dict (text must match exactly).
Each LLM bullet → dimension in LLM Judge Rubric.

```markdown
### Automated Criteria (60%)

- [ ] Output file `output.json` exists
- [ ] `total` field equals 1234.56

### LLM Judge Criteria (40%)

- [ ] Agent explained its computation steps clearly
```

### `## Automated Checks`

Python function `grade(transcript, workspace_path) -> dict`.

Rules:
- Keys must exactly match Automated Criteria bullet text
- Scores: 0.0 (fail) to 1.0 (pass); partial credit allowed
- Never raise exceptions — catch all errors, default to 0.0
- Standard library only unless package is guaranteed by Environment Specification
- No `...` or `pass` placeholders — fully implemented

```python
def grade(transcript: list, workspace_path: str) -> dict:
    import os, json

    scores = {}

    # criterion: "Output file `output.json` exists"
    out_path = os.path.join(workspace_path, "output.json")
    scores["Output file `output.json` exists"] = 1.0 if os.path.isfile(out_path) else 0.0

    # criterion: "`total` field equals 1234.56"
    try:
        with open(out_path) as f:
            data = json.load(f)
        scores["`total` field equals 1234.56"] = 1.0 if abs(data.get("total", 0) - 1234.56) < 0.01 else 0.0
    except Exception:
        scores["`total` field equals 1234.56"] = 0.0

    return scores
```

### `## LLM Judge Rubric`

One rubric block per LLM Judge Criteria bullet. Each block has explicit 1.0 / 0.5 / 0.0 anchors describing concrete, observable agent behaviors.

```markdown
**Computation explanation (0–1):** Whether the agent explained how it computed the total.
- **1.0** — Agent explicitly stated the column used, the operation performed, and the result value
- **0.5** — Agent mentioned the computation but omitted one key detail (column name or result)
- **0.0** — Agent produced no explanation or explanation was factually wrong
```
