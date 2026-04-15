# Synthesis Patterns

Patterns for generating scripts/, tools/, assets/, and grading logic.

## Table of Contents
1. [scripts/ Template and Patterns](#scripts-template-and-patterns)
2. [Tool Synthesis Patterns](#tool-synthesis-patterns)
3. [Asset Synthesis Patterns](#asset-synthesis-patterns)
4. [Automated Grader Code Patterns](#automated-grader-code-patterns)
   - [Output-focused graders](#output-focused-graders)
   - [Behavior-focused graders (agent asked / inspected / proposed)](#behavior-focused-graders)
5. [LLM Judge Rubric Anchors](#llm-judge-rubric-anchors)
   - [Output quality](#output-quality-content-correctness--completeness--format)
   - [Reasoning / approach quality](#reasoning--approach-quality)
   - [Clarification behavior](#clarification-behavior-underspecified--ambiguous-tasks)

---

## scripts/ Template and Patterns

`scripts/env_spec.sh` is always required. Additional helper scripts (init_db.py, seed_data.py, start_server.py, etc.) also go in `scripts/`.

### env_spec.sh base template

> ⚠️ Replace all `{PLACEHOLDER}` values before using.
>
> **Installation rules (strictly enforced):**
> - **Check before install.** Don't blindly install — first check if the binary/package already exists. Common runtimes (python3, node, git, curl) are usually pre-installed.
> - **User-local first.** For language packages, always install to user-local paths: `pip install --user`, `npm install` (local `node_modules`), `$HOME/.local/bin`, etc.
> - **System packages as fallback.** For tools that can only be installed system-wide (ffmpeg, imagemagick, etc.), check existence first, then try `sudo apt-get` only if missing. If sudo is unavailable, fail with a clear error message — never silently skip.
>
> **Path rules (strictly enforced):**
> - Use `$WORKSPACE`, `$TASK_PATH`, `$HOME` — never hardcode absolute paths like `/tmp/`, `/workspace/`, `/usr/local/bin/`
> - Output files and generated artifacts go under `$WORKSPACE`, not `/tmp/` or any fixed system path
> - To make a mock script available on PATH, use `export PATH="$HOME/.local/bin:$PATH"` rather than copying to `/usr/local/bin/`
> - Mock user-environment dependencies (e.g., a CLI tool that requires GPU/manual install) in `env_spec.sh` when the evaluation environment cannot satisfy them — but keep mocks path-clean using the rules above

```bash
#!/usr/bin/env bash
# scripts/env_spec.sh — Task execution environment initialization
# Task: {task-id}
set -euo pipefail

WORKSPACE="${WORKSPACE_PATH:?Error: WORKSPACE_PATH is not set}"
TASK_PATH="${TASK_PATH:?Error: TASK_PATH is not set}"

# User-local bin on PATH
export PATH="$HOME/.local/bin:$PATH"

echo "=== Copying assets to workspace ==="
mkdir -p "$WORKSPACE"
cp -r "$TASK_PATH/assets/." "$WORKSPACE/"

echo "=== Checking & installing system CLI tools ==="
# Check first; install only if missing and sudo is available
require_cli() {
    if command -v "$1" &>/dev/null; then
        echo "  ✓ $1 already installed"
    elif sudo -n true 2>/dev/null; then
        echo "  ⟳ Installing $1 via apt-get..."
        sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends "$1"
    else
        echo "  ✗ $1 not found and sudo not available" >&2; exit 1
    fi
}
# require_cli "ffmpeg"
# require_cli "jq"

echo "=== Checking & installing Python packages ==="
# Check before install; user-local only
install_pip_pkg() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
# install_pip_pkg "pandas" "pandas==2.1.0"
# install_pip_pkg "requests" "requests==2.31.0"

echo "=== Installing custom tool plugins ==="
bash "$TASK_PATH/scripts/plugin_install.sh"

echo "=== Environment variables ==="
# export {ENV_VAR_NAME}="{value}"

echo "=== One-time setup ==="
# python3 "$TASK_PATH/scripts/init_db.py"

echo "=== Environment ready ==="
```

### Patterns for common environments

**System CLI tools (check → sudo fallback → fail):**
```bash
require_cli() {
    if command -v "$1" &>/dev/null; then
        echo "  ✓ $1 already installed"
    elif sudo -n true 2>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends "$1"
    else
        echo "  ✗ $1 not found and sudo not available" >&2; exit 1
    fi
}
require_cli "ffmpeg"
require_cli "jq"
```

**Python packages (user-local, check first):**
```bash
install_pip_pkg() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
install_pip_pkg "pandas" "pandas==2.1.0"
install_pip_pkg "numpy" "numpy"
install_pip_pkg "sklearn" "scikit-learn"
```

**Node.js packages (local node_modules):**
```bash
cd "$WORKSPACE" && npm install
```

**SQLite database init:**
```bash
python3 "$TASK_PATH/scripts/init_db.py"
```

**No network access (default):**
```bash
# No network setup needed — task runs offline
```

**Service startup (e.g., local API mock):**
```bash
python3 "$TASK_PATH/scripts/start_server.py" &
sleep 1  # wait for server to start
```

---

## Tool Synthesis Patterns

Tool synthesis is governed by the **`openclaw-plugin-creator`** skill — load it for the complete structure (`openclaw.plugin.json`, `package.json`, `src/index.ts`) and implementation guidance.

Task-specific note: tools in `tools/` are invoked **by the agent during evaluation**. When the task has a fixed expected output, implement a deterministic mock rather than a live integration — the mock should return realistic, precomputed data consistent with the synthesized assets.

### plugin_install.sh

After synthesizing all tool plugins, create `scripts/plugin_install.sh` to install them. This script is referenced in the Step 4 static checklist.

```bash
#!/usr/bin/env bash
# scripts/plugin_install.sh — Install all custom tool plugins for this task
# Task: {task-id}
set -euo pipefail

TASK_PATH="${TASK_PATH:?Error: TASK_PATH is not set}"

echo "=== Installing custom tool plugins ==="
for plugin_dir in "$TASK_PATH/tools"/*/; do
    if [ -f "$plugin_dir/openclaw.plugin.json" ]; then
        echo "Installing plugin: $(basename "$plugin_dir")"
        cd "$plugin_dir" && openclaw plugins install .
    fi
done

echo "=== Plugin installation complete ==="
```

---

## Asset Synthesis Patterns

### Principle: Realistic, Self-Consistent Content

Assets must be realistic enough that an agent can complete the task without confusion. Avoid Lorem Ipsum unless the task is explicitly about text corpora.

### Text / Markdown files

Generate content matching the task topic. Include realistic structure (headings, lists, code blocks) if the task involves document processing.

### CSV / JSON data files

- Define a clear schema in the task's **Assets Preparation** section first
- Generate 20–100 rows; more for data analysis tasks, fewer for simple parsing tasks
- Ensure the "expected answer" in the grader is precomputed from the generated data
- Use consistent column names and types; avoid nulls unless the task tests null handling

```python
# Example: generating a CSV asset programmatically
import csv, random, io

rows = [{"id": i, "name": f"Student_{i}", "score": round(random.uniform(40, 100), 2)}
        for i in range(1, 51)]
out = io.StringIO()
writer = csv.DictWriter(out, fieldnames=["id", "name", "score"])
writer.writeheader()
writer.writerows(rows)
csv_content = out.getvalue()
```

### Code / Project directories

Scaffold a realistic project structure. Include:
- A main entry point (e.g., `main.py`, `index.js`)
- At least one module with meaningful stub functions
- A README or docstring explaining what the project is supposed to do
- Any config files the task requires (e.g., `requirements.txt`, `package.json`)

### Database files (SQLite)

Provide a Python script at `scripts/init_db.py` that creates and populates the database, and call it from `env_spec.sh`. Do not commit binary `.sqlite` files directly.

```python
# scripts/init_db.py
import sqlite3, os

# Use WORKSPACE_PATH env var (injected at runtime); fall back to cwd for local testing
workspace = os.environ.get("WORKSPACE_PATH", os.getcwd())
db_path = os.environ.get("DB_PATH", os.path.join(workspace, "db.sqlite"))
conn = sqlite3.connect(db_path)
conn.execute("""CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY,
    name TEXT,
    value REAL
)""")
conn.executemany("INSERT INTO records VALUES (?, ?, ?)", [
    (1, "alpha", 42.0),
    (2, "beta",  17.5),
    (3, "gamma", 99.1),
])
conn.commit()
conn.close()
print(f"Database initialized at {db_path}")
```

---

## Automated Grader Code Patterns

Two sub-types, pick based on what the task actually grades:

- **Output-focused** — check files or values the agent produced (existence, content, format, numeric accuracy). Primary evidence is `workspace_path`.
- **Behavior-focused** — check how the agent acted (which tools it called, what it said, whether it asked before proceeding). Primary evidence is `transcript`.

Both types can coexist in a single `grade()` function; just mix snippets as needed.

### Output-focused graders

### File existence

```python
exists = os.path.isfile(os.path.join(workspace_path, "output.md"))
scores["File `output.md` created"] = 1.0 if exists else 0.0
```

### Non-empty content

```python
content = open(path).read().strip() if os.path.isfile(path) else ""
scores["Output is non-empty"] = 1.0 if content else 0.0
```

### Regex match

```python
import re
scores["Output contains a Markdown heading"] = (
    1.0 if re.search(r"^#{1,6} \w", content, re.MULTILINE) else 0.0
)
```

### JSON validity + schema check

```python
import json
try:
    data = json.loads(content)
    scores["Output is valid JSON"] = 1.0
    scores["JSON has required keys"] = 1.0 if {"name", "value"} <= data.keys() else 0.0
except (json.JSONDecodeError, AttributeError):
    scores["Output is valid JSON"] = 0.0
    scores["JSON has required keys"] = 0.0
```

### Script execution

```python
import subprocess
result = subprocess.run(
    ["python3", os.path.join(workspace_path, "solution.py")],
    capture_output=True, timeout=30, cwd=workspace_path
)
scores["Script runs without error"] = 1.0 if result.returncode == 0 else 0.0
```

### Numeric accuracy

```python
try:
    val = float(open(os.path.join(workspace_path, "output.txt")).read().strip())
    expected = 73.42
    scores["Value within ±0.01 of expected"] = 1.0 if abs(val - expected) <= 0.01 else 0.0
except (ValueError, FileNotFoundError):
    scores["Value within ±0.01 of expected"] = 0.0
```

### Transcript-based check

```python
tool_calls = [m.get("tool_name") for m in transcript if m.get("type") == "tool_call"]
scores["Agent used query_database tool"] = 1.0 if "query_database" in tool_calls else 0.0
```

### Partial credit

```python
lines = [l for l in content.splitlines() if l.strip()]
scores["Output has sufficient lines"] = min(1.0, len(lines) / 10)
```

### Safe wrapper (always use in grade())

```python
def grade(transcript: list, workspace_path: str) -> dict:
    import os, re
    scores = {}
    try:
        # ... all grading logic here — no `pass` or `...` placeholders ...
        scores["criterion text"] = 1.0  # replace with real logic
    except Exception:
        # Never let grade() raise — default unscored criteria to 0.0
        for key in ["criterion text"]:  # list all expected keys
            scores.setdefault(key, 0.0)
    return scores
```

---

### Behavior-focused graders (agent asked / inspected / proposed)

For tasks where conditions omit something the query implies, the agent's **transcript** is the primary evidence. Correct behavior is about how the agent responds to the gap, not what file it produces.

#### Helper: extract all agent text turns

```python
def _agent_text(transcript: list) -> str:
    """Concatenate all assistant/agent text messages into one searchable string."""
    parts = []
    for m in transcript:
        if m.get("role") in ("assistant", "agent") and isinstance(m.get("content"), str):
            parts.append(m["content"])
    return "\n".join(parts).lower()
```

#### Check: agent named the missing parameter

```python
agent_text = _agent_text(transcript)
# Replace "output format" with the actual missing parameter name
named_gap = any(kw in agent_text for kw in ["output format", "format", "which format"])
scores["Agent identified the missing parameter by name"] = 1.0 if named_gap else 0.0
```

#### Check: agent inspected the environment before responding

```python
tool_calls = [m.get("tool_name", "") for m in transcript if m.get("type") == "tool_call"]
env_tools = {"list_files", "read_file", "read_config", "ls", "cat"}  # adjust to task toolset
inspected = bool(env_tools & set(tool_calls))
scores["Agent inspected environment before responding"] = 1.0 if inspected else 0.0
```

#### Check: agent proposed concrete options (not a vague "please clarify")

```python
import re
agent_text = _agent_text(transcript)
# Look for enumeration patterns: "1.", "- option", "either X or Y", etc.
has_options = bool(re.search(
    r"(\b(option|choice|either|or)\b.*\b(option|choice|either|or)\b"
    r"|\b1[\.\)]\s+\w|\b-\s+\w.*\n\s*-\s+\w)",
    agent_text, re.IGNORECASE | re.DOTALL
))
scores["Agent proposed 2+ concrete options with reasoning"] = 1.0 if has_options else 0.0
```

#### Check: agent did NOT silently proceed to produce output

```python
import os
# The task's expected output file should NOT exist if agent correctly withheld action
output_exists = os.path.isfile(os.path.join(workspace_path, "output.txt"))
scores["Agent did not silently produce output without resolving ambiguity"] = (
    0.0 if output_exists else 1.0
)
```

#### Check: agent asked a targeted clarifying question

```python
agent_text = _agent_text(transcript)
# Look for question marks and parameter-related keywords in the same sentence
sentences = re.split(r"[.!]\s+", agent_text)
asked = any("?" in s and any(kw in s for kw in ["format", "output", "destination", "version"])
            for s in sentences)
scores["Agent asked a targeted clarifying question"] = 1.0 if asked else 0.0
```

*For LLM judge rubric templates matching these graders, see [LLM Judge Rubric Anchors → Clarification behavior](#clarification-behavior-underspecified--ambiguous-tasks) below.*

---

## LLM Judge Rubric Anchors

Use LLM judge criteria whenever a script cannot capture the requirement — output quality, reasoning depth, or clarification behavior. The three most common scenarios:

### Output quality (content correctness / completeness / format)

Use when the output is a natural-language artifact (summary, explanation, report, code review, plan) and correctness requires understanding intent, not just pattern matching.

```markdown
**{Criterion name} (0–1):**
- **1.0** — {describe what a fully correct, complete, well-formed response looks like}
- **0.5** — {describe a partially correct response: present but incomplete, slightly off, or minor format issue}
- **0.0** — {describe a clearly wrong response: missing, hallucinated, or structurally broken}
```

Example — "Summary covers all key points":
```markdown
**Summary covers all key points (0–1):**
- **1.0** — Summary mentions all {N} major topics from the source document with accurate facts
- **0.5** — Summary covers most topics but omits 1–2 important points, or includes minor inaccuracies
- **0.0** — Summary is missing, covers fewer than half the topics, or contains significant hallucinations
```

### Reasoning / approach quality

Use when the task is open-ended and the path matters as much as (or more than) the final answer — e.g., debugging steps, investigation approach, multi-step planning.

```markdown
**{Criterion name} (0–1):**
- **1.0** — Agent followed a logical, systematic approach: {describe the ideal sequence of steps or tool calls}
- **0.5** — Agent reached the correct conclusion but via an inefficient or partially incorrect route
- **0.0** — Agent skipped key steps, jumped to conclusions without evidence, or produced an incorrect result
```

Example — "Agent diagnosed root cause correctly":
```markdown
**Agent diagnosed root cause correctly (0–1):**
- **1.0** — Agent inspected relevant logs/files, identified the specific error and its cause, and stated it clearly
- **0.5** — Agent found the symptom but misidentified the root cause, or identified the cause without citing evidence
- **0.0** — Agent did not investigate, guessed without evidence, or identified an unrelated issue
```

### Clarification behavior (underspecified / ambiguous tasks)

Use when the query intentionally omits required information and the correct behavior is to ask, not to proceed silently.

```markdown
**Agent identified the missing parameter (0–1):**
- **1.0** — Agent explicitly named "{parameter}" as missing or unknown, before taking any action
- **0.5** — Agent expressed uncertainty but did not name the specific parameter
- **0.0** — Agent proceeded without acknowledging the gap, or asked a completely generic question

**Agent proposed concrete options (0–1):**
- **1.0** — Agent listed 2+ specific, named options with a brief rationale for each
- **0.5** — Agent suggested options exist but did not name them, or listed only one
- **0.0** — Agent gave no options; asked only "please clarify" with no guidance

**Agent did not proceed without required information (0–1):**
- **1.0** — Agent withheld all output-producing actions until the gap was resolved
- **0.5** — Agent produced partial output with an explicit caveat about the assumption made
- **0.0** — Agent silently assumed a value and produced output without flagging the assumption
```
