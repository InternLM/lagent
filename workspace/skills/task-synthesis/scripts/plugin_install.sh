#!/usr/bin/env bash
# task_initialize.sh — Initialize a synthesized task package
#
# Usage:
#   ./task_initialize.sh <task_dir> [skill_install_dir]
#
# Arguments:
#   task_dir          Path to the synthesized task directory (contains task.md, assets/, tools/, scripts/)
#   skill_install_dir (optional) Directory to install the source skill into
#                     Default: /mnt/shared-storage-user/llmit/user/sunyanan/openclaw/runtime_skills
#
# What this script does:
#   1. Run scripts/env_spec.sh   — set up the task execution environment
#   2. Install tools/             — install each custom tool plugin into OpenClaw via `openclaw plugins install --link`
#   3. Install the source skill   — copy the skill directory to <skill_install_dir>

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_dir> [skill_install_dir]" >&2
  exit 1
fi

TASK_DIR="$(realpath "$1")"
SKILL_INSTALL_DIR="$(realpath "${2:-/mnt/shared-storage-user/llmit/user/sunyanan/openclaw/runtime_skills}")"

if [[ ! -d "$TASK_DIR" ]]; then
  echo "Error: task_dir '$TASK_DIR' does not exist." >&2
  exit 1
fi

echo "=== task_initialize ==="
echo "Task dir  : $TASK_DIR"
echo "Skill dir : $SKILL_INSTALL_DIR"
echo

# ── Step 1: Run env_spec.sh ───────────────────────────────────────────────────

ENV_SCRIPT="$TASK_DIR/scripts/env_spec.sh"

echo "--- Step 1: Running env_spec.sh ---"
if [[ -f "$ENV_SCRIPT" ]]; then
  bash "$ENV_SCRIPT"
  echo "env_spec.sh completed."
else
  echo "WARNING: $ENV_SCRIPT not found, skipping environment setup."
fi
echo

# ── Step 2: Install tools/ plugins into OpenClaw ─────────────────────────────

TOOLS_DIR="$TASK_DIR/tools"

echo "--- Step 2: Installing tool plugins ---"
if [[ -d "$TOOLS_DIR" ]]; then
  TOOL_COUNT=0
  for tool_path in "$TOOLS_DIR"/*/; do
    if [[ -d "$tool_path" ]]; then
      tool_name="$(basename "$tool_path")"
      echo "  Installing plugin: $tool_name ($tool_path)"
      openclaw plugins install --link "$tool_path"
      TOOL_COUNT=$((TOOL_COUNT + 1))
    fi
  done
  if [[ $TOOL_COUNT -eq 0 ]]; then
    echo "  No custom tool plugins found in $TOOLS_DIR — skipping."
  else
    echo "  Installed $TOOL_COUNT plugin(s)."
  fi
else
  echo "  tools/ directory not found — skipping plugin installation."
fi
echo

# ── Step 3: Install the source skill ─────────────────────────────────────────

# Looks up the skill's relative_path from the skill catalog JSONL, then copies
# the skill directory (parent of SKILL.md) to <skill_install_dir>.

SKILL_CATALOG="/mnt/shared-storage-user/llmit/user/tangyinhao/skills/data/awesome_openclaw_skills_and_skillshtop100_tag_v01.jsonl"
SKILL_ROOT="/mnt/shared-storage-user/llmit/user/tangyinhao/skills"

echo "--- Step 3: Installing source skill ---"

# Extract skill_id from task.md frontmatter (skill_set: [skill-id])
TASK_MD="$TASK_DIR/task.md"
if [[ ! -f "$TASK_MD" ]]; then
  echo "  WARNING: task.md not found at $TASK_MD — skipping skill installation."
else
  SKILL_ID=$(grep -A1 'skill_set:' "$TASK_MD" | grep '^\s*-' | head -1 | sed 's/.*-\s*//' | tr -d '[:space:]')
  echo "  Detected skill_id from task.md: '$SKILL_ID'"

  if [[ -z "$SKILL_ID" || "$SKILL_ID" == "null" ]]; then
    echo "  WARNING: Could not parse skill_id from task.md — skipping skill installation."
  elif [[ ! -f "$SKILL_CATALOG" ]]; then
    echo "  WARNING: Skill catalog not found at $SKILL_CATALOG — skipping skill installation."
  else
    # Look up relative_path by id in the JSONL catalog
    RELATIVE_PATH=$(python3 - <<PYEOF
import json, sys

catalog = "$SKILL_CATALOG"
skill_id = "$SKILL_ID"

with open(catalog, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("id") == skill_id:
            rp = entry.get("relative_path", "")
            # relative_path points to SKILL.md — we want the parent directory
            if rp.endswith("/SKILL.md"):
                rp = rp[: -len("/SKILL.md")]
            print(rp)
            sys.exit(0)

# not found
sys.exit(1)
PYEOF
    )

    if [[ $? -ne 0 || -z "$RELATIVE_PATH" ]]; then
      echo "  WARNING: skill_id '$SKILL_ID' not found in catalog $SKILL_CATALOG — skipping skill installation."
    else
      SKILL_SOURCE_DIR="$SKILL_ROOT/$RELATIVE_PATH"
      echo "  Resolved skill source: $SKILL_SOURCE_DIR"

      if [[ ! -d "$SKILL_SOURCE_DIR" ]]; then
        echo "  WARNING: Skill directory '$SKILL_SOURCE_DIR' does not exist — skipping skill installation."
      else
        SKILL_NAME="$(basename "$SKILL_SOURCE_DIR")"
        DEST="$SKILL_INSTALL_DIR/$SKILL_NAME"
        echo "  Copying skill '$SKILL_NAME' -> $DEST"
        mkdir -p "$SKILL_INSTALL_DIR"
        cp -r "$SKILL_SOURCE_DIR" "$DEST"
        echo "  Skill installed to: $DEST"
      fi
    fi
  fi
fi
echo

echo "=== task_initialize complete ==="
