---
id: {task-id}
name: {task-name}
scene: {scene}
category: {category}
family: {family}
archetype: {archetype}
skill_set:
  - {skill-id}
grading_mode: automated|llm-judge|hybrid
timeout_seconds: 180
workspace_files:
  - assets/{filename-or-dir}
difficulty: low|medium|high
status: preparing|ready|deprecated
---

## Prompt

{task prompt — natural language, goal-oriented user message}

## Expected Behavior

The agent should:

- [ ] {action or intermediate step}
- [ ] {action or intermediate step}
- [ ] {final output produced}

## Assets Preparation

- `assets/{filename}` — {format; content description including schema, size, key values}

## Toolset Preparation

- `{tool_name}({param}: {type}, ...) -> {return_type}` — {description}
  - Input: `{param}` ({type}) — {description}
  - Output: `{structure description}`
  - Type: built-in | static mock | input-derived mock

## Environment Specification

- Platform: Ubuntu 22.04
- Python: 3.11
- pip: {package==version, ...}
- System: none
- Network: none
- Environment variables: none
- Services: none

## Grading Criteria

### Automated Criteria ({N}%)

<!-- Omit this section entirely if the task uses LLM judge only -->

- [ ] {deterministically checkable criterion}

### LLM Judge Criteria ({N}%)

<!-- Omit this section entirely if the task uses automated grading only -->

- [ ] {semantically evaluated criterion}

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """
    Args:
        transcript:     Parsed JSONL conversation transcript as list of dicts.
        workspace_path: Absolute path to the task's isolated workspace directory.
    Returns:
        Dict mapping each Automated Criteria bullet (exact text) to score 0.0–1.0.
    """
    import os

    scores = {}

    # --- implement grading logic here ---

    return scores
```

## LLM Judge Rubric

**{Criterion name} (0–1):** {what is being evaluated}

- **1.0** — {excellent / fully correct}
- **0.5** — {partial / acceptable}
- **0.0** — {failure / missing}
