"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from lagent.agents.aggregator import DefaultAggregator
from lagent.schema import ActionReturn


class InternClawContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path, tools: List[Dict] = None):
        self.workspace = Path(workspace)
        self.tools = tools or []  # List of available tools, can be populated from skills or elsewhere

    def build_system_prompt(self, env_info: Dict[str, Any] = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity(env_info)]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        if env_info:
            memory_info = env_info.get("memory")
            if (
                memory_info
                and isinstance(memory_info, dict)
                and memory_info.get("available")
                and memory_info.get("long_term")
            ):
                parts.append(f"# Memory\n\n{memory_info['long_term']}")

            active_skills = env_info.get("active_skills")
            if active_skills:
                parts.append(f"# Active Skills\n\n{active_skills}")

            skills_summary = env_info.get("skills")
            if skills_summary:
                parts.append(
                    f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}"""
                )

        parts.append(self._build_runtime_context(None, None))

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, env_info: Dict[str, Any] = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())

        # Prefer runtime info from env_info (remote env), fallback to local platform
        runtime_info = (env_info or {}).get('runtime', {})
        system = runtime_info.get('system') or platform.system()
        machine = runtime_info.get('machine') or platform.machine()
        python_version = runtime_info.get('python_version') or platform.python_version()
        runtime = f"{'macOS' if system == 'Darwin' else system} {machine}, Python {python_version}"

        return f"""# InternClaw 🐈

You are InternClaw, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## InternClaw Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return InternClawContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def aggregate(
        self, messages, name: str, parser=None, system_instruction: str = None, tools: List[Dict] = None
    ) -> List[Dict[str, str]]:
        """Aggregate messages into a format suitable for the agent."""
        messages_list = messages.get_memory()

        # Find the latest env_info
        latest_env_info = None
        for message in messages_list:
            if getattr(message, 'env_info', None) is not None:
                latest_env_info = message.env_info

        _message = [dict(role='system', content=self.build_system_prompt(env_info=latest_env_info))]

        # ── Handle conversation summary (compact) ────────────────
        # If env_info contains conversation_summary + compact_boundary,
        # inject summary as a user message and skip messages before boundary.
        compact_summary = None
        compact_boundary = None
        if latest_env_info:
            compact_summary = latest_env_info.get("conversation_summary")
            compact_boundary = latest_env_info.get("compact_boundary")

        if compact_summary and compact_boundary is not None:
            _message.append(
                dict(
                    role='user',
                    content=(
                        f"[Conversation Summary — the following is a summary "
                        f"of the conversation up to this point]\n\n"
                        f"{compact_summary}"
                    ),
                )
            )
            # Only process messages AFTER the boundary index
            messages_to_process = messages_list[compact_boundary:]
        else:
            messages_to_process = messages_list

        for message in messages_to_process:
            if message.sender == name:
                # msg = {'role': 'assistant', 'content': message.content or ''}
                msg = message.to_model_request()
                if message.tool_calls:
                    # msg['tool_calls'] = message.tool_calls
                    # When tool_calls are present, content should be None or empty for some APIs
                    if not message.content:
                        msg['content'] = None
                # if message.reasoning_content:
                # msg['reasoning_content'] = message.reasoning_content
                _message.append(msg)
            else:
                user_message = message.content
                if isinstance(user_message, list):
                    for m in user_message:
                        if isinstance(m, dict):
                            m = dict(m)  # shallow copy to avoid mutating memory
                            tool_call_id = m.pop('tool_call_id', '')
                            m = ActionReturn(**m)
                        else:
                            tool_call_id = ''
                        assert isinstance(m, ActionReturn), f"Expected m to be ActionReturn, but got {type(m)}"
                        tool_msg = dict(role='tool', content=m.format_result(), name=m.type)
                        if tool_call_id:
                            tool_msg['tool_call_id'] = tool_call_id
                        _message.append(tool_msg)
                else:
                    if len(_message) > 0 and _message[-1]['role'] == 'user':
                        _message[-1]['content'] += user_message
                    else:
                        _message.append(dict(role='user', content=user_message))

        tools_to_use = tools or self.tools
        if latest_env_info and latest_env_info.get("tools"):
            tools_to_use = latest_env_info.get("tools")

        return _message, tools_to_use


if __name__ == "__main__":
    # Example usage
    from lagent.memory import Memory
    from lagent.schema import AgentMessage

    builder = InternClawContextBuilder(
        Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace")
    )
    env_info = {
        "skills": "<skills><skill><name>weather</name></skill></skills>",
        "active_skills": "weather skill content",
        "memory": {"available": True, "long_term": "It's always sunny in Philadelphia."},
    }
    system_prompt = builder.build_system_prompt(env_info=env_info)
    print(system_prompt)
    session = Memory()
    session.add(
        [
            AgentMessage(sender="user", content="What is the weather today?", role="user", env_info=env_info),
            AgentMessage(sender="agent", content="The weather is sunny.", role="assistant"),
            AgentMessage(sender="user", content="What about tomorrow?", role="user"),
            AgentMessage(sender="agent", content="Tomorrow will be cloudy.", role="assistant"),
            AgentMessage(sender="user", content="Any plans for the weekend?", role="user"),
            AgentMessage(sender="agent", content="I am planning to go hiking.", role="assistant"),
        ]
    )
    msgs, tools = builder.aggregate(session, name="agent")
    for msg in msgs:
        print(f"{msg['role']}: {msg['content']}\n")
