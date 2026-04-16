"""Skills loader for agent capabilities."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable
import asyncio

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class BaseSkillsBackend:
    """Abstract backend for skill discovery and loading."""

    async def list_skill_entries(self) -> list[dict[str, str]]:
        raise NotImplementedError

    async def read_skill(self, name: str) -> str | None:
        raise NotImplementedError


class FilesystemSkillsBackend(BaseSkillsBackend):
    """Filesystem-backed skill backend."""

    def __init__(self, workspace_skills: Path, builtin_skills: Path | None = None):
        self.workspace_skills = workspace_skills
        self.builtin_skills = builtin_skills

    async def list_skill_entries(self) -> list[dict[str, str]]:
        skills: list[dict[str, str]] = []

        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        return skills

    async def read_skill(self, name: str) -> str | None:
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return await asyncio.to_thread(workspace_skill.read_text, encoding="utf-8")

        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return await asyncio.to_thread(builtin_skill.read_text, encoding="utf-8")

        return None


class SandboxSkillsBackend(BaseSkillsBackend):
    """Sandbox-backed skills storage that executes commands in a remote execution environment."""

    def __init__(
        self,
        action: Any,
        *,
        workspace_root: str = ".",
        builtin_skills: Path | None = None,
        command_builder: Callable[[str, str], str] | None = None,
    ):
        self.action = action
        self.workspace_root = workspace_root.rstrip("/") or "."
        self.builtin_skills = builtin_skills
        self.command_builder = command_builder or self._default_command_builder

    @staticmethod
    def _b64_python(script: str) -> str:
        """Encode a Python script as base64 and return a shell command that decodes and executes it.

        This avoids all quoting/escaping issues when running Python via a remote shell.
        """
        import base64
        encoded = base64.b64encode(script.encode()).decode()
        return f"echo {encoded} | base64 -d | python3"

    def _default_command_builder(self, op: str, target: str) -> str:
        skills_root = f"{self.workspace_root}/skills"
        if op == "list":
            script = (
                "import json\n"
                "from pathlib import Path\n"
                f"root = Path('{skills_root}')\n"
                "items = [\n"
                "    {'name': d.name, 'path': str(d / 'SKILL.md'), 'source': 'workspace'}\n"
                "    for d in (root.iterdir() if root.exists() else [])\n"
                "    if d.is_dir() and (d / 'SKILL.md').exists()\n"
                "]\n"
                "print(json.dumps(items, ensure_ascii=False))\n"
            )
            return self._b64_python(script)
        if op == "read":
            skill_file = f"{skills_root}/{target}/SKILL.md"
            return f"cat '{skill_file}'"
        raise ValueError(f"Unsupported operation: {op}")

    async def _run(self, command: str) -> str | None:
        """异步执行命令并解析输出"""
        import json
        from lagent.schema import ActionStatusCode

        result = await self.action.run(command=command)
        if result.state != ActionStatusCode.SUCCESS:
            return None

        try:
            # 尝试解析 MCP 格式的返回
            if isinstance(result.result, list) and len(result.result) > 0:
                content_str = result.result[0].get('content', '')
                content_dict = json.loads(content_str)
                if content_dict.get('exit_code') == 0:
                    return content_dict.get('stdout', '').strip()
                else:
                    return None
        except Exception:
            pass

        return None

    async def list_skill_entries(self) -> list[dict[str, str]]:
        """异步获取 skill 列表"""
        import json
        raw = await self._run(self.command_builder("list", ""))
        skills: list[dict[str, str]] = []
        if raw:
            # stdout 可能包含多行，尝试从每行中找到 JSON 数组
            for line in raw.splitlines():
                line = line.strip()
                if not line or not line.startswith('['):
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, list):
                        skills = [
                            s for s in parsed
                            if isinstance(s, dict) and 'name' in s
                        ]
                        break
                except json.JSONDecodeError:
                    continue
        # 合并内置 skills
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})
        return skills

    async def read_skill(self, name: str) -> str | None:
        """异步读取 skill 内容"""
        content = await self._run(self.command_builder("read", name))
        if content:
            return content
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return await asyncio.to_thread(builtin_skill.read_text, encoding="utf-8")
        return None


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = Path(workspace)
        self.workspace_skills = self.workspace / "skills"
        self.builtin_skills = Path(builtin_skills_dir) if builtin_skills_dir else BUILTIN_SKILLS_DIR
        self.backend: BaseSkillsBackend = FilesystemSkillsBackend(self.workspace_skills, self.builtin_skills)

    def bind_backend(self, backend: BaseSkillsBackend) -> None:
        """Replace the default filesystem backend with a custom backend."""
        self.backend = backend

    async def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        skills = await self.backend.list_skill_entries()

        # Filter by requirements
        if filter_unavailable:
            filtered_skills = []
            for s in skills:
                meta = await self._get_skill_meta(s["name"])
                if self._check_requirements(meta):
                    filtered_skills.append(s)
            return filtered_skills
        return skills

    async def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        return await self.backend.read_skill(name)

    async def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = await self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    async def build_skills_summary(self) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Returns:
            XML-formatted skills summary.
        """
        all_skills = await self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(await self._get_skill_description(s["name"]))
            skill_meta = await self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            # Show missing requirements for unavailable skills
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """Get a description of missing requirements."""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    async def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = await self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    def _parse_internclaw_metadata(self, raw: str) -> dict:
        """Parse skill metadata JSON from frontmatter (supports internclaw and openclaw keys)."""
        try:
            data = json.loads(raw)
            return data.get("internclaw", data.get("openclaw", {})) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    async def _get_skill_meta(self, name: str) -> dict:
        """Get internclaw metadata for a skill (cached in frontmatter)."""
        meta = await self.get_skill_metadata(name) or {}
        return self._parse_internclaw_metadata(meta.get("metadata", ""))

    async def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        result = []
        for s in await self.list_skills(filter_unavailable=True):
            meta = await self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_internclaw_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    async def get_skill_metadata(self, name: str) -> dict | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        content = await self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # Simple YAML parsing
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None


if __name__ == '__main__':
    from lagent.actions.mcp_client import AsyncMCPClientSandbox
    import asyncio
    import json
    from pathlib import Path
    init_dir = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace"

    async def main():
        shell_action = AsyncMCPClientSandbox('http', url='http://simple-shell.ailab.ailab.ai/mcp', init_dir=init_dir)
        home_path = await shell_action.run(command='ls -la')
        home_path = json.loads(home_path.result[0]['content'])['cwd']

        skill_loader = SkillsLoader(Path(home_path))
        backend = SandboxSkillsBackend(shell_action, workspace_root=os.path.join(home_path, 'workspace'))
        skill_loader.bind_backend(backend)
        
        try:
            skills = await skill_loader.list_skills(filter_unavailable=False)
            print("Skills:")
            for s in skills:
                print(s)
        except Exception as e:
            print("Failed to parse skills:", e)
        
        print(await skill_loader.build_skills_summary())
        print(await skill_loader.load_skills_for_context(['weather']))

    asyncio.run(main())
