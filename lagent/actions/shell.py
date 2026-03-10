import asyncio
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Type

from asyncer import asyncify

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class ShellAction(AsyncActionMixin, BaseAction):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: Optional[str] = None,
        deny_patterns: Optional[List[str]] = None,
        allow_patterns: Optional[List[str]] = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    def _guard_command(self, command: str, cwd: str) -> Optional[str]:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"
            
            cwd_path = Path(cwd).resolve()
            # A rough implementation of checking absolute paths
            paths = re.findall(r'(/[^\s]+)', cmd)
            for raw in paths:
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"
        return None

    @tool_api
    @asyncify
    def run(self, command: str, working_dir: Optional[str] = None) -> ActionReturn:
        """Execute a shell command and return its output. Use with caution.
        
        Args:
            command (str): The shell command to execute.
            working_dir (str, optional): Optional working directory for the command.
        """
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return ActionReturn(
                type=self.name,
                errmsg=guard_error,
                state=ActionStatusCode.API_ERROR
            )
        
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        # We need an event loop, since run needs to be synchronously resolvable but internally we use async logic.
        # Since it's decorated with @asyncify, we can actually write sync code, but since the original used
        # asyncio.create_subprocess_shell, we'll wrap it.

        try:
            result = asyncio.run(self._execute_async(command, cwd, env))
            return ActionReturn(
                type=self.name,
                result=[dict(type='text', content=result)],
                state=ActionStatusCode.SUCCESS
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"Error executing command: {str(e)}",
                state=ActionStatusCode.API_ERROR
            )

    async def _execute_async(self, command: str, cwd: str, env: dict) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            raise TimeoutError(f"Command timed out after {self.timeout} seconds")
        
        output_parts = []
        
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        
        if stderr:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text.strip():
                output_parts.append(f"STDERR:\n{stderr_text}")
        
        if process.returncode != 0:
            output_parts.append(f"\nExit code: {process.returncode}")
        
        result = "\n".join(output_parts) if output_parts else "(no output)"
        
        # Truncate very long output
        max_len = 10000
        if len(result) > max_len:
            result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"
        
        return result


if __name__ == "__main__":
    import asyncio
    action = ShellAction(timeout=10, restrict_to_workspace=True)
    async def test():
        result = await action.run("echo Hello World && ls -la && sleep 2 && echo Done")
    asyncio.run(test())