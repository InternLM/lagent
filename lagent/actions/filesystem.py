import difflib
from pathlib import Path
from typing import Any, Optional, Type

from asyncer import asyncify

from lagent.actions.base_action import BaseAction, tool_api, AsyncActionMixin
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


def _resolve_path(
    path: str, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class ReadFileAction(AsyncActionMixin, BaseAction):
    """Tool to read file contents."""

    _MAX_CHARS = 128_000

    def __init__(
        self,
        workspace: Optional[str] = None,
        allowed_dir: Optional[str] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        self._workspace = Path(workspace) if workspace else None
        self._allowed_dir = Path(allowed_dir) if allowed_dir else None

    @tool_api
    @asyncify
    def run(self, path: str) -> ActionReturn:
        """Read the contents of a file at the given path.
        
        Args:
            path (str): The file path to read
        """
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Error: File not found: {path}",
                    state=ActionStatusCode.API_ERROR,
                )
            if not file_path.is_file():
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Error: Not a file: {path}",
                    state=ActionStatusCode.API_ERROR,
                )

            size = file_path.stat().st_size
            if size > self._MAX_CHARS * 4:
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Error: File too large ({size:,} bytes). Use exec tool with head/tail/grep to read portions.",
                    state=ActionStatusCode.API_ERROR,
                )

            content = file_path.read_text(encoding="utf-8")
            if len(content) > self._MAX_CHARS:
                content = content[: self._MAX_CHARS] + f"\n\n... (truncated — file is {len(content):,} chars, limit {self._MAX_CHARS:,})"
            
            return ActionReturn(
                type=self.name,
                result=[dict(type='text', content=content)],
                state=ActionStatusCode.SUCCESS,
            )
        except PermissionError as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"PermissionError: {e}",
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"Error reading file: {str(e)}",
                state=ActionStatusCode.API_ERROR,
            )


class WriteFileAction(AsyncActionMixin, BaseAction):
    """Tool to write content to a file."""

    def __init__(
        self,
        workspace: Optional[str] = None,
        allowed_dir: Optional[str] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        self._workspace = Path(workspace) if workspace else None
        self._allowed_dir = Path(allowed_dir) if allowed_dir else None

    @tool_api
    @asyncify
    def run(self, path: str, content: str) -> ActionReturn:
        """Write content to a file at the given path. Creates parent directories if needed.
        
        Args:
            path (str): The file path to write to.
            content (str): The content to write.
        """
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            
            needs_newline = content and not content.endswith("\n")
            if needs_newline:
                content += "\n"

            verb = "Updated" if file_path.exists() else "Created"
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            
            return ActionReturn(
                type=self.name,
                result=[dict(type='text', content=f"Successfully {verb.lower()} file at {path}")],
                state=ActionStatusCode.SUCCESS,
            )
        except PermissionError as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"PermissionError: {e}",
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"Error writing file: {str(e)}",
                state=ActionStatusCode.API_ERROR,
            )


class EditFileAction(AsyncActionMixin, BaseAction):
    """Tool to edit a file using search and replace blocks."""

    def __init__(
        self,
        workspace: Optional[str] = None,
        allowed_dir: Optional[str] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        self._workspace = Path(workspace) if workspace else None
        self._allowed_dir = Path(allowed_dir) if allowed_dir else None

    @tool_api
    @asyncify
    def run(self, path: str, search: str, replace: str) -> ActionReturn:
        """Edit a file by replacing a specific block of text.
        
        Args:
            path (str): The file path to edit.
            search (str): The exact text string to search for and replace. Must match the file content exactly, including whitespace.
            replace (str): The new text to replace the search block with.
        """
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Error: File not found: {path}",
                    state=ActionStatusCode.API_ERROR,
                )
            if not file_path.is_file():
                return ActionReturn(
                    type=self.name,
                    errmsg=f"Error: Not a file: {path}",
                    state=ActionStatusCode.API_ERROR,
                )

            content = file_path.read_text(encoding="utf-8")
            
            count = content.count(search)
            if count == 0:
                return ActionReturn(
                    type=self.name,
                    errmsg="Error: Search text not found exactly in file. Ensure exact whitespace matching.",
                    state=ActionStatusCode.API_ERROR,
                )
            if count > 1:
                return ActionReturn(
                    type=self.name,
                    errmsg="Error: Search text matched multiple times. Provide more context to make it unique.",
                    state=ActionStatusCode.API_ERROR,
                )

            new_content = content.replace(search, replace)
            file_path.write_text(new_content, encoding="utf-8")
            
            return ActionReturn(
                type=self.name,
                result=[dict(type='text', content=f"Successfully edited file at {path}")],
                state=ActionStatusCode.SUCCESS,
            )
        except PermissionError as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"PermissionError: {e}",
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=f"Error editing file: {str(e)}",
                state=ActionStatusCode.API_ERROR,
            )


if __name__ == "__main__":
    # Example usage
    workspace = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/tests/test_actions"
    read_action = ReadFileAction(workspace=workspace)
    write_action = WriteFileAction(workspace=workspace)
    edit_action = EditFileAction(workspace=workspace)
    async def test_actions():
        # Test writing a file
        write_result = await write_action.run(path="test.txt", content="Hello, world!")
        print(write_result)

        # Test reading the file
        read_result = await read_action.run(path="test.txt")
        print(read_result)

        # Test editing the file
        edit_result = await edit_action.run(path="test.txt", search="world", replace="universe")
        print(edit_result)

        # Read the file again to see changes
        read_result_after_edit = await read_action.run(path="test.txt")
        print(read_result_after_edit)
    import asyncio
    asyncio.run(test_actions())