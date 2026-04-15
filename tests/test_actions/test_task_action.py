"""Unit tests for TaskAction (lagent/actions/task.py).

Bypasses the circular import in lagent packages by importing the
task modules directly before any lagent __init__.py gets triggered.
"""

import os
import sys
import types

# --- bypass circular import in lagent.services.__init__.py ---
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)
if "lagent.services" not in sys.modules:
    _pkg = types.ModuleType("lagent.services")
    _pkg.__path__ = [os.path.join(_here, "lagent", "services")]
    _pkg.__package__ = "lagent.services"
    sys.modules["lagent.services"] = _pkg

import pytest

from lagent.actions.task import TaskAction, AsyncTaskAction
from lagent.schema import ActionStatusCode
from lagent.services.task import TaskBoard


def make_action() -> TaskAction:
    board = TaskBoard()
    return TaskAction(task_board=board)


class TestTaskActionCreate:
    def test_create_via_method(self):
        action = make_action()
        result = action.create(subject="Fix bug", description="Fix it")
        assert "Created task #1" in result.result[0]["content"]

    def test_create_with_blocked_by(self):
        action = make_action()
        action.create(subject="A", description="desc")
        result = action.create(subject="B", description="desc", blocked_by="1")
        assert "blocked by #1" in result.result[0]["content"]

    def test_create_with_metadata(self):
        action = make_action()
        result = action.create(
            subject="Task", description="desc",
            metadata='{"priority": "high"}',
        )
        assert result.result is not None
        task = action._board.get("1")
        assert task.metadata == {"priority": "high"}

    def test_create_with_active_form(self):
        action = make_action()
        action.create(subject="Run tests", description="desc", active_form="Running tests")
        task = action._board.get("1")
        assert task.active_form == "Running tests"


class TestTaskActionUpdate:
    def test_update_status(self):
        action = make_action()
        action.create(subject="Task", description="desc")
        result = action.update(task_id="1", status="in_progress")
        assert "in_progress" in result.result[0]["content"]

    def test_update_nonexistent(self):
        action = make_action()
        result = action.update(task_id="999", status="completed")
        assert result.state == ActionStatusCode.API_ERROR

    def test_update_deleted(self):
        action = make_action()
        action.create(subject="Task", description="desc")
        result = action.update(task_id="1", status="deleted")
        assert "deleted" in result.result[0]["content"]
        assert action._board.get("1") is None

    def test_update_no_fields(self):
        action = make_action()
        action.create(subject="Task", description="desc")
        result = action.update(task_id="1")
        assert result.state == ActionStatusCode.ARGS_ERROR

    def test_update_add_blocked_by(self):
        action = make_action()
        action.create(subject="A", description="desc")
        action.create(subject="B", description="desc")
        action.update(task_id="2", add_blocked_by="1")
        t2 = action._board.get("2")
        assert "1" in t2.blocked_by

    def test_update_metadata_merge(self):
        action = make_action()
        action.create(subject="Task", description="desc", metadata='{"a": 1}')
        action.update(task_id="1", metadata='{"b": 2}')
        task = action._board.get("1")
        assert task.metadata == {"a": 1, "b": 2}


class TestTaskActionGet:
    def test_get_existing(self):
        action = make_action()
        action.create(subject="Fix bug", description="Fix the login bug")
        result = action.get(task_id="1")
        assert "Fix bug" in result.result[0]["content"]
        assert "Fix the login bug" in result.result[0]["content"]

    def test_get_nonexistent(self):
        action = make_action()
        result = action.get(task_id="999")
        assert result.state == ActionStatusCode.API_ERROR


class TestTaskActionList:
    def test_list_all(self):
        action = make_action()
        action.create(subject="A", description="desc")
        action.create(subject="B", description="desc")
        result = action.list()
        content = result.result[0]["content"]
        assert "#1" in content
        assert "#2" in content

    def test_list_filtered(self):
        action = make_action()
        action.create(subject="A", description="desc")
        action.create(subject="B", description="desc")
        action.update(task_id="1", status="completed")
        result = action.list(status="completed")
        content = result.result[0]["content"]
        assert "#1" in content
        assert "#2" not in content

    def test_list_empty(self):
        action = make_action()
        result = action.list()
        assert "No tasks" in result.result[0]["content"]

    def test_list_shows_blockers(self):
        action = make_action()
        action.create(subject="A", description="desc")
        action.create(subject="B", description="desc", blocked_by="1")
        result = action.list()
        content = result.result[0]["content"]
        assert "blocked by #1" in content


class TestAsyncTaskAction:
    def test_instantiation(self):
        board = TaskBoard()
        action = AsyncTaskAction(task_board=board)
        assert action._board is board


class TestToolDescription:
    def test_is_toolkit(self):
        action = make_action()
        assert action.is_toolkit

    def test_has_four_apis(self):
        action = make_action()
        desc = action.description
        api_names = {api["name"] for api in desc["api_list"]}
        assert "create" in api_names
        assert "update" in api_names
        assert "get" in api_names
        assert "list" in api_names
