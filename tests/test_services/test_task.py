"""Unit tests for TaskBoard (lagent/services/task.py).

Uses direct file-based import to bypass the circular import in
lagent.services.__init__.py (a pre-existing issue, not introduced by
the task module).
"""

import json
import sys
import tempfile
import types
import os
from pathlib import Path

import pytest

# --- bypass circular import in lagent.services.__init__.py ---
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)
if "lagent.services" not in sys.modules:
    _pkg = types.ModuleType("lagent.services")
    _pkg.__path__ = [os.path.join(_here, "lagent", "services")]
    _pkg.__package__ = "lagent.services"
    sys.modules["lagent.services"] = _pkg

from lagent.services.task import Task, TaskBoard, ClaimResult


# ── Helpers ──────────────────────────────────────────────────────────

def make_board(**kwargs) -> TaskBoard:
    return TaskBoard(**kwargs)


# ── CRUD ─────────────────────────────────────────────────────────────

class TestCreate:
    def test_basic_create(self):
        board = make_board()
        task = board.create("Fix bug", "Fix the login bug")
        assert task.id == "1"
        assert task.subject == "Fix bug"
        assert task.description == "Fix the login bug"
        assert task.status == "pending"
        assert task.blocks == []
        assert task.blocked_by == []

    def test_auto_increment_ids(self):
        board = make_board()
        t1 = board.create("Task 1", "desc")
        t2 = board.create("Task 2", "desc")
        t3 = board.create("Task 3", "desc")
        assert t1.id == "1"
        assert t2.id == "2"
        assert t3.id == "3"

    def test_create_with_active_form(self):
        board = make_board()
        task = board.create("Run tests", "desc", active_form="Running tests")
        assert task.active_form == "Running tests"

    def test_create_with_blocked_by(self):
        board = make_board()
        t1 = board.create("Setup", "desc")
        t2 = board.create("Build", "desc", blocked_by=["1"])
        assert t2.blocked_by == ["1"]
        # Bidirectional: t1 should now block t2
        t1_refreshed = board.get("1")
        assert "2" in t1_refreshed.blocks

    def test_create_with_metadata(self):
        board = make_board()
        task = board.create("Task", "desc", metadata={"priority": "high"})
        assert task.metadata == {"priority": "high"}


class TestUpdate:
    def test_update_status(self):
        board = make_board()
        board.create("Task", "desc")
        updated = board.update("1", status="in_progress")
        assert updated is not None
        assert updated.status == "in_progress"

    def test_update_subject(self):
        board = make_board()
        board.create("Old title", "desc")
        updated = board.update("1", subject="New title")
        assert updated.subject == "New title"

    def test_update_nonexistent_returns_none(self):
        board = make_board()
        result = board.update("999", status="completed")
        assert result is None

    def test_update_add_blocked_by(self):
        board = make_board()
        board.create("Task A", "desc")
        board.create("Task B", "desc")
        board.update("2", add_blocked_by=["1"])
        t1 = board.get("1")
        t2 = board.get("2")
        assert "1" in t2.blocked_by
        assert "2" in t1.blocks

    def test_update_add_blocks(self):
        board = make_board()
        board.create("Task A", "desc")
        board.create("Task B", "desc")
        board.update("1", add_blocks=["2"])
        t1 = board.get("1")
        t2 = board.get("2")
        assert "2" in t1.blocks
        assert "1" in t2.blocked_by

    def test_update_metadata_merge(self):
        board = make_board()
        board.create("Task", "desc", metadata={"a": 1, "b": 2})
        board.update("1", metadata={"b": 99, "c": 3})
        task = board.get("1")
        assert task.metadata == {"a": 1, "b": 99, "c": 3}

    def test_update_metadata_delete_key(self):
        board = make_board()
        board.create("Task", "desc", metadata={"a": 1, "b": 2})
        board.update("1", metadata={"b": None})
        task = board.get("1")
        assert task.metadata == {"a": 1}

    def test_update_deleted_status(self):
        board = make_board()
        board.create("Task", "desc")
        result = board.update("1", status="deleted")
        assert result is None  # deleted returns None
        assert board.get("1") is None
        assert len(board.list()) == 0


class TestDelete:
    def test_delete_existing(self):
        board = make_board()
        board.create("Task", "desc")
        assert board.delete("1") is True
        assert board.get("1") is None

    def test_delete_nonexistent(self):
        board = make_board()
        assert board.delete("999") is False

    def test_cascade_cleanup(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.create("C", "desc", blocked_by=["1"])
        board.delete("1")
        t2 = board.get("2")
        t3 = board.get("3")
        assert "1" not in t2.blocked_by
        assert "1" not in t3.blocked_by


class TestGetAndList:
    def test_get_existing(self):
        board = make_board()
        board.create("Task", "desc")
        task = board.get("1")
        assert task is not None
        assert task.subject == "Task"

    def test_get_nonexistent(self):
        board = make_board()
        assert board.get("999") is None

    def test_list_all(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        assert len(board.list()) == 2

    def test_list_filtered(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.update("1", status="completed")
        assert len(board.list(status="completed")) == 1
        assert len(board.list(status="pending")) == 1


# ── High water mark ──────────────────────────────────────────────────

class TestHighWaterMark:
    def test_ids_never_reused_after_delete(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.create("C", "desc")
        board.delete("2")
        t4 = board.create("D", "desc")
        assert t4.id == "4"

    def test_ids_never_reused_after_delete_all(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.delete("1")
        board.delete("2")
        t3 = board.create("C", "desc")
        assert t3.id == "3"


# ── Dependency graph ─────────────────────────────────────────────────

class TestDependencyGraph:
    def test_bidirectional_on_create(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        a = board.get("1")
        b = board.get("2")
        assert "2" in a.blocks
        assert "1" in b.blocked_by

    def test_bidirectional_on_update(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.update("2", add_blocked_by=["1"])
        a = board.get("1")
        b = board.get("2")
        assert "2" in a.blocks
        assert "1" in b.blocked_by

    def test_cascade_delete_cleans_both_directions(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.create("C", "desc")
        board.update("1", add_blocks=["3"])
        assert "2" in board.get("1").blocks
        assert "3" in board.get("1").blocks
        board.delete("1")
        assert "1" not in board.get("2").blocked_by
        assert "1" not in board.get("3").blocked_by

    def test_no_duplicate_deps(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.update("2", add_blocked_by=["1"])
        b = board.get("2")
        assert b.blocked_by.count("1") == 1
        a = board.get("1")
        assert a.blocks.count("2") == 1


# ── Claim ────────────────────────────────────────────────────────────

class TestClaim:
    def test_claim_success(self):
        board = make_board()
        board.create("Task", "desc")
        result = board.claim("1", "worker-A")
        assert result.success is True
        assert result.task.owner == "worker-A"
        assert result.task.status == "in_progress"

    def test_claim_nonexistent(self):
        board = make_board()
        result = board.claim("999", "worker-A")
        assert result.success is False
        assert result.reason == "task_not_found"

    def test_claim_completed_task(self):
        board = make_board()
        board.create("Task", "desc")
        board.update("1", status="completed")
        result = board.claim("1", "worker-A")
        assert result.success is False
        assert result.reason == "already_completed"

    def test_claim_blocked_task(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        result = board.claim("2", "worker-A")
        assert result.success is False
        assert result.reason == "blocked"
        assert "1" in result.blocked_by_tasks

    def test_claim_unblocked_after_completion(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.update("1", status="completed")
        result = board.claim("2", "worker-A")
        assert result.success is True

    def test_claim_agent_busy(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.claim("1", "worker-A")
        result = board.claim("2", "worker-A", check_busy=True)
        assert result.success is False
        assert result.reason == "agent_busy"

    def test_claim_agent_busy_disabled(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.claim("1", "worker-A")
        result = board.claim("2", "worker-A", check_busy=False)
        assert result.success is True


# ── Release agent ────────────────────────────────────────────────────

class TestReleaseAgent:
    def test_release(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.claim("1", "worker-A", check_busy=False)
        board.claim("2", "worker-A", check_busy=False)
        released = board.release_agent("worker-A")
        assert len(released) == 2
        for t in released:
            assert t.owner is None
            assert t.status == "pending"

    def test_release_preserves_completed(self):
        board = make_board()
        board.create("A", "desc")
        board.claim("1", "worker-A")
        board.update("1", status="completed")
        released = board.release_agent("worker-A")
        assert len(released) == 0
        assert board.get("1").status == "completed"


# ── Query helpers ────────────────────────────────────────────────────

class TestQueryHelpers:
    def test_all_completed_empty(self):
        board = make_board()
        assert board.all_completed() is True

    def test_all_completed_false(self):
        board = make_board()
        board.create("A", "desc")
        assert board.all_completed() is False

    def test_all_completed_true(self):
        board = make_board()
        board.create("A", "desc")
        board.update("1", status="completed")
        assert board.all_completed() is True

    def test_list_available(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.create("C", "desc")
        available = board.list_available()
        ids = [t.id for t in available]
        assert "1" in ids
        assert "3" in ids
        assert "2" not in ids

    def test_list_available_excludes_owned(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.claim("1", "worker-A")
        available = board.list_available()
        ids = [t.id for t in available]
        assert "1" not in ids
        assert "2" in ids

    def test_get_summary(self):
        board = make_board()
        board.create("Audit code", "desc")
        board.create("Write tests", "desc", blocked_by=["1"])
        board.update("1", status="in_progress", owner="coder")
        summary = board.get_summary()
        assert "#1" in summary
        assert "#2" in summary
        assert "in_progress" in summary
        assert "@coder" in summary
        assert "blocked by #1" in summary


# ── state_dict / load_state_dict ─────────────────────────────────────

class TestStateDictRoundTrip:
    def test_roundtrip(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc", blocked_by=["1"])
        board.update("1", status="completed")

        state = board.state_dict()
        board2 = make_board()
        board2.load_state_dict(state)

        assert len(board2.list()) == 2
        assert board2.get("1").status == "completed"
        assert board2.get("2").blocked_by == ["1"]
        t3 = board2.create("C", "desc")
        assert t3.id == "3"

    def test_high_water_mark_preserved(self):
        board = make_board()
        board.create("A", "desc")
        board.create("B", "desc")
        board.delete("2")

        state = board.state_dict()
        board2 = make_board()
        board2.load_state_dict(state)

        t = board2.create("C", "desc")
        assert t.id == "3"


# ── File persistence ─────────────────────────────────────────────────

class TestFilePersistence:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.json"
            board = TaskBoard(store_path=path)
            board.create("A", "desc")
            board.create("B", "desc", blocked_by=["1"])
            board.update("1", status="completed")
            assert path.exists()

            board2 = TaskBoard(store_path=path)
            assert len(board2.list()) == 2
            assert board2.get("1").status == "completed"
            assert board2.get("2").blocked_by == ["1"]
            t3 = board2.create("C", "desc")
            assert t3.id == "3"

    def test_persist_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.json"
            board = TaskBoard(store_path=path)
            board.create("Test", "desc")

            data = json.loads(path.read_text("utf-8"))
            assert data["version"] == 1
            assert data["next_id"] == 2
            assert len(data["tasks"]) == 1
            assert data["tasks"][0]["subject"] == "Test"
