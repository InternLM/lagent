"""Unit tests for CronAction (lagent/actions/cron.py)."""

import os
import sys
import tempfile
import types
from datetime import datetime, timezone, timedelta
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

from lagent.actions.cron import CronAction, AsyncCronAction
from lagent.schema import ActionStatusCode
from lagent.services.cron import CronService


def _make_action(tmpdir) -> CronAction:
    path = Path(tmpdir) / "jobs.json"
    svc = CronService(store_path=path)
    return CronAction(cron_service=svc, channel="test", chat_id="c1")


# ── add ──────────────────────────────────────────────────────────────

class TestCronActionAdd:
    def test_add_every(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="repeat",
                message="do stuff",
                schedule_kind="every",
                every_seconds=30.0,
            )
            assert result.result is not None
            assert "repeat" in result.result[0]["content"]

    def test_add_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            result = action.add(
                name="once",
                message="remind",
                schedule_kind="at",
                at=future,
            )
            assert result.result is not None
            assert "once" in result.result[0]["content"]

    def test_add_cron(self):
        pytest.importorskip("croniter")
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="daily",
                message="report",
                schedule_kind="cron",
                cron_expr="0 9 * * *",
                timezone="UTC",
            )
            assert result.result is not None
            assert "daily" in result.result[0]["content"]

    def test_add_invalid_kind_returns_args_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="bad",
                message="x",
                schedule_kind="invalid",
            )
            assert result.state == ActionStatusCode.ARGS_ERROR
            assert "Invalid schedule_kind" in result.errmsg

    def test_add_at_without_datetime_returns_api_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="bad",
                message="x",
                schedule_kind="at",
            )
            assert result.state == ActionStatusCode.API_ERROR

    def test_add_every_without_seconds_returns_api_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="bad",
                message="x",
                schedule_kind="every",
            )
            assert result.state == ActionStatusCode.API_ERROR

    def test_add_cron_without_expr_returns_api_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.add(
                name="bad",
                message="x",
                schedule_kind="cron",
            )
            assert result.state == ActionStatusCode.API_ERROR

    def test_add_populates_channel_and_chat_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(
                name="j1",
                message="hello",
                schedule_kind="every",
                every_seconds=60.0,
            )
            job = action._cron.list_jobs(include_disabled=True)[0]
            assert job.payload["channel"] == "test"
            assert job.payload["chat_id"] == "c1"

    def test_add_at_sets_delete_after_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            action.add(
                name="once",
                message="x",
                schedule_kind="at",
                at=future,
            )
            job = action._cron.list_jobs(include_disabled=True)[0]
            assert job.delete_after_run is True

    def test_add_every_does_not_set_delete_after_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(
                name="repeat",
                message="x",
                schedule_kind="every",
                every_seconds=10.0,
            )
            job = action._cron.list_jobs(include_disabled=True)[0]
            assert job.delete_after_run is False


# ── list ─────────────────────────────────────────────────────────────

class TestCronActionList:
    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.list()
            assert "No active" in result.result[0]["content"]

    def test_list_with_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(name="j1", message="a", schedule_kind="every", every_seconds=10.0)
            action.add(name="j2", message="b", schedule_kind="every", every_seconds=20.0)
            result = action.list()
            content = result.result[0]["content"]
            assert "j1" in content
            assert "j2" in content

    def test_list_hides_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(name="j1", message="a", schedule_kind="every", every_seconds=10.0)
            # Manually disable the job
            action._cron.list_jobs(include_disabled=True)[0].enabled = False
            result = action.list()
            assert "No active" in result.result[0]["content"]

    def test_list_shows_schedule_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(name="j1", message="a", schedule_kind="every", every_seconds=30.0)
            result = action.list()
            content = result.result[0]["content"]
            assert "every 30.0s" in content


# ── remove ───────────────────────────────────────────────────────────

class TestCronActionRemove:
    def test_remove_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            action.add(name="j1", message="a", schedule_kind="every", every_seconds=10.0)
            job_id = action._cron.list_jobs(include_disabled=True)[0].id
            result = action.remove(job_id=job_id)
            assert "removed" in result.result[0]["content"]
            assert len(action._cron.list_jobs(include_disabled=True)) == 0

    def test_remove_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            result = action.remove(job_id="nope")
            assert result.state == ActionStatusCode.API_ERROR
            assert "not found" in result.errmsg


# ── toolkit metadata ─────────────────────────────────────────────────

class TestCronActionMeta:
    def test_is_toolkit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            assert action.is_toolkit

    def test_has_three_apis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = _make_action(tmpdir)
            desc = action.description
            api_names = {api["name"] for api in desc["api_list"]}
            assert api_names == {"add", "list", "remove"}


# ── async variant ────────────────────────────────────────────────────

class TestAsyncCronAction:
    def test_instantiation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path)
            action = AsyncCronAction(cron_service=svc)
            assert action._cron is svc
