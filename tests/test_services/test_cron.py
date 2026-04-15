"""Unit tests for CronService (lagent/services/cron.py)."""

import asyncio
import json
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

from lagent.services.cron import (
    CronService, CronJob, Schedule, JobState, compute_next_run, _now_ms,
    _load_jobs, _save_jobs,
)


# ═══════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════

class TestSchedule:
    def test_defaults(self):
        s = Schedule()
        assert s.kind == "at"
        assert s.at is None
        assert s.every_seconds is None
        assert s.expr is None
        assert s.tz is None


class TestCronJob:
    def test_defaults(self):
        job = CronJob()
        assert len(job.id) == 8
        assert job.name == ""
        assert job.enabled is True
        assert isinstance(job.schedule, Schedule)
        assert isinstance(job.state, JobState)

    def test_to_dict_from_dict_roundtrip(self):
        job = CronJob(
            name="test-job",
            schedule=Schedule(kind="every", every_seconds=60),
            payload={"message": "hello", "channel": "cli"},
            state=JobState(next_run_at_ms=12345, consecutive_errors=2),
            delete_after_run=True,
        )
        d = job.to_dict()
        restored = CronJob.from_dict(d)
        assert restored.name == "test-job"
        assert restored.schedule.kind == "every"
        assert restored.schedule.every_seconds == 60
        assert restored.state.next_run_at_ms == 12345
        assert restored.state.consecutive_errors == 2
        assert restored.delete_after_run is True
        assert restored.payload["message"] == "hello"

    def test_from_dict_missing_fields(self):
        job = CronJob.from_dict({})
        assert job.name == ""
        assert job.enabled is True
        assert job.schedule.kind == "at"

    def test_from_dict_ignores_extra_keys(self):
        job = CronJob.from_dict({"name": "ok", "schedule": {"kind": "at", "bogus": 1}})
        assert job.name == "ok"
        assert job.schedule.kind == "at"


# ═══════════════════════════════════════════════════════════════════════
#  compute_next_run
# ═══════════════════════════════════════════════════════════════════════

class TestComputeNextRun:
    def test_at_future(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        s = Schedule(kind="at", at=future.isoformat())
        result = compute_next_run(s, _now_ms())
        assert result is not None
        assert result > _now_ms()

    def test_at_past_returns_none(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        s = Schedule(kind="at", at=past.isoformat())
        result = compute_next_run(s, _now_ms())
        assert result is None

    def test_at_missing_returns_none(self):
        s = Schedule(kind="at", at=None)
        assert compute_next_run(s, _now_ms()) is None

    def test_at_invalid_iso_returns_none(self):
        s = Schedule(kind="at", at="not-a-date")
        assert compute_next_run(s, _now_ms()) is None

    def test_every(self):
        s = Schedule(kind="every", every_seconds=30)
        now = _now_ms()
        result = compute_next_run(s, now)
        assert result == now + 30_000

    def test_every_zero_returns_none(self):
        s = Schedule(kind="every", every_seconds=0)
        assert compute_next_run(s, _now_ms()) is None

    def test_every_negative_returns_none(self):
        s = Schedule(kind="every", every_seconds=-5)
        assert compute_next_run(s, _now_ms()) is None

    def test_every_missing_returns_none(self):
        s = Schedule(kind="every", every_seconds=None)
        assert compute_next_run(s, _now_ms()) is None

    def test_cron_without_expr_returns_none(self):
        s = Schedule(kind="cron", expr=None)
        assert compute_next_run(s, _now_ms()) is None

    def test_cron_valid(self):
        pytest.importorskip("croniter")
        s = Schedule(kind="cron", expr="* * * * *")  # every minute
        result = compute_next_run(s, _now_ms())
        assert result is not None
        assert result > _now_ms()

    def test_unknown_kind_returns_none(self):
        s = Schedule(kind="bogus")
        assert compute_next_run(s, _now_ms()) is None


# ═══════════════════════════════════════════════════════════════════════
#  PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════

class TestPersistenceHelpers:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            jobs = [
                CronJob(name="j1", schedule=Schedule(kind="every", every_seconds=10)),
                CronJob(name="j2", schedule=Schedule(kind="at", at="2030-01-01T00:00:00Z")),
            ]
            _save_jobs(jobs, path)
            assert path.exists()

            loaded = _load_jobs(path)
            assert len(loaded) == 2
            assert loaded[0].name == "j1"
            assert loaded[1].name == "j2"

    def test_load_missing_file(self):
        result = _load_jobs(Path("/nonexistent/path.json"))
        assert result == []

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text("not json", "utf-8")
            result = _load_jobs(path)
            assert result == []

    def test_load_wrong_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text(json.dumps({"version": 99, "jobs": []}), "utf-8")
            result = _load_jobs(path)
            assert result == []

    def test_save_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            _save_jobs([CronJob(name="x")], path)
            data = json.loads(path.read_text("utf-8"))
            assert data["version"] == 1
            assert len(data["jobs"]) == 1
            assert data["jobs"][0]["name"] == "x"


# ═══════════════════════════════════════════════════════════════════════
#  CRON SERVICE
# ═══════════════════════════════════════════════════════════════════════

class TestCronServiceCRUD:
    def _make_service(self, tmpdir) -> CronService:
        path = Path(tmpdir) / "jobs.json"
        return CronService(store_path=path)

    def test_add_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            job = svc.add_job(
                name="test",
                schedule=Schedule(kind="every", every_seconds=60),
                message="hello",
            )
            assert job.name == "test"
            assert job.payload["message"] == "hello"
            assert len(svc.list_jobs(include_disabled=True)) == 1

    def test_add_job_validates_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            with pytest.raises(ValueError, match="'at'"):
                svc.add_job(
                    name="bad",
                    schedule=Schedule(kind="at", at=None),
                    message="x",
                )

    def test_add_job_validates_every(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            with pytest.raises(ValueError, match="every_seconds"):
                svc.add_job(
                    name="bad",
                    schedule=Schedule(kind="every", every_seconds=0),
                    message="x",
                )

    def test_add_job_validates_cron(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            with pytest.raises(ValueError, match="'expr'"):
                svc.add_job(
                    name="bad",
                    schedule=Schedule(kind="cron", expr=None),
                    message="x",
                )

    def test_remove_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            job = svc.add_job(
                name="test",
                schedule=Schedule(kind="every", every_seconds=60),
                message="hello",
            )
            assert svc.remove_job(job.id) is True
            assert len(svc.list_jobs(include_disabled=True)) == 0

    def test_remove_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            assert svc.remove_job("nonexistent") is False

    def test_get_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            job = svc.add_job(
                name="test",
                schedule=Schedule(kind="every", every_seconds=60),
                message="hello",
            )
            found = svc.get_job(job.id)
            assert found is not None
            assert found.name == "test"

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            assert svc.get_job("nope") is None

    def test_list_jobs_filters_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            j1 = svc.add_job(
                name="enabled",
                schedule=Schedule(kind="every", every_seconds=60),
                message="a",
            )
            j2 = svc.add_job(
                name="disabled",
                schedule=Schedule(kind="every", every_seconds=60),
                message="b",
            )
            j2.enabled = False
            assert len(svc.list_jobs(include_disabled=False)) == 1
            assert len(svc.list_jobs(include_disabled=True)) == 2

    def test_list_jobs_sorted_by_next_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            j1 = svc.add_job(
                name="later",
                schedule=Schedule(kind="every", every_seconds=120),
                message="a",
            )
            j2 = svc.add_job(
                name="sooner",
                schedule=Schedule(kind="every", every_seconds=10),
                message="b",
            )
            jobs = svc.list_jobs()
            assert jobs[0].name == "sooner"
            assert jobs[1].name == "later"


class TestCronServicePersistence:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc1 = CronService(store_path=path)
            svc1.add_job(
                name="persist-test",
                schedule=Schedule(kind="every", every_seconds=30),
                message="hi",
            )
            # Load in a new service instance
            svc2 = CronService(store_path=path)
            loaded = _load_jobs(path)
            assert len(loaded) == 1
            assert loaded[0].name == "persist-test"


class TestCronServiceStateDict:
    def test_state_dict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path)
            svc.add_job(
                name="j1",
                schedule=Schedule(kind="every", every_seconds=60),
                message="hello",
            )
            svc.add_job(
                name="j2",
                schedule=Schedule(kind="every", every_seconds=120),
                message="world",
            )

            state = svc.state_dict()
            assert state["version"] == 1
            assert len(state["jobs"]) == 2

            # Restore into a fresh service
            path2 = Path(tmpdir) / "jobs2.json"
            svc2 = CronService(store_path=path2)
            svc2.load_state_dict(state)

            jobs = svc2.list_jobs(include_disabled=True)
            assert len(jobs) == 2
            names = {j.name for j in jobs}
            assert "j1" in names
            assert "j2" in names


# ═══════════════════════════════════════════════════════════════════════
#  ASYNC TIMER EXECUTION (end-to-end)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestCronTimerExecution:
    """Test that jobs actually fire via the async timer engine."""

    async def test_every_job_fires(self):
        """Add a 1-second recurring job, verify on_job is called."""
        fired = []

        async def on_job(job):
            fired.append(job.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path, on_job=on_job)
            await svc.start()

            svc.add_job(
                name="fast-repeat",
                schedule=Schedule(kind="every", every_seconds=1),
                message="tick",
            )

            # Wait long enough for at least 1 fire
            # _MIN_REFIRE_GAP_S=2.0, so timer fires at ~2s, then job is due
            await asyncio.sleep(5)
            svc.stop()

            assert len(fired) >= 1
            assert fired[0] == "fast-repeat"

    async def test_at_job_fires_once(self):
        """Add a one-shot 'at' job 2s in the future, verify it fires once."""
        fired = []

        async def on_job(job):
            fired.append(job.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path, on_job=on_job)
            await svc.start()

            future = (
                datetime.now(timezone.utc) + timedelta(seconds=2)
            ).isoformat()
            svc.add_job(
                name="one-shot",
                schedule=Schedule(kind="at", at=future),
                message="boom",
                delete_after_run=True,
            )

            await asyncio.sleep(5)
            svc.stop()

            assert len(fired) == 1
            assert fired[0] == "one-shot"
            # Job should be deleted after run
            assert len(svc.list_jobs(include_disabled=True)) == 0

    async def test_job_error_records_state(self):
        """A failing on_job should record error state."""
        async def on_job(job):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path, on_job=on_job)
            await svc.start()

            svc.add_job(
                name="fail-job",
                schedule=Schedule(kind="every", every_seconds=1),
                message="fail",
            )

            await asyncio.sleep(5)
            svc.stop()

            job = svc.list_jobs(include_disabled=True)[0]
            assert job.state.last_status == "error"
            assert job.state.last_error == "boom"
            assert job.state.consecutive_errors >= 1

    async def test_no_on_job_callback(self):
        """Service without on_job should not crash when jobs fire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path, on_job=None)
            await svc.start()

            svc.add_job(
                name="silent",
                schedule=Schedule(kind="every", every_seconds=1),
                message="noop",
            )

            await asyncio.sleep(4)
            svc.stop()

            job = svc.list_jobs(include_disabled=True)[0]
            assert job.state.last_status == "ok"

    async def test_stop_prevents_further_fires(self):
        """After stop(), no more jobs should fire."""
        fired = []

        async def on_job(job):
            fired.append(job.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            svc = CronService(store_path=path, on_job=on_job)
            await svc.start()

            svc.add_job(
                name="stoppable",
                schedule=Schedule(kind="every", every_seconds=1),
                message="tick",
            )

            await asyncio.sleep(4)
            count_before_stop = len(fired)
            svc.stop()

            await asyncio.sleep(3)
            assert len(fired) == count_before_stop  # no new fires


if __name__ == "__main__":
    async def _run_all():
        t = TestCronTimerExecution()
        tests = [
            ("every_job_fires", t.test_every_job_fires),
            ("at_job_fires_once", t.test_at_job_fires_once),
            ("job_error_records_state", t.test_job_error_records_state),
            ("no_on_job_callback", t.test_no_on_job_callback),
            ("stop_prevents_further_fires", t.test_stop_prevents_further_fires),
        ]
        for name, fn in tests:
            print(f"Running {name}...", end=" ", flush=True)
            try:
                await fn()
                print("PASSED")
            except Exception as e:
                print(f"FAILED: {e}")

    asyncio.run(_run_all())