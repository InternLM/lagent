import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from lagent.serving.sandbox.providers.clusterx import ClusterXProvider


@pytest.fixture
def sandbox(tmp_path):
    state = tmp_path / "sandbox"
    state.mkdir()
    provider = ClusterXProvider(state_dir=tmp_path)
    client = SimpleNamespace(aclose=AsyncMock())
    provider._jobs["job-1"] = {
        "state_dir": str(state),
        "cluster": "aliyun",
        "client": client,
    }
    provider._ensure_runtime = AsyncMock(return_value={})
    return provider, client, state


@pytest.mark.parametrize("status", ["Stopped", "Failed", "Succeeded", "terminated", "deleted"])
def test_delete_is_idempotent_for_terminal_job(sandbox, status):
    provider, client, state = sandbox
    provider._rpc = AsyncMock(side_effect=[RuntimeError("stop rejected"), {"status": status}])

    asyncio.run(provider.delete("job-1"))

    assert provider._rpc.await_args_list == [
        call("stop", cluster_name="aliyun", job_id="job-1"),
        call("get", cluster_name="aliyun", job_id="job-1"),
    ]
    client.aclose.assert_awaited_once()
    assert not state.exists()
    assert provider.list() == []


@pytest.mark.parametrize("lookup", [{"status": "Running"}, {}, None, [], "Stopped", RuntimeError("lookup failed")])
def test_delete_preserves_unconfirmed_job(sandbox, lookup):
    provider, client, state = sandbox
    error = RuntimeError("Job can't be stopped")
    provider._rpc = AsyncMock(side_effect=[error, lookup])

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(provider.delete("job-1"))

    assert caught.value is error
    client.aclose.assert_awaited_once()
    assert state.exists()
    assert provider.list()[0]["job_id"] == "job-1"


def test_delete_normal_stop_needs_no_status_query(sandbox):
    provider, client, state = sandbox
    provider._rpc = AsyncMock(return_value={"stopped": "job-1"})

    asyncio.run(provider.delete("job-1"))

    provider._rpc.assert_awaited_once_with("stop", cluster_name="aliyun", job_id="job-1")
    client.aclose.assert_awaited_once()
    assert not state.exists()
