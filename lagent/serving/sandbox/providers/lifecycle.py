"""Allocation rules shared by the Gateway and direct DBox adapters.

Request transport and cancellation of rollback remain adapter responsibilities.
This module owns the ordering that makes allocated IDs recoverable, and the
readiness rules applied after that evidence has been published.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager

from .base import AllocationCallback, SandboxEnvironment, SandboxStatus

_TERMINAL_STATES = frozenset(
    {
        'error',
        'failed',
        'deleted',
        'deleting',
        'terminated',
        'terminating',
        'expired',
        'stopped',
        'stopping',
        'paused',
        'succeeded',
        'completed',
    }
)


@contextmanager
def allocation_preflight() -> Iterator[None]:
    """Mark only local validation failures, before any create request is sent."""
    try:
        yield
    except Exception as exc:
        setattr(exc, 'allocation_state', 'not_allocated')
        raise


@asynccontextmanager
async def allocated_sandbox(
    sandbox_id: str,
    owned_ids: set[str],
    notify: AllocationCallback | None,
    rollback: Callable[[str], Awaitable[None]],
) -> AsyncIterator[None]:
    """Remember and publish an ID before readiness; preserve the primary error.

    The adapter's delete implementation removes ownership only on success.
    Its rollback callable retains its own transport cancellation behavior.
    """
    owned_ids.add(sandbox_id)
    try:
        if notify is not None:
            await notify(sandbox_id)
        yield
    except BaseException:
        try:
            await rollback(sandbox_id)
        except BaseException:
            # A failed/cancelled rollback cannot replace the creation failure.
            # The owned ID and the caller's earlier notification permit retry.
            pass
        raise


async def wait_ready(
    sandbox_id: str,
    initial: SandboxEnvironment,
    get_status: Callable[[str], Awaitable[SandboxStatus]],
    *,
    timeout: float,
    poll_interval: float,
) -> SandboxEnvironment:
    """Wait for ready, rejecting terminal states under one readiness budget."""
    deadline = time.monotonic() + timeout
    info = initial
    while info.get('status') != 'ready':
        if info.get('status') in _TERMINAL_STATES:
            raise RuntimeError('Sandbox reached a terminal state before ready')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Sandbox platform readiness timed out')
        await asyncio.sleep(min(poll_interval, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Sandbox platform readiness timed out')
        try:
            response = await asyncio.wait_for(get_status(sandbox_id), timeout=remaining)
        except asyncio.TimeoutError:
            raise TimeoutError('Sandbox platform readiness timed out') from None
        info = response['env']
    return info
