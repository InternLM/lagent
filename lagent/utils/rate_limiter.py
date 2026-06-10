"""Async rate limiters used across lagent (mcp client, custom actions, RL recipes).

Single source of truth — both the in-process limiter class and the
process-shared registry helper live here so that callers don't fork their own
implementations.
"""

from __future__ import annotations
import asyncio
import threading
import time
from collections import deque
from typing import Deque


class FairAsyncTokenBucket:
    """FIFO async token bucket.

    A single drainer task wakes waiters in arrival order, so high-concurrency
    callers don't form a thundering herd around `asyncio.sleep`.
    """

    def __init__(self, rate_limit: float, capacity: float | None = None):
        """rate_limit: tokens per second; capacity: burst size (defaults to rate_limit)."""
        self.rate_limit = float(rate_limit)
        self.capacity = float(capacity) if capacity is not None else float(rate_limit)
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: Deque[asyncio.Future] = deque()
        self._drainer_running = False

    def _refill_unlocked(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_update
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_limit)
        self.last_update = now

    async def _drain_waiters(self) -> None:
        try:
            while True:
                fut_to_wake: asyncio.Future | None = None
                sleep_time: float | None = None

                async with self._lock:
                    self._refill_unlocked()

                    if not self._waiters:
                        self._drainer_running = False
                        return

                    if self.tokens >= 1:
                        self.tokens -= 1
                        fut_to_wake = self._waiters.popleft()
                        sleep_time = 0.0
                    else:
                        missing = 1.0 - self.tokens
                        sleep_time = max(0.0, missing / self.rate_limit)

                if fut_to_wake is not None and not fut_to_wake.done():
                    fut_to_wake.set_result(None)

                if sleep_time == 0.0:
                    continue

                await asyncio.sleep(sleep_time)
        finally:
            async with self._lock:
                self._drainer_running = False

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()

        async with self._lock:
            self._refill_unlocked()

            if self.tokens >= 1 and not self._waiters:
                self.tokens -= 1
                return

            fut = loop.create_future()
            self._waiters.append(fut)

            if not self._drainer_running:
                self._drainer_running = True
                asyncio.create_task(self._drain_waiters())

        await fut


_SHARED_LIMITERS: dict[str, FairAsyncTokenBucket] = {}
_SHARED_LIMITERS_LOCK = threading.Lock()


def get_shared_async_token_bucket(
    key: str,
    rate_limit: float,
    capacity: float | None = None,
) -> FairAsyncTokenBucket:
    """Return a process-shared FairAsyncTokenBucket registered under `key`.

    First call for a key wins the rate/capacity; subsequent calls reuse the
    existing limiter (their rate/capacity arguments are ignored). Limiters live
    for the lifetime of the process and are scoped per-process — Ray actors are
    separate processes and therefore each maintain their own registry.
    """
    if not key:
        raise ValueError('rate limiter key must be non-empty')
    with _SHARED_LIMITERS_LOCK:
        limiter = _SHARED_LIMITERS.get(key)
        if limiter is None:
            limiter = FairAsyncTokenBucket(rate_limit=rate_limit, capacity=capacity)
            _SHARED_LIMITERS[key] = limiter
        return limiter


__all__ = ['FairAsyncTokenBucket', 'get_shared_async_token_bucket']
