"""Loop-agnostic async priority lock for sharing one GPU between live dictation and batch jobs."""

from __future__ import annotations

import asyncio
import heapq
from contextlib import asynccontextmanager
from typing import AsyncIterator

PRIORITY_LIVE_DICTATION = 0
PRIORITY_BATCH_SEGMENTS = 10


class AsyncPriorityLock:
    """Priority-ordered asyncio lock (lower priority integer acquires first, FIFO within ties).

    Safe across multiple event loops (e.g. FastAPI TestClient runs) because it binds
    waiter futures to the currently running loop only when contention occurs.
    """

    def __init__(self) -> None:
        self._held = False
        self._seq = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    @property
    def locked(self) -> bool:
        return self._held

    @property
    def pending_count(self) -> int:
        return sum(1 for _, _, fut in self._waiters if not fut.done())

    async def acquire(self, priority: int = PRIORITY_LIVE_DICTATION) -> None:
        if not self._held:
            self._held = True
            return

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (int(priority), self._seq, fut))
        try:
            await fut
        except BaseException:
            if not fut.done():
                fut.cancel()
            elif not fut.cancelled():
                # We were handed ownership right as we were cancelled; pass it on.
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            _prio, _seq, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._held = False

    @asynccontextmanager
    async def hold(self, priority: int = PRIORITY_LIVE_DICTATION) -> AsyncIterator[None]:
        await self.acquire(priority=priority)
        try:
            yield
        finally:
            self.release()
