"""In-process pub/sub for SSE + worker assignment (single API instance MVP)."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Optional, Set


class EventBus:
    def __init__(self) -> None:
        self._job_subs: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._idle_workers: asyncio.Queue = asyncio.Queue()
        self._worker_sockets: Dict[str, Any] = {}  # worker_id -> websocket

    def register_ws(self, worker_id: str, ws: Any) -> None:
        self._worker_sockets[worker_id] = ws

    def unregister_ws(self, worker_id: str) -> None:
        self._worker_sockets.pop(worker_id, None)

    def get_ws(self, worker_id: str) -> Optional[Any]:
        return self._worker_sockets.get(worker_id)

    async def publish_job(self, job_id: str, event: dict) -> None:
        dead = []
        for q in list(self._job_subs.get(job_id, set())):
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            self._job_subs[job_id].discard(q)

    def subscribe_job(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._job_subs[job_id].add(q)
        return q

    def unsubscribe_job(self, job_id: str, q: asyncio.Queue) -> None:
        self._job_subs[job_id].discard(q)

    async def mark_idle(self, worker_id: str) -> None:
        await self._idle_workers.put(worker_id)

    async def wait_idle_worker(self, timeout: float = 0.5) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._idle_workers.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


bus = EventBus()
