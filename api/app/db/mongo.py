"""MongoDB data-access layer for Blippy."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import MONGODB_URI

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client


def db() -> AsyncIOMotorDatabase:
    client = get_client()
    # Prefer DB name from URI path; fall back to nexgrade
    try:
        name = client.get_default_database().name  # type: ignore
        if name:
            return client[name]
    except Exception:
        pass
    return client["nexgrade"]


def now() -> datetime:
    return datetime.now(timezone.utc)


async def ensure_indexes() -> None:
    d = db()
    await d.jobs.create_index([("status", 1), ("created_at", 1)])
    await d.workers.create_index("worker_id", unique=True)
    await d.job_events.create_index([("job_id", 1), ("ts", 1)])


async def create_job(client_id: str, task: str, workspace: dict) -> dict:
    doc = {
        "client_id": client_id,
        "task": task,
        "workspace": workspace or {"files": []},
        "status": "queued",
        "worker_id": None,
        "created_at": now(),
        "updated_at": now(),
        "result": None,
        "error": None,
    }
    r = await db().jobs.insert_one(doc)
    doc["_id"] = r.inserted_id
    doc["job_id"] = str(r.inserted_id)
    return doc


async def get_job(job_id: str) -> Optional[dict]:
    try:
        doc = await db().jobs.find_one({"_id": ObjectId(job_id)})
    except Exception:
        return None
    if doc:
        doc["job_id"] = str(doc["_id"])
    return doc


async def update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = now()
    await db().jobs.update_one({"_id": ObjectId(job_id)}, {"$set": fields})


async def claim_next_job(worker_id: str) -> Optional[dict]:
    """Atomically assign one queued job to this worker."""
    doc = await db().jobs.find_one_and_update(
        {"status": "queued"},
        {
            "$set": {
                "status": "assigned",
                "worker_id": worker_id,
                "updated_at": now(),
            }
        },
        sort=[("created_at", 1)],
        return_document=True,
    )
    if doc:
        doc["job_id"] = str(doc["_id"])
    return doc


async def requeue_stale_jobs(worker_id: str) -> int:
    """If worker dies, requeue its assigned/running jobs."""
    r = await db().jobs.update_many(
        {"worker_id": worker_id, "status": {"$in": ["assigned", "running"]}},
        {
            "$set": {
                "status": "queued",
                "worker_id": None,
                "updated_at": now(),
            }
        },
    )
    return r.modified_count


async def upsert_worker(worker_id: str, status: str = "idle", meta: Optional[dict] = None) -> None:
    await db().workers.update_one(
        {"worker_id": worker_id},
        {
            "$set": {
                "worker_id": worker_id,
                "status": status,
                "last_seen": now(),
                "meta": meta or {},
            },
            "$setOnInsert": {"created_at": now()},
        },
        upsert=True,
    )


async def append_event(job_id: str, event: dict) -> dict:
    doc = {
        "job_id": job_id,
        "ts": now(),
        **event,
    }
    await db().job_events.insert_one(doc)
    return doc


async def list_events(job_id: str, after_ts: Optional[datetime] = None) -> List[dict]:
    q: Dict[str, Any] = {"job_id": job_id}
    if after_ts:
        q["ts"] = {"$gt": after_ts}
    cur = db().job_events.find(q).sort("ts", 1)
    return await cur.to_list(length=5000)
