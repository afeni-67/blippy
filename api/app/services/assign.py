"""Assign queued jobs to idle workers over their persistent WebSocket."""
from __future__ import annotations

import asyncio
import logging

from app.db import mongo
from app.services.bus import bus

logger = logging.getLogger("blippy.assign")


async def try_assign_one() -> bool:
    worker_id = await bus.wait_idle_worker(timeout=0.05)
    if not worker_id:
        return False
    ws = bus.get_ws(worker_id)
    if not ws:
        return False
    job = await mongo.claim_next_job(worker_id)
    if not job:
        # no job — put worker back idle
        await bus.mark_idle(worker_id)
        return False
    job_id = job["job_id"]
    payload = {
        "type": "job_assigned",
        "job_id": job_id,
        "task": job.get("task"),
        "workspace": job.get("workspace") or {"files": []},
    }
    try:
        await ws.send_json(payload)
        await mongo.update_job(job_id, status="running")
        ev = {"type": "worker_assigned", "worker_id": worker_id}
        await mongo.append_event(job_id, ev)
        await bus.publish_job(job_id, ev)
        await mongo.upsert_worker(worker_id, status="busy")
        logger.info("assigned job %s -> worker %s", job_id, worker_id)
        return True
    except Exception as e:
        logger.exception("assign failed: %s", e)
        await mongo.update_job(job_id, status="queued", worker_id=None)
        return False


async def assign_loop() -> None:
    while True:
        try:
            did = await try_assign_one()
            if not did:
                await asyncio.sleep(0.25)
        except Exception:
            logger.exception("assign_loop error")
            await asyncio.sleep(1.0)
