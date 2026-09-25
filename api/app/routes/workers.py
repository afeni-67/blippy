from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.config import WORKER_TOKEN
from app.db import mongo
from app.services.bus import bus
from app.services.assign import try_assign_one

router = APIRouter()
logger = logging.getLogger("blippy.workers")


def _check_token(token: Optional[str]) -> None:
    if not token or token != WORKER_TOKEN:
        raise HTTPException(401, "invalid worker token")


class RegisterBody(BaseModel):
    worker_id: str
    meta: dict = {}


@router.post("/workers/register")
async def register(body: RegisterBody, x_worker_token: Optional[str] = Header(None)):
    _check_token(x_worker_token)
    await mongo.upsert_worker(body.worker_id, status="idle", meta=body.meta)
    return {"ok": True, "worker_id": body.worker_id}


@router.post("/workers/heartbeat")
async def heartbeat(body: RegisterBody, x_worker_token: Optional[str] = Header(None)):
    _check_token(x_worker_token)
    await mongo.upsert_worker(body.worker_id, status=body.meta.get("status", "idle"), meta=body.meta)
    return {"ok": True}


@router.websocket("/workers/ws")
async def worker_ws(websocket: WebSocket):
    token = websocket.query_params.get("token")
    worker_id = websocket.query_params.get("worker_id")
    if token != WORKER_TOKEN or not worker_id:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    bus.register_ws(worker_id, websocket)
    await mongo.upsert_worker(worker_id, status="idle", meta={})
    await bus.mark_idle(worker_id)
    logger.info("worker connected %s", worker_id)
    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            if mtype == "idle":
                await mongo.upsert_worker(worker_id, status="idle")
                await bus.mark_idle(worker_id)
                await try_assign_one()
            elif mtype == "heartbeat":
                await mongo.upsert_worker(worker_id, status=msg.get("status", "idle"), meta=msg.get("meta") or {})
            elif mtype == "event":
                job_id = msg.get("job_id")
                event = msg.get("event") or {}
                if job_id and event:
                    await mongo.append_event(job_id, event)
                    await bus.publish_job(job_id, event)
            elif mtype == "job_result":
                job_id = msg.get("job_id")
                status = msg.get("status", "completed")
                result = msg.get("result")
                error = msg.get("error")
                if job_id:
                    await mongo.update_job(
                        job_id,
                        status=status,
                        result=result,
                        error=error,
                    )
                    ev = {
                        "type": "job_completed" if status == "completed" else "job_failed",
                        "result": result,
                        "error": error,
                    }
                    await mongo.append_event(job_id, ev)
                    await bus.publish_job(job_id, ev)
                await mongo.upsert_worker(worker_id, status="idle")
                await bus.mark_idle(worker_id)
                await try_assign_one()
    except WebSocketDisconnect:
        logger.info("worker disconnected %s", worker_id)
    except Exception:
        logger.exception("worker ws error %s", worker_id)
    finally:
        bus.unregister_ws(worker_id)
        await mongo.requeue_stale_jobs(worker_id)
        await mongo.upsert_worker(worker_id, status="offline")
