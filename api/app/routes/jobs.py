from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import CLIENT_TOKEN
from app.db import mongo
from app.services.bus import bus
from app.services.assign import try_assign_one

router = APIRouter()


def _auth_client(authorization: Optional[str]) -> None:
    if not authorization:
        # allow open MVP if client token is default in local dev
        return
    token = authorization.replace("Bearer ", "").strip()
    if token and token != CLIENT_TOKEN and CLIENT_TOKEN != "dev-client-token":
        raise HTTPException(401, "invalid client token")


class WorkspaceFile(BaseModel):
    path: str
    content: str = ""


class Workspace(BaseModel):
    files: List[WorkspaceFile] = Field(default_factory=list)


class CreateJobBody(BaseModel):
    client_id: str = "anonymous"
    task: str
    workspace: Workspace = Field(default_factory=Workspace)


@router.post("/jobs")
async def create_job(body: CreateJobBody, authorization: Optional[str] = Header(None)):
    _auth_client(authorization)
    if not body.task.strip():
        raise HTTPException(400, "task required")
    ws = {"files": [f.model_dump() for f in body.workspace.files]}
    job = await mongo.create_job(body.client_id, body.task.strip(), ws)
    await mongo.append_event(job["job_id"], {"type": "job_queued"})
    await bus.publish_job(job["job_id"], {"type": "job_queued"})
    # try immediate assignment
    asyncio.create_task(try_assign_one())
    return {"job_id": job["job_id"], "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await mongo.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "task": job.get("task"),
        "worker_id": job.get("worker_id"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/events")
async def job_events_sse(job_id: str, request: Request):
    job = await mongo.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def gen():
        # replay history
        for ev in await mongo.list_events(job_id):
            payload = {k: v for k, v in ev.items() if k not in ("_id", "ts")}
            if "ts" in ev:
                payload["ts"] = ev["ts"].isoformat()
            yield f"data: {json.dumps(payload, default=str)}\n\n"

        q = bus.subscribe_job(job_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    # if job finished, end stream
                    j = await mongo.get_job(job_id)
                    if j and j.get("status") in ("completed", "failed", "cancelled"):
                        break
                    continue
                yield f"data: {json.dumps(ev, default=str)}\n\n"
                if ev.get("type") in ("job_completed", "job_failed"):
                    break
        finally:
            bus.unsubscribe_job(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream")
