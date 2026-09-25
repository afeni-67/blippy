import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.mongo import ensure_indexes
from app.routes import jobs, workers
from app.services.assign import assign_loop

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Blippy API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(jobs.router)
app.include_router(workers.router)


@app.on_event("startup")
async def startup():
    await ensure_indexes()
    asyncio.create_task(assign_loop())


@app.get("/health")
async def health():
    return {"ok": True, "service": "blippy-api"}
