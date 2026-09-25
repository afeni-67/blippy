#!/usr/bin/env python3
"""
Blippy worker — runs on GitHub Actions.

Connects OUTBOUND to Blippy API over WebSocket.
Loads Qwen LOCALLY on this runner (users never install Qwen).
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Allow `python -m worker.main` or `python worker/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.qwen.model_runner import download_model, load_llm
from worker.qwen.inference import chat
from worker.harness.agent import run_coding_agent

API_URL = os.environ.get("BLIPPY_API_URL", "http://localhost:8000").rstrip("/")
WS_URL = API_URL.replace("https://", "wss://").replace("http://", "ws://") + "/workers/ws"
TOKEN = os.environ.get("BLIPPY_WORKER_TOKEN", "dev-worker-token")
WORKER_ID = os.environ.get("BLIPPY_WORKER_ID") or f"worker-{uuid.uuid4().hex[:8]}"


async def run_worker() -> None:
    import websockets
    import json

    print(f"[blippy] worker {WORKER_ID} starting", flush=True)
    print("[blippy] downloading/loading Qwen on this runner (not on user device)...", flush=True)
    path = download_model()
    llm = load_llm(path)
    print("[blippy] Qwen ready — connecting outbound to API", flush=True)

    uri = f"{WS_URL}?token={TOKEN}&worker_id={WORKER_ID}"
    backoff = 2
    while True:
        try:
            async with websockets.connect(uri, max_size=20_000_000, ping_interval=20) as ws:
                print(f"[blippy] connected to {API_URL}", flush=True)
                await ws.send(json.dumps({"type": "idle"}))
                backoff = 2
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "job_assigned":
                        continue
                    job_id = msg["job_id"]
                    task = msg.get("task") or ""
                    workspace = msg.get("workspace") or {"files": []}
                    print(f"[blippy] job {job_id}: {task[:80]}", flush=True)

                    async def on_event(ev: dict):
                        await ws.send(
                            json.dumps({"type": "event", "job_id": job_id, "event": ev})
                        )

                    def chat_fn(model, messages, max_tokens=512, temperature=0.2):
                        return chat(
                            model,
                            messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )

                    try:
                        result = await run_coding_agent(
                            llm,
                            task,
                            workspace.get("files") or [],
                            chat_fn=chat_fn,
                            on_event=on_event,
                            max_iters=int(os.environ.get("BLIPPY_MAX_ITERS", "12")),
                        )
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "job_result",
                                    "job_id": job_id,
                                    "status": "completed" if result.get("ok") else "failed",
                                    "result": {
                                        "summary": result.get("summary"),
                                        "changes": result.get("changes") or [],
                                        "iters": result.get("iters"),
                                    },
                                    "error": None if result.get("ok") else result.get("summary"),
                                }
                            )
                        )
                    except Exception as e:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "job_result",
                                    "job_id": job_id,
                                    "status": "failed",
                                    "result": None,
                                    "error": str(e),
                                }
                            )
                        )
                    await ws.send(json.dumps({"type": "idle"}))
        except Exception as e:
            print(f"[blippy] connection error: {e}; retry in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


if __name__ == "__main__":
    asyncio.run(run_worker())
