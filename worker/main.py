#!/usr/bin/env python3
"""
Blippy worker on GitHub Actions.

  outbound WebSocket → Blippy API
  Qwen Responses adapter (local) ← Codex model_provider boundary
  official Codex CLI (codex exec) = ONLY agent loop / tools / apply_patch / shell

Users never install Qwen or Codex for inference on their device.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.harness.workspace import Workspace
from worker.codex.run_codex_job import write_codex_home, run_codex_exec

API_URL = os.environ.get("BLIPPY_API_URL", "http://localhost:8000").rstrip("/")
WS_URL = API_URL.replace("https://", "wss://").replace("http://", "ws://") + "/workers/ws"
TOKEN = os.environ.get("BLIPPY_WORKER_TOKEN", "dev-worker-token")
WORKER_ID = os.environ.get("BLIPPY_WORKER_ID") or f"worker-{uuid.uuid4().hex[:8]}"
QWEN_PORT = int(os.environ.get("QWEN_RESPONSES_PORT", "8090"))
CODEX_BIN = os.environ.get("CODEX_BIN", str(Path.home() / ".local/bin/codex"))


def ensure_codex_binary() -> str:
    if Path(CODEX_BIN).exists():
        return CODEX_BIN
    script = ROOT / "worker/codex/install_codex.sh"
    subprocess.check_call(["bash", str(script)], cwd=str(ROOT))
    return CODEX_BIN


def start_qwen_responses_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["QWEN_RESPONSES_PORT"] = str(QWEN_PORT)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "worker.qwen.responses_server"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _pump():
        assert proc.stdout
        for line in proc.stdout:
            print(line.rstrip(), flush=True)

    threading.Thread(target=_pump, daemon=True).start()
    # wait until up
    import urllib.request

    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{QWEN_PORT}/health", timeout=2)
            print("[blippy] Qwen Responses adapter ready", flush=True)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("Qwen Responses server exited early")
            time.sleep(2)
    raise RuntimeError("Qwen Responses server did not become ready")


async def handle_job(ws, job_id: str, task: str, workspace_files: list) -> None:
    async def emit(ev: dict):
        await ws.send(json.dumps({"type": "event", "job_id": job_id, "event": ev}))

    await emit({"type": "agent.started", "message": "Codex agent starting"})
    await emit({"type": "thinking", "message": "Codex agent started"})

    ws_obj = Workspace(workspace_files)
    codex_home = Path(ws_obj.root) / ".blippy-codex-home"
    write_codex_home(codex_home, qwen_base=f"http://127.0.0.1:{QWEN_PORT}/v1")
    codex_bin = ensure_codex_binary()

    loop = asyncio.get_event_loop()
    log_buf: list[str] = []

    def on_line(line: str):
        log_buf.append(line)
        # Best-effort map Codex stdout → Blippy events
        low = line.lower()
        if "apply_patch" in low or "apply patch" in low:
            asyncio.run_coroutine_threadsafe(
                emit({"type": "thinking", "message": "Applying patch (Codex apply_patch)"}),
                loop,
            )
        elif "exec" in low or "shell" in low or "running" in low:
            asyncio.run_coroutine_threadsafe(
                emit({"type": "command_started", "command": line[:200]}),
                loop,
            )

    await emit({"type": "model.started", "message": "Qwen generating via Codex provider"})
    result = await loop.run_in_executor(
        None,
        lambda: run_codex_exec(
            Path(ws_obj.root),
            task,
            codex_bin=codex_bin,
            codex_home=codex_home,
            on_line=on_line,
        ),
    )
    changes = ws_obj.diff_changes()
    ws_obj.cleanup()

    if result.get("ok") or changes:
        await emit({"type": "agent.completed", "message": "Codex agent completed"})
        await ws.send(
            json.dumps(
                {
                    "type": "job_result",
                    "job_id": job_id,
                    "status": "completed",
                    "result": {
                        "summary": "codex exec finished",
                        "changes": changes,
                        "codex_ok": result.get("ok"),
                        "log_tail": (result.get("log") or "")[-2000:],
                    },
                    "error": None if (result.get("ok") or changes) else result.get("error"),
                }
            )
        )
    else:
        await ws.send(
            json.dumps(
                {
                    "type": "job_result",
                    "job_id": job_id,
                    "status": "failed",
                    "result": {"changes": changes, "log_tail": (result.get("log") or "")[-2000:]},
                    "error": result.get("error") or f"codex exit {result.get('exit_code')}",
                }
            )
        )


async def run_worker() -> None:
    import websockets

    print(f"[blippy] worker {WORKER_ID}", flush=True)
    print("[blippy] ensuring Codex CLI binary (upstream harness)...", flush=True)
    ensure_codex_binary()
    print("[blippy] starting Qwen Responses adapter (model boundary only)...", flush=True)
    qwen_proc = start_qwen_responses_server()

    uri = f"{WS_URL}?token={TOKEN}&worker_id={WORKER_ID}"
    backoff = 2
    try:
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
                        try:
                            await handle_job(ws, job_id, task, workspace.get("files") or [])
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
                print(f"[blippy] ws error: {e}; retry {backoff}s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)
    finally:
        qwen_proc.terminate()


if __name__ == "__main__":
    asyncio.run(run_worker())
