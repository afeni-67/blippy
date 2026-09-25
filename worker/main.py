#!/usr/bin/env python3
"""
Blippy worker on GitHub Actions.

  outbound WebSocket → Blippy API
  local Qwen GGUF via worker/qwen/chat_server.py (model boundary only)
  official Qwen Code CLI (`qwen -p`, headless) = ONLY agent loop / tools

Users never install Qwen or Qwen Code for inference on their device.
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
from worker.qwencode.run_qwencode_job import write_qwen_home, run_qwen_exec

API_URL = os.environ.get("BLIPPY_API_URL", "http://localhost:8000").rstrip("/")
WS_URL = API_URL.replace("https://", "wss://").replace("http://", "ws://") + "/workers/ws"
TOKEN = os.environ.get("BLIPPY_WORKER_TOKEN", "dev-worker-token")
WORKER_ID = os.environ.get("BLIPPY_WORKER_ID") or f"worker-{uuid.uuid4().hex[:8]}"
QWEN_CHAT_PORT = int(os.environ.get("QWEN_CHAT_PORT", "8091"))
QWEN_BIN = os.environ.get("QWEN_BIN", "qwen")


def ensure_qwen_binary() -> str:
    found = QWEN_BIN if os.path.isabs(QWEN_BIN) else None
    if found and Path(found).exists():
        return found
    import shutil as _shutil

    path = _shutil.which(QWEN_BIN)
    if path:
        return path
    script = ROOT / "worker/qwencode/install_qwencode.sh"
    subprocess.check_call(["bash", str(script)], cwd=str(ROOT))
    path = _shutil.which("qwen") or QWEN_BIN
    return path


def start_qwen_chat_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["QWEN_CHAT_PORT"] = str(QWEN_CHAT_PORT)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "worker.qwen.chat_server"],
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
    import urllib.request

    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{QWEN_CHAT_PORT}/health", timeout=2)
            print("[blippy] Qwen chat adapter ready", flush=True)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("Qwen chat server exited early")
            time.sleep(2)
    raise RuntimeError("Qwen chat server did not become ready")


async def handle_job(ws, job_id: str, task: str, workspace_files: list) -> None:
    async def emit(ev: dict):
        await ws.send(json.dumps({"type": "event", "job_id": job_id, "event": ev}))

    await emit({"type": "agent.started", "message": "Qwen Code agent starting"})
    await emit({"type": "thinking", "message": "Qwen Code agent started"})

    ws_obj = Workspace(workspace_files)
    qwen_home = Path(ws_obj.root) / ".blippy-qwen-home"
    write_qwen_home(qwen_home, qwen_base=f"http://127.0.0.1:{QWEN_CHAT_PORT}/v1")
    qwen_bin = ensure_qwen_binary()

    loop = asyncio.get_event_loop()
    log_buf: list[str] = []

    def on_line(line: str):
        log_buf.append(line)
        low = line.lower()
        if any(k in low for k in ("write_file", "edit", "apply", "creating", "writing")):
            asyncio.run_coroutine_threadsafe(
                emit({"type": "thinking", "message": line[:300]}), loop)
        elif any(k in low for k in ("exec", "shell", "running", "bash")):
            asyncio.run_coroutine_threadsafe(
                emit({"type": "command_started", "command": line[:300]}), loop)

    await emit({"type": "model.started", "message": "Qwen generating via Qwen Code provider"})
    result = await loop.run_in_executor(
        None,
        lambda: run_qwen_exec(
            Path(ws_obj.root), task, qwen_bin=qwen_bin, qwen_home=qwen_home,
            on_line=on_line,
        ),
    )
    changes = ws_obj.diff_changes()
    ws_obj.cleanup()

    if result.get("ok") or changes:
        await emit({"type": "agent.completed", "message": "Qwen Code agent completed"})
        await ws.send(json.dumps({
            "type": "job_result", "job_id": job_id, "status": "completed",
            "result": {"summary": "qwen exec finished", "changes": changes,
                       "qwen_ok": result.get("ok"),
                       "log_tail": (result.get("log") or "")[-2000:]},
            "error": None if (result.get("ok") or changes) else result.get("error"),
        }))
    else:
        await ws.send(json.dumps({
            "type": "job_result", "job_id": job_id, "status": "failed",
            "result": {"changes": changes, "log_tail": (result.get("log") or "")[-2000:]},
            "error": result.get("error") or f"qwen exit {result.get('exit_code')}",
        }))


async def run_worker() -> None:
    import websockets

    print(f"[blippy] worker {WORKER_ID}", flush=True)
    print("[blippy] ensuring Qwen Code CLI binary (upstream harness)...", flush=True)
    ensure_qwen_binary()
    print("[blippy] starting Qwen chat adapter (model boundary only)...", flush=True)
    qwen_proc = start_qwen_chat_server()

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
                            await ws.send(json.dumps({
                                "type": "job_result", "job_id": job_id,
                                "status": "failed", "result": None, "error": str(e)}))
                        await ws.send(json.dumps({"type": "idle"}))
            except Exception as e:
                print(f"[blippy] ws error: {e}; retry {backoff}s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)
    finally:
        qwen_proc.terminate()


if __name__ == "__main__":
    asyncio.run(run_worker())
