#!/usr/bin/env python3
"""
Run ONE Blippy job through the ACTUAL Codex CLI (codex exec).

Agent loop / tools / apply_patch / shell = upstream Codex binary.
Model = Qwen via local Responses adapter (worker/qwen/responses_server.py).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def write_codex_home(codex_home: Path, qwen_base: str, model: str = "qwen3-4b-q4_k_m") -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    # Official config shape: model_providers.<id> with base_url + wire_api=responses
    cfg = f'''
model = "{model}"
model_provider = "qwen_local"
approval_policy = "never"
sandbox_mode = "workspace-write"

[model_providers.qwen_local]
name = "Blippy Qwen (local GGUF on worker)"
base_url = "{qwen_base}"
wire_api = "responses"
experimental_bearer_token = "blippy-local"
'''
    (codex_home / "config.toml").write_text(cfg.strip() + "\n", encoding="utf-8")


def codex_event_to_blippy(obj: dict) -> Optional[dict]:
    """Map a genuine `codex exec --json` event to a Blippy event.

    Returns None for noisy/internal events. No fake thinking/file events are
    invented: every mapped event comes from an actual Codex JSONL line.
    """
    t = obj.get("type")
    if t == "thread.started":
        return {"type": "agent.started", "message": "Codex agent starting"}
    if t == "turn.started":
        return {"type": "thinking", "message": "Codex turn started"}
    if t == "turn.completed":
        return {"type": "thinking", "message": "Codex turn completed"}
    if t in ("item.started", "item.completed", "item.updated"):
        item = obj.get("item") or {}
        itype = item.get("type")
        if itype == "command_execution":
            cmd = item.get("command") or ""
            status = item.get("status")
            if t == "item.started" or item.get("exit_code") is None and status == "in_progress":
                return {"type": "command_started", "command": str(cmd)[:300]}
            return {
                "type": "command_finished",
                "command": str(cmd)[:300],
                "exit_code": item.get("exit_code"),
                "output": str(item.get("aggregated_output") or "")[-1000:],
            }
        if itype == "file_change":
            # Real Codex file mutation event (apply_patch path).
            change = item.get("change") or {}
            kind = change.get("kind") or item.get("action") or "update"
            path = change.get("path") or item.get("path") or ""
            if "create" in str(kind).lower() or "add" in str(kind).lower():
                return {"type": "file_created", "path": str(path)}
            if "delete" in str(kind).lower() or "remove" in str(kind).lower():
                return {"type": "file_deleted", "path": str(path)}
            return {"type": "file_modified", "path": str(path)}
        if itype == "agent_message":
            text = item.get("text") or ""
            return {"type": "thinking", "message": str(text)[:500]}
        if itype == "reasoning":
            text = item.get("text") or ""
            if text:
                return {"type": "thinking", "message": str(text)[:500]}
            return None
        if itype == "error":
            return {"type": "thinking", "message": f"Codex note: {item.get('message') or ''}"[:300]}
        return None
    if t == "error":
        return {"type": "thinking", "message": f"Codex error: {obj.get('message') or ''}"[:300]}
    return None


def run_codex_exec(
    workspace: Path,
    task: str,
    *,
    codex_bin: str,
    codex_home: Path,
    on_line: Optional[Callable[[str], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    timeout_s: int = 2400,
) -> dict:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    # Prefer never asking for approval in headless CI.
    # --json gives machine-readable Codex events (thread/turn/item lines);
    # on_line still receives the raw line for logging/back-compat.
    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "-c",
        'approval_policy="never"',
        task,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    deadline = time.time() + timeout_s
    while True:
        if time.time() > deadline:
            proc.kill()
            return {"ok": False, "error": "codex exec timeout", "log": "\n".join(lines[-200:])}
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line.rstrip())
            if on_line:
                on_line(line.rstrip())
            if on_event:
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    bev = codex_event_to_blippy(obj)
                    if bev is not None:
                        on_event(bev)
    code = proc.wait()
    return {
        "ok": code == 0,
        "exit_code": code,
        "log": "\n".join(lines[-500:]),
    }
