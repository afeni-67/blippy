#!/usr/bin/env python3
"""
Run ONE job through the official Qwen Code CLI (headless `qwen -p`).

Agent loop / tools / file edits / shell = upstream Qwen Code binary.
Model = worker-local Qwen GGUF via worker/qwen/chat_server.py
(OpenAI-compatible /v1/chat/completions on QWEN_CHAT_PORT).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional


def write_qwen_home(qwen_home: Path, qwen_base: str, model: str = "qwen3-4b-q4_k_m",
                    ctx: int = 16384) -> None:
    """QWEN_HOME dir holding settings.json with the local provider only."""
    qwen_home.mkdir(parents=True, exist_ok=True)
    settings = {
        "modelProviders": {
            "openai": [
                {
                    "id": model,
                    "name": f"Blippy Qwen (local GGUF on worker)",
                    "baseUrl": qwen_base,
                    "envKey": "BLIPPY_QWEN_API_KEY",
                    "generationConfig": {
                        "timeout": 600000,
                        "streamIdleTimeoutMs": 600000,
                        "maxRetries": 1,
                        "contextWindowSize": ctx,
                        "samplingParams": {"temperature": 0.2, "top_p": 0.9,
                                           "max_tokens": 4096},
                    },
                }
            ]
        },
        "security": {"auth": {"selectedType": "openai"}},
        "model": {"name": model},
    }
    (qwen_home / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def run_qwen_exec(
    workspace: Path,
    task: str,
    *,
    qwen_bin: str,
    qwen_home: Path,
    on_line: Optional[Callable[[str], None]] = None,
    timeout_s: int = 2400,
    max_turns: int = 10,
    wall_time: str = "25m",
) -> dict:
    env = os.environ.copy()
    env["QWEN_HOME"] = str(qwen_home)
    env.setdefault("BLIPPY_QWEN_API_KEY", "blippy-local")
    env["TERM"] = env.get("TERM") or "dumb"
    cmd = [
        qwen_bin, "-p", task,
        "--yolo",
        "--max-session-turns", str(max_turns),
        "--max-wall-time", wall_time,
        "--output-format", "json",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(workspace), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    deadline = time.time() + timeout_s
    while True:
        if time.time() > deadline:
            proc.kill()
            return {"ok": False, "error": "qwen exec timeout", "log": "\n".join(lines[-200:])}
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line.rstrip())
            if on_line:
                on_line(line.rstrip())
    code = proc.wait()
    return {"ok": code == 0, "exit_code": code, "log": "\n".join(lines[-500:])}
