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


def run_codex_exec(
    workspace: Path,
    task: str,
    *,
    codex_bin: str,
    codex_home: Path,
    on_line: Optional[Callable[[str], None]] = None,
    timeout_s: int = 2400,
) -> dict:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    # Prefer never asking for approval in headless CI
    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
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
    code = proc.wait()
    return {
        "ok": code == 0,
        "exit_code": code,
        "log": "\n".join(lines[-500:]),
    }
