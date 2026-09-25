#!/usr/bin/env python3
"""
Blippy Qwen x Codex end-to-end test — runs ON the GitHub Actions worker.

Uses the REAL pieces, no mocks:
  real Qwen GGUF (already loaded by worker/qwen/responses_server.py on :8090)
  real official Codex CLI binary (`codex exec --json`)
  real worker code (worker/codex/run_codex_job.py, worker/harness/workspace.py)
  real Codex tools (exec_command shell; file edits happen through it in 0.145+)

Exits 0 only if every test passes; prints a JSON summary for the workflow log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.codex.run_codex_job import write_codex_home, run_codex_exec
from worker.harness.workspace import Workspace

QWEN_BASE = os.environ.get("QWEN_BASE", "http://127.0.0.1:8090/v1")
CODEX_BIN = os.environ.get("CODEX_BIN", str(Path.home() / ".local/bin/codex"))
TASK_TIMEOUT = int(os.environ.get("E2E_TASK_TIMEOUT", "1500"))
RESULTS: dict = {}


def note(name: str, ok: bool, detail: str = "") -> None:
    RESULTS[name] = {"ok": ok, "detail": detail[:2000]}
    print(f"[{ 'PASS' if ok else 'FAIL' }] {name} {detail[:300]}", flush=True)


def post_responses(body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        QWEN_BASE + "/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def codex_version() -> str:
    try:
        return subprocess.check_output([CODEX_BIN, "--version"], text=True, timeout=30).strip()
    except Exception as e:
        return f"unknown ({e})"


def run_task(task: str, files: list, codex_home: Path) -> tuple:
    """Run one task through REAL codex exec in an isolated Workspace.

    Returns (ok, files_snapshot, changes, log_tail). Workspace is NOT cleaned
    so the caller can assert on resulting files.
    """
    ws = Workspace(files)
    write_codex_home(codex_home, qwen_base=QWEN_BASE)
    events: list = []

    def on_event(ev: dict) -> None:
        events.append(ev)

    res = run_codex_exec(
        Path(ws.root), task, codex_bin=CODEX_BIN, codex_home=codex_home,
        on_event=on_event, timeout_s=TASK_TIMEOUT,
    )
    snapshot = ws.snapshot_files()
    changes = ws.diff_changes()
    log_tail = (res.get("log") or "")[-3000:]
    return res, snapshot, changes, events, log_tail, ws


def main() -> int:
    print(f"codex version: {codex_version()}", flush=True)
    print(f"qwen base: {QWEN_BASE}", flush=True)
    base_home = Path(os.environ.get("E2E_HOME", "/tmp/blippy-e2e-home"))
    base_home.mkdir(parents=True, exist_ok=True)

    # --- T1: direct Responses check (model/instructions/tools round-trip) ---
    try:
        body = {
            "model": "qwen3-4b-q4_k_m",
            "instructions": "You are a test assistant.",
            "input": "Reply with exactly: PROVIDER_OK",
            "tools": [
                {
                    "type": "function", "name": "exec_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        }
        resp = post_responses(body)
        out = resp.get("output") or []
        kinds = [o.get("type") for o in out]
        ok = bool(resp.get("id")) and resp.get("status") == "completed" and kinds and kinds[0] in ("message", "function_call")
        detail = f"output_types={kinds} text={(json.dumps(out)[:300])}"
        note("T1_responses_direct", ok, detail)
    except Exception as e:
        note("T1_responses_direct", False, f"error: {e}")

    # --- T2: TOOL CALL — read hello.txt, write result.txt (the critical proof) ---
    try:
        home = base_home / "t2"
        res, snap, changes, events, tail, ws = run_task(
            "Read hello.txt and create result.txt containing a summary of its contents.",
            [{"path": "hello.txt", "content": "hello blippy file content for summary test"}],
            home,
        )
        has_result = "result.txt" in snap and len((snap.get("result.txt") or "").strip()) > 0
        tool_events = [e for e in events if e.get("type") in ("command_started", "command_finished")]
        ok = has_result and len(tool_events) > 0
        note("T2_tool_call", ok,
             f"codex_ok={res.get('ok')} result.txt={repr((snap.get('result.txt') or '')[:120])} "
             f"tool_events={len(tool_events)} changes={len(changes)} tail={tail[-400:]}")
        ws.cleanup()
    except Exception as e:
        note("T2_tool_call", False, f"error: {e}")

    # --- T3: APPLY — index.html with Hello Blippy heading ---
    try:
        home = base_home / "t3"
        res, snap, changes, events, tail, ws = run_task(
            "Create a small index.html page with a heading saying Hello Blippy.",
            [],
            home,
        )
        html = snap.get("index.html") or ""
        ok = "Hello Blippy" in html and "<h1" in html.lower()
        note("T3_apply_patch", ok,
             f"codex_ok={res.get('ok')} index.html={repr(html[:200])} tail={tail[-400:]}")
        ws.cleanup()
    except Exception as e:
        note("T3_apply_patch", False, f"error: {e}")

    # --- T4: SHELL — generated.txt containing hello ---
    try:
        home = base_home / "t4"
        res, snap, changes, events, tail, ws = run_task(
            'Create a file called generated.txt containing "hello", then verify it exists.',
            [],
            home,
        )
        ok = (snap.get("generated.txt") or "").strip() == "hello"
        note("T4_shell", ok,
             f"codex_ok={res.get('ok')} generated.txt={repr((snap.get('generated.txt') or '')[:80])} tail={tail[-400:]}")
        ws.cleanup()
    except Exception as e:
        note("T4_shell", False, f"error: {e}")

    # --- T5: ITERATION — create, inspect, fix (needs >= 2 tool turns) ---
    try:
        home = base_home / "t5"
        res, snap, changes, events, tail, ws = run_task(
            "Create fixme.html with a heading, then read it back and correct it if the heading is wrong. The final heading must say Fixed OK.",
            [],
            home,
        )
        html = snap.get("fixme.html") or ""
        tool_events = [e for e in events if e.get("type") in ("command_started", "command_finished")]
        n_tool_turns = len([e for e in tool_events if e.get("type") == "command_started"])
        ok = "Fixed OK" in html and n_tool_turns >= 2
        note("T5_multiturn", ok,
             f"codex_ok={res.get('ok')} tool_starts={n_tool_turns} fixme.html={repr(html[:200])} tail={tail[-400:]}")
        ws.cleanup()
    except Exception as e:
        note("T5_multiturn", False, f"error: {e}")

    print("E2E_SUMMARY " + json.dumps(RESULTS), flush=True)
    failed = [k for k, v in RESULTS.items() if not v["ok"]]
    if failed:
        print(f"E2E FAILED: {failed}", flush=True)
        return 1
    print("E2E ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
