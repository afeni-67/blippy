#!/usr/bin/env python3
"""
Blippy Qwen Code end-to-end test — runs ON the GitHub Actions worker.

REAL pieces, no mocks:
  real Qwen GGUF (loaded by worker/qwen/chat_server.py on :8091)
  real official Qwen Code CLI (`qwen -p`, headless)
  real worker code (worker/qwencode/run_qwencode_job.py, harness Workspace)
  real Qwen Code tools (write_file/edit/shell) mutating a temp workspace

Task: an About page (about.html) for a fashion brand site.
Exits 0 only if the page exists with the required markers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.harness.workspace import Workspace
from worker.qwencode.run_qwencode_job import write_qwen_home, run_qwen_exec

QWEN_BASE = os.environ.get("QWEN_CHAT_BASE", "http://127.0.0.1:8091/v1")
QWEN_BIN = os.environ.get("QWEN_BIN", "qwen")
TASK_TIMEOUT = int(os.environ.get("E2E_TASK_TIMEOUT", "1800"))
TASK = (
    "Create a file named about.html: an About page for a modern fashion brand "
    "website called Blippy Atelier. It must contain an h1 heading with the brand "
    "name, one paragraph telling the brand story, and a footer. "
    "Single self-contained HTML file, no external assets."
)


def qwen_version() -> str:
    try:
        return subprocess.check_output([QWEN_BIN, "--version"], text=True, timeout=60).strip()
    except Exception as e:
        return f"unknown ({e})"


def main() -> int:
    print(f"qwen version: {qwen_version()}", flush=True)
    print(f"qwen chat base: {QWEN_BASE}", flush=True)
    base_home = Path(os.environ.get("E2E_HOME", "/tmp/blippy-qwencode-home"))
    base_home.mkdir(parents=True, exist_ok=True)

    ws = Workspace([])
    qwen_home = base_home / "about"
    write_qwen_home(qwen_home, qwen_base=QWEN_BASE)
    log_lines: list = []
    res = run_qwen_exec(
        Path(ws.root), TASK, qwen_bin=QWEN_BIN, qwen_home=qwen_home,
        on_line=log_lines.append, timeout_s=TASK_TIMEOUT,
        max_turns=int(os.environ.get("E2E_MAX_TURNS", "8")),
        wall_time=os.environ.get("E2E_WALL_TIME", "25m"),
    )
    snapshot = ws.snapshot_files()
    changes = ws.diff_changes()
    html = snapshot.get("about.html") or ""
    checks = {
        "exit_ok": bool(res.get("ok")),
        "file_created": "about.html" in snapshot,
        "has_heading": "<h1" in html.lower(),
        "has_brand": "blippy" in html.lower(),
        "has_story": "<p" in html.lower(),
        "nontrivial_size": len(html) > 300,
    }
    ok = all(checks.values())
    print("ABOUT_SUMMARY " + json.dumps({"ok": ok, "checks": checks,
                                         "files": sorted(snapshot.keys()),
                                         "n_changes": len(changes)}), flush=True)
    print("ABOUT_HTML_HEAD " + html[:600].replace("\n", "\\n"), flush=True)
    print("QWEN_LOG_TAIL " + (res.get("log") or "")[-2500:].replace("\n", "\\n"), flush=True)
    try:
        out_dir = Path(os.environ.get("GITHUB_WORKSPACE", "."))
        if html:
            (out_dir / "about.html").write_text(html, encoding="utf-8")
            print(f"saved about.html ({len(html)} chars) for artifact", flush=True)
    except Exception as e:
        print(f"artifact save failed: {e}", flush=True)
    ws.cleanup()
    print("ABOUT " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
