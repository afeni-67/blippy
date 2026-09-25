#!/usr/bin/env python3
"""
Minimal Blippy CLI — runs on the USER device only.
Does NOT load Qwen. Connects to Blippy API over HTTPS + SSE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)

API = os.environ.get("BLIPPY_API_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("BLIPPY_CLIENT_TOKEN", "dev-client-token")


def collect_workspace(root: Path, max_files: int = 40, max_bytes: int = 80_000) -> list:
    files = []
    total = 0
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "models"}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip for part in p.parts):
            continue
        if p.stat().st_size > 200_000:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if total + len(text) > max_bytes:
            break
        files.append({"path": rel, "content": text})
        total += len(text)
        if len(files) >= max_files:
            break
    return files


def apply_changes(root: Path, changes: list) -> int:
    n = 0
    for ch in changes or []:
        op = ch.get("operation")
        path = ch.get("path") or ""
        if not path or ".." in path:
            continue
        dest = root / path
        if op == "delete":
            if dest.exists():
                dest.unlink()
                n += 1
        elif op in ("create", "modify"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(ch.get("content") or "", encoding="utf-8")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(prog="blippy")
    ap.add_argument("task", nargs="?", help='e.g. "Build me a modern fashion landing page"')
    ap.add_argument("--cwd", type=Path, default=Path("."))
    ap.add_argument("--empty", action="store_true", help="Send empty workspace")
    args = ap.parse_args()
    if not args.task:
        print('Usage: blippy "Build me a modern fashion landing page"')
        return 2

    root = args.cwd.resolve()
    print("Blippy")
    print("> Sending task...")
    workspace = {"files": [] if args.empty else collect_workspace(root)}
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            f"{API}/jobs",
            json={"client_id": "cli", "task": args.task, "workspace": workspace},
            headers=headers,
        )
        if r.status_code >= 400:
            print("Failed:", r.text)
            return 1
        job_id = r.json()["job_id"]
        print("✓ Job accepted")
        print(f"  job_id={job_id}")

    # SSE
    changes = []
    with httpx.stream("GET", f"{API}/jobs/{job_id}/events", timeout=None) as stream:
        for line in stream.iter_lines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            t = data.get("type")
            if t == "worker_assigned":
                print(f"✓ Worker assigned ({data.get('worker_id')})")
            elif t == "thinking":
                print(f"◐ {data.get('message')}")
            elif t == "file_created":
                print(f"◐ Creating {data.get('path')}")
            elif t == "file_modified":
                print(f"◐ Modifying {data.get('path')}")
            elif t == "command_started":
                print(f"◐ Running `{data.get('command')}`...")
            elif t == "command_finished":
                print(f"  exit {data.get('exit_code')}")
            elif t == "job_completed":
                result = data.get("result") or {}
                changes = result.get("changes") or []
                print("✓ Done")
                break
            elif t == "job_failed":
                print("✗ Failed:", data.get("error"))
                result = data.get("result") or {}
                changes = result.get("changes") or []
                break

    n = apply_changes(root, changes)
    print(f"\n{n} files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
