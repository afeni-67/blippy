"""
Blippy Codex-style coding harness.

Qwen (on this worker) is the brain. Tools run in an isolated temp workspace.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from worker.harness.workspace import Workspace

SYSTEM = """You are Blippy, a coding agent running on a remote worker.
You solve coding tasks by calling tools. Reply with ONLY one JSON object per turn (no markdown).

Tools:
{"tool":"list_dir","path":"."}
{"tool":"read_file","path":"relative/path"}
{"tool":"write_file","path":"relative/path","content":"full file content"}
{"tool":"run_command","command":"shell command"}
{"tool":"finish","summary":"what you did"}

Rules:
- Prefer small, complete files.
- After creating HTML/CSS/JS, you may run simple checks.
- When the task is done, call finish.
- Never request network installs of large packages unless needed.
- Output JSON only.
"""


def _parse_action(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def run_coding_agent(
    llm,
    task: str,
    workspace_files: List[dict],
    *,
    chat_fn,
    on_event: Callable[[dict], Any],
    max_iters: int = 12,
) -> dict:
    """
    chat_fn(llm, messages, max_tokens=...) -> {"raw": str, ...}
    on_event can be sync or async.
    """
    import inspect

    async def emit(ev: dict):
        r = on_event(ev)
        if inspect.isawaitable(r):
            await r

    ws = Workspace(workspace_files)
    await emit({"type": "thinking", "message": "Preparing isolated workspace"})
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Task:\n{task}\n\nExisting files: {ws.list_dir('.') or ['(empty)']}\nStart by listing or reading as needed.",
        },
    ]

    try:
        for i in range(max_iters):
            await emit({"type": "thinking", "message": f"Planning step {i+1}/{max_iters}"})
            out = chat_fn(llm, messages, max_tokens=1024, temperature=0.15)
            raw = out.get("raw") or ""
            action = _parse_action(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw[:1500]})
                messages.append(
                    {
                        "role": "user",
                        "content": 'Invalid JSON. Reply with only one tool object, e.g. {"tool":"write_file","path":"index.html","content":"..."}',
                    }
                )
                continue

            tool = (action.get("tool") or "").strip()
            messages.append({"role": "assistant", "content": json.dumps(action)[:4000]})

            if tool == "list_dir":
                path = action.get("path") or "."
                await emit({"type": "reading_file", "path": path})
                listing = ws.list_dir(path)
                messages.append({"role": "user", "content": json.dumps({"files": listing})})
            elif tool == "read_file":
                path = action.get("path") or ""
                await emit({"type": "reading_file", "path": path})
                content = ws.read_file(path)
                messages.append({"role": "user", "content": content[:10000]})
            elif tool == "write_file":
                path = action.get("path") or ""
                content = action.get("content")
                if content is None:
                    messages.append({"role": "user", "content": "write_file requires content"})
                    continue
                existed = (ws.root / path).exists() if path else False
                ws.write_file(path, str(content))
                await emit(
                    {
                        "type": "file_modified" if existed else "file_created",
                        "path": path,
                    }
                )
                messages.append({"role": "user", "content": f"Wrote {path} ({len(str(content))} chars)"})
            elif tool == "run_command":
                cmd = action.get("command") or ""
                await emit({"type": "command_started", "command": cmd})
                result = ws.run_command(cmd)
                await emit(
                    {
                        "type": "command_finished",
                        "command": cmd,
                        "exit_code": result["exit_code"],
                    }
                )
                messages.append({"role": "user", "content": json.dumps(result)[:6000]})
            elif tool == "finish":
                summary = action.get("summary") or "done"
                await emit({"type": "thinking", "message": summary})
                changes = ws.diff_changes()
                return {
                    "ok": True,
                    "summary": summary,
                    "changes": changes,
                    "iters": i + 1,
                }
            else:
                messages.append({"role": "user", "content": f"Unknown tool: {tool}"})

        changes = ws.diff_changes()
        return {
            "ok": bool(changes),
            "summary": "iteration limit reached",
            "changes": changes,
            "iters": max_iters,
        }
    finally:
        ws.cleanup()
