#!/usr/bin/env python3
"""
Minimal OpenAI Responses API (/v1/responses) adapter for Codex.

Codex (upstream) only speaks WireApi::Responses — chat completions wire API was removed.
This server is the model/provider boundary:

  Codex  →  POST /v1/responses  →  this adapter  →  existing model_runner + llama.cpp  →  Qwen GGUF

It does NOT implement the agent loop, tools, apply_patch, or shell — those stay in Codex.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from worker.qwen.inference import chat
from worker.qwen.model_runner import download_model, load_llm

_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            path = download_model()
            _llm = load_llm(path)
        return _llm


def _messages_from_input(inp: Any) -> List[Dict[str, str]]:
    """Best-effort convert Responses `input` into chat messages for Qwen."""
    messages: List[Dict[str, str]] = []
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
        return messages
    if not isinstance(inp, list):
        messages.append({"role": "user", "content": json.dumps(inp)[:8000]})
        return messages
    for item in inp:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("type") or "user"
        if role in ("system", "developer"):
            role = "system"
        elif role not in ("user", "assistant"):
            # function_call_output etc.
            if item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result ({item.get('call_id')}): {item.get('output')}",
                    }
                )
                continue
            role = "user"
        content = item.get("content")
        if isinstance(content, list):
            texts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text"):
                    texts.append(c.get("text") or "")
                elif isinstance(c, str):
                    texts.append(c)
            content = "\n".join(texts)
        if content is None:
            content = item.get("text") or json.dumps(item)[:4000]
        messages.append({"role": role, "content": str(content)})
    if not messages:
        messages.append({"role": "user", "content": "Continue."})
    return messages


def _tools_hint(tools: Optional[list]) -> str:
    if not tools:
        return ""
    # Surface tool schemas so Qwen can emit function_call-like JSON if needed.
    # Full Codex tool protocol still relies on the Responses shape we return.
    names = []
    for t in tools:
        if isinstance(t, dict):
            names.append(t.get("name") or (t.get("function") or {}).get("name") or "tool")
    return (
        "\nAvailable Codex tools (use the Responses function_call protocol when applicable): "
        + ", ".join(names[:40])
    )


def run_response(body: dict) -> dict:
    """Non-streaming Responses-like object (Codex can also consume streamed SSE)."""
    llm = get_llm()
    instructions = body.get("instructions") or ""
    messages = _messages_from_input(body.get("input"))
    if instructions:
        messages = [{"role": "system", "content": str(instructions) + _tools_hint(body.get("tools"))}] + messages
    elif body.get("tools"):
        messages = [{"role": "system", "content": _tools_hint(body.get("tools"))}] + messages

    out = chat(llm, messages, max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "1024")), temperature=0.2)
    text = (out.get("raw") or "").strip()
    resp_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model") or "qwen3-4b-q4_k_m",
        "output": [
            {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[qwen-responses] {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health") or self.path.startswith("/v1/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3-4b-q4_k_m",
                            "object": "model",
                            "owned_by": "blippy-local",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/responses"):
            self._json(404, {"error": "only POST /v1/responses is supported"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        stream = bool(body.get("stream"))
        try:
            resp = run_response(body)
        except Exception as e:
            self._json(500, {"error": {"message": str(e)}})
            return
        if not stream:
            self._json(200, resp)
            return
        # Minimal SSE stream Codex can parse
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send_event(etype: str, data: dict):
            payload = {"type": etype, **data}
            self.wfile.write(f"event: {etype}\n".encode())
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        send_event("response.created", {"response": {"id": resp["id"], "status": "in_progress"}})
        text = resp["output"][0]["content"][0]["text"]
        item_id = resp["output"][0]["id"]
        send_event(
            "response.output_item.added",
            {"item": {"type": "message", "id": item_id, "role": "assistant"}},
        )
        send_event(
            "response.output_text.delta",
            {"item_id": item_id, "delta": text},
        )
        send_event(
            "response.output_text.done",
            {"item_id": item_id, "text": text},
        )
        send_event("response.completed", {"response": resp})


def main():
    port = int(os.environ.get("QWEN_RESPONSES_PORT", "8090"))
    # Preload model
    get_llm()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[qwen-responses] listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
