#!/usr/bin/env python3
"""
OpenAI Chat Completions API (/v1/chat/completions) adapter for Qwen Code.

Qwen Code (openai auth type) speaks Chat Completions with OpenAI-style
`tools` (function declarations). This server is the model boundary:

  Qwen Code → POST /v1/chat/completions → this adapter → llama.cpp → Qwen GGUF

It does NOT implement the agent loop or tools — those stay in Qwen Code.
`tools`/`tool_choice` from the request are passed straight through to
llama-cpp-python, which natively supports OpenAI tool calling; whatever the
model returns (content and/or tool_calls) is relayed verbatim.

Preserved inference path: worker/qwen/model_runner.py + inference helpers.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

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


def _log(body: dict, note: str = "") -> None:
    if os.environ.get("BLIPPY_LOG_REQUESTS") != "1":
        return
    try:
        msgs = body.get("messages") or []
        rec = {
            "api": "chat_completions",
            "model": body.get("model"),
            "stream": body.get("stream"),
            "n_messages": len(msgs) if isinstance(msgs, list) else msgs,
            "roles": [m.get("role") for m in msgs] if isinstance(msgs, list) else None,
            "n_tools": len(body.get("tools") or []),
            "tool_names": [
                (t.get("function") or {}).get("name") for t in (body.get("tools") or []) if isinstance(t, dict)
            ][:12],
            "note": note,
        }
        with open(os.environ.get("BLIPPY_REQUEST_LOG", "/tmp/blippy-chat-requests.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[qwen-chat] request log failed: {e}", flush=True)


def _text_of_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if part.get("type") in ("text", "input_text", "output_text"):
            return str(part.get("text") or "")
        if "text" in part and isinstance(part["text"], str):
            return part["text"]
        if part.get("type") in ("image_url", "input_image"):
            return "[image omitted: vision not supported by this model]"
        return json.dumps(part)[:2000]
    return str(part)


def _sanitize_messages(messages: Any) -> List[dict]:
    """Normalize messages to plain-text content for the GGUF chat template.

    Qwen Code (Gemini lineage) may send content as lists with part types the
    llama.cpp Qwen template cannot concatenate (this produced
    `can only concatenate str (not "list") to str` 500s on every request).
    Tool definitions and tool_calls fields are preserved untouched — only the
    human-readable `content` is flattened.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return [{"role": "user", "content": str(messages)}]
    out: List[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(t for t in (_text_of_part(p) for p in content) if t)
        elif not isinstance(content, str):
            content = "" if content is None else str(content)
        clean: dict = {"role": role, "content": content}
        for k in ("name", "tool_call_id", "tool_calls"):
            if m.get(k) is not None:
                clean[k] = m[k]
        out.append(clean)
    return out or [{"role": "user", "content": "Continue."}]


def run_chat(body: dict, llm=None) -> dict:
    """Non-streaming Chat Completions object via llama.cpp (tools passed through)."""
    llm = llm if llm is not None else get_llm()
    messages = _sanitize_messages(body.get("messages"))
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "temperature": float(body.get("temperature", 0.2)),
        "max_tokens": int(body.get("max_tokens") or int(os.environ.get("QWEN_MAX_TOKENS", "2048"))),
        "top_p": float(body.get("top_p", 0.9)),
    }
    if body.get("tools"):
        kwargs["tools"] = body["tools"]
    if body.get("tool_choice"):
        kwargs["tool_choice"] = body["tool_choice"]
    if body.get("stop") is not None:
        kwargs["stop"] = body["stop"]
    t0 = time.perf_counter()
    out = llm.create_chat_completion(**kwargs)
    elapsed = time.perf_counter() - t0
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, str) and "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    print(f"[qwen-chat] inference {elapsed:.1f}s content_len={len(content)} "
          f"tool_calls={len(msg.get('tool_calls') or [])} finish={choice.get('finish_reason')}", flush=True)
    if os.environ.get("BLIPPY_LOG_QWEN") == "1":
        try:
            with open(os.environ.get("BLIPPY_QWEN_LOG", "/tmp/blippy-qwen-chat.txt"), "a") as f:
                f.write(json.dumps(msg)[:6000] + "\n====\n")
        except Exception:
            pass
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "qwen3-4b-q4_k_m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {}),
                },
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        ],
        "usage": out.get("usage") or {},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[qwen-chat] {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health") or "models" in self.path:
            self._json(200, {"object": "list",
                             "data": [{"id": "qwen3-4b-q4_k_m", "object": "model",
                                       "owned_by": "blippy-local"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if "chat/completions" not in self.path:
            self._json(404, {"error": "only POST /v1/chat/completions is supported"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        stream = bool(body.get("stream"))
        _log(body)
        try:
            resp = run_chat(body)
        except Exception as e:
            import traceback as _tb
            print(f"[qwen-chat] inference error: {e}\n{_tb.format_exc()}", flush=True)
            self._json(500, {"error": {"message": str(e)}})
            return
        if not stream:
            self._json(200, resp)
            return
        # Minimal OpenAI SSE: one delta chunk + [DONE]. Qwen Code parses this.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(data: str):
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        ch = resp["choices"][0]
        chunk = {
            "id": resp["id"], "object": "chat.completion.chunk",
            "created": resp["created"], "model": resp["model"],
            "choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": ch["message"].get("content") or "",
                                   **({"tool_calls": ch["message"]["tool_calls"]} if ch["message"].get("tool_calls") else {})},
                         "finish_reason": None}],
        }
        send(json.dumps(chunk))
        done = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"],
                "model": resp["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason") or "stop"}]}
        send(json.dumps(done))
        send("[DONE]")


def main():
    port = int(os.environ.get("QWEN_CHAT_PORT", "8091"))
    get_llm()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[qwen-chat] listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
