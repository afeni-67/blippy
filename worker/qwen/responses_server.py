#!/usr/bin/env python3
"""
Minimal OpenAI Responses API (/v1/responses) adapter for Codex.

Codex (upstream) only speaks WireApi::Responses — chat completions wire API was removed.
This server is the model/provider boundary:

  Codex  →  POST /v1/responses  →  this adapter  →  existing model_runner + llama.cpp  →  Qwen GGUF

It does NOT implement the agent loop, tools, apply_patch, or shell — those stay in Codex.

Protocol notes (verified against live `codex exec` + upstream source
codex-rs/codex-api/src/sse/responses.rs @ 4b1c0c30):
- Codex sends stream=true with body keys: model, instructions, input, tools,
  tool_choice, parallel_tool_calls, reasoning, store, stream, include, ...
- tools are {"type":"function","name":...,"parameters":{...}} (e.g. exec_command).
- Streaming MUST be: response.created -> response.output_item.done (full item,
  once per output item) -> response.completed. Delta-only events without a
  preceding valid output_item cause "OutputTextDelta without active item".
- Non-streaming returns the full response object with output[] items.
- Output items Codex understands include:
    {"type":"message","role":"assistant","content":[{"type":"output_text","text":...}]}
    {"type":"function_call","name":...,"arguments":"{...json string...}","call_id":...}
  (arguments is a JSON *string*, not an object.)
- After a function_call, Codex appends function_call + function_call_output
  items to `input` on the next turn. This adapter converts those back into
  chat context so Qwen can iterate multi-turn.
- Qwen itself does not natively emit Responses function_call items, so the
  adapter prompts it to emit a single-line JSON tool request which is parsed
  here into a Responses function_call. That translation is the ONLY agent-ish
  logic in this file; execution stays in Codex.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Input conversion: Responses `input` -> chat messages for Qwen
# ---------------------------------------------------------------------------

def _text_of_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") in ("input_text", "output_text", "text"):
                    texts.append(c.get("text") or "")
                elif "text" in c:
                    texts.append(str(c.get("text") or ""))
            elif isinstance(c, str):
                texts.append(c)
        return "\n".join(texts)
    return str(content)


def _messages_from_input(inp: Any) -> List[Dict[str, str]]:
    """Best-effort convert Responses `input` into chat messages for Qwen.

    Preserves multi-turn tool history: previous function_call items become
    assistant context lines; function_call_output items become user lines
    carrying the tool result so Qwen can continue reasoning.
    """
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
        itype = item.get("type")
        if itype == "function_call_output":
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result ({item.get('call_id') or item.get('name') or 'tool'}): {item.get('output')}",
                }
            )
            continue
        if itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Called tool {item.get('name')} with arguments: {item.get('arguments')}",
                }
            )
            continue
        if itype == "reasoning":
            continue
        role = item.get("role") or "user"
        if role in ("system", "developer"):
            role = "system"
        elif role not in ("user", "assistant"):
            role = "user"
        content = _text_of_content(item.get("content"))
        if not content:
            content = item.get("text") or ""
        if not content:
            continue
        messages.append({"role": role, "content": str(content)})
    if not messages:
        messages.append({"role": "user", "content": "Continue."})
    return messages


# ---------------------------------------------------------------------------
# Tool prompting + parsing: Qwen text -> Responses function_call
# ---------------------------------------------------------------------------

def _available_function_tools(tools: Optional[list]) -> List[dict]:
    out = []
    if not tools:
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and t.get("name"):
            out.append(t)
    return out


def _tools_instruction(tools: Optional[list]) -> str:
    fn_tools = _available_function_tools(tools)
    if not fn_tools:
        return ""
    lines = [
        "You control a coding agent by EITHER answering in plain text OR requesting exactly one tool call.",
        "To request a tool call, output ONLY a single line of JSON with this shape:",
        '{"name": "<tool>", "arguments": { ... }}',
        "Available tools:",
    ]
    for t in fn_tools[:12]:
        params = t.get("parameters") or {}
        props = list((params.get("properties") or {}).keys())[:12]
        req = params.get("required") or []
        lines.append(f"- {t.get('name')}: {t.get('description') or ''} args={props} required={req}")
    lines.append(
        "Rules: output the JSON tool request alone with no surrounding text when you need a tool; "
        "otherwise output plain text. Never invent tool results."
    )
    return "\n".join(lines)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first plausible JSON object from Qwen output."""
    text = text.strip()
    # Prefer fenced ```json blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    # Fall back to outermost {...}
    m2 = _JSON_OBJ_RE.search(text)
    if m2:
        candidates.append(m2.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    # Lenient fallback: Qwen sometimes emits multi-line shell (heredocs) with
    # literal newlines inside the JSON string, which strict JSON rejects.
    # Extract {"name": ..., "cmd": ...} tolerantly for the exec_command case.
    m = re.search(r'"name"\s*:\s*"(?P<name>[^"]+)"', text)
    if m:
        name = m.group("name")
        m2 = re.search(r'"cmd"\s*:\s*"(?P<cmd>.*)"\s*\}\s*\}?', text, re.DOTALL)
        if m2:
            raw_cmd = m2.group("cmd")
            # Unescape common sequences; keep literal newlines as-is for shell.
            try:
                cmd = json.loads(f'"{raw_cmd}"')
            except Exception:
                cmd = raw_cmd.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            return {"name": name, "arguments": {"cmd": cmd}}
    return None


def parse_tool_call(text: str, tools: Optional[list]) -> Optional[Tuple[str, dict, str]]:
    """Parse Qwen output into (name, arguments_dict, preamble_text).

    Returns None when the output is a plain-text answer (no tool call).
    """
    fn_tools = _available_function_tools(tools)
    valid_names = {t["name"] for t in fn_tools}
    obj = _extract_json_object(text)
    if not obj:
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    # Some models emit {"tool": ..., "cmd": "..."} flat shapes for exec_command
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    # Accept {"cmd": ...} flattened into the top level
    if "cmd" in obj and "arguments" not in obj and "args" not in obj:
        args = {"cmd": obj["cmd"]}
    if valid_names and name not in valid_names:
        return None
    # Preamble = text before the JSON object, if any
    preamble = ""
    try:
        start = text.index("{")
        preamble = text[:start].strip()
    except ValueError:
        pass
    if preamble and len(preamble) > 800:
        preamble = preamble[:800]
    # Strip thinking remnants
    if "</think>" in preamble:
        preamble = preamble.split("</think>", 1)[-1].strip()
    return name, args, preamble


# ---------------------------------------------------------------------------
# Response building (non-streaming object + SSE)
# ---------------------------------------------------------------------------

def _new_ids() -> Tuple[str, str, str]:
    return (
        f"resp_{uuid.uuid4().hex}",
        f"msg_{uuid.uuid4().hex}",
        f"call_{uuid.uuid4().hex[:24]}",
    )


def run_response(body: dict, llm=None) -> dict:
    """Build a non-streaming Responses-like object from Qwen output."""
    llm = llm if llm is not None else get_llm()
    instructions = body.get("instructions") or ""
    tools = body.get("tools")
    messages = _messages_from_input(body.get("input"))
    tool_instr = _tools_instruction(tools)
    sys_parts = []
    if instructions:
        sys_parts.append(str(instructions))
    if tool_instr:
        sys_parts.append(tool_instr)
    if sys_parts:
        messages = [{"role": "system", "content": "\n\n".join(sys_parts)}] + messages

    out = chat(llm, messages, max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "1024")), temperature=0.2)
    text = (out.get("raw") or "").strip()
    resp_id = f"resp_{uuid.uuid4().hex}"
    model = body.get("model") or "qwen3-4b-q4_k_m"

    parsed = parse_tool_call(text, tools) if text else None
    output: List[dict] = []
    if parsed:
        name, args, preamble = parsed
        if preamble:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": preamble}],
                    "status": "completed",
                }
            )
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                "name": name,
                "arguments": json.dumps(args),
            }
        )
    else:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "status": "completed",
            }
        )
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Optional request logging (test observability only; off by default).
# Set BLIPPY_LOG_REQUESTS=1 and BLIPPY_REQUEST_LOG=/path/to.jsonl to record
# a compact summary of every Responses request Codex sends.
# ---------------------------------------------------------------------------

def _log_request(body: dict) -> None:
    if os.environ.get("BLIPPY_LOG_REQUESTS") != "1":
        return
    try:
        inp = body.get("input")
        types = [it.get("type") if isinstance(it, dict) else "?" for it in inp] if isinstance(inp, list) else inp
        rec = {
            "model": body.get("model"),
            "stream": body.get("stream"),
            "keys": sorted(body.keys()),
            "n_input": len(inp) if isinstance(inp, list) else inp,
            "input_types": types if isinstance(types, list) and len(types) <= 12 else str(types)[:500],
            "n_tools": len(body.get("tools") or []),
            "tool_names": [
                t.get("name") for t in (body.get("tools") or []) if isinstance(t, dict)
            ][:12],
        }
        with open(os.environ.get("BLIPPY_REQUEST_LOG", "/tmp/blippy-requests.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[qwen-responses] request log failed: {e}", flush=True)


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
        if "/responses" not in self.path:
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
        _log_request(body)
        try:
            resp = run_response(body)
        except Exception as e:
            self._json(500, {"error": {"message": str(e)}})
            return
        if not stream:
            self._json(200, resp)
            return
        # SSE shape Codex actually parses: created -> output_item.done* -> completed.
        # (Delta-only streams without a valid output_item break the turn.)
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
        for idx, item in enumerate(resp["output"]):
            send_event("response.output_item.done", {"output_index": idx, "item": item})
        send_event(
            "response.completed",
            {
                "response": {
                    "id": resp["id"],
                    "status": "completed",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
            },
        )


def main():
    port = int(os.environ.get("QWEN_RESPONSES_PORT", "8090"))
    # Preload model
    get_llm()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[qwen-responses] listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
