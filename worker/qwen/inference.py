"""Thin wrapper around existing llama-cpp-python chat completion."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def chat(
    llm,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Run one chat completion on the worker-local Qwen model.
    Preserves the existing llama-cpp-python create_chat_completion path.
    """
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    content = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    # Strip Qwen3 thinking blocks if present for cleaner tool JSON
    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    elif content.strip().startswith("<think>"):
        # truncated think — try after last newline
        parts = content.rsplit("\n", 1)
        content = parts[-1] if len(parts) > 1 else content
    return {"raw": content, "elapsed_s": time.perf_counter() - t0}
