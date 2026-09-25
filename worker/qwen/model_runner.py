#!/usr/bin/env python3
"""Download GGUF model and run local CPU inference via llama-cpp-python."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))

SYSTEM_PROMPT = (
    "You are a classifier. Reply with ONLY a single JSON object, no markdown, no extra text. "
    'Schema: {"category":"<one of billing|technical|shipping|account|product|other>",'
    '"sentiment":"<one of negative|neutral|positive>","confidence":<number 0..1>}'
)


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILE


def download_model() -> Path:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1_000_000:
        raise RuntimeError(f"Model download failed or file too small: {p}")
    print(f"Model ready: {p} ({p.stat().st_size / 1e9:.2f} GB)")
    return p


def load_llm(path: Optional[Path] = None):
    from llama_cpp import Llama

    path = path or model_path()
    if not path.exists():
        path = download_model()

    n_threads = int(os.environ.get("LLM_THREADS", max(1, (os.cpu_count() or 2) - 1)))
    n_ctx = int(os.environ.get("LLM_CTX", "4096"))
    print(f"Loading model with n_threads={n_threads}, n_ctx={n_ctx}, n_gpu_layers=0")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")
    return llm


def classify(llm, text: str, max_tokens: int = 96) -> Dict[str, Any]:
    """Run one classification; returns raw text + elapsed seconds."""
    user = (
        "Classify this customer support message.\n"
        f"Message: {text}\n"
        "Return JSON only."
    )
    # Qwen3 chat template via messages API
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    elapsed = time.perf_counter() - t0
    content = out["choices"][0]["message"]["content"] or ""
    return {"raw": content, "elapsed_s": elapsed}
