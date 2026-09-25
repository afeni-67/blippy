# Blippy

Remote coding agent: **official Qwen Code CLI harness** on a GitHub Actions worker, with **Qwen (local GGUF)** as the only model backend.

> **Qwen inference runs remotely on the Blippy worker. Users do not install or run Qwen locally.**

> **GitHub Actions keep-alive is out of scope** (owner provides ScholarReach-style 20-worker / +3h waves).

## Architecture

```
User (Blippy CLI only)
        │ HTTPS + SSE
        ▼
Blippy API (Render + MongoDB)
        │ WebSocket (outbound from worker)
        ▼
GitHub Actions worker
        ├── official Qwen Code CLI (`qwen -p`, headless)  ← ONLY agent loop / tools
        └── Qwen chat adapter → model_runner → llama.cpp → Qwen3-4B Q4_K_M GGUF
```

There is **one agent loop: Qwen Code**. Blippy does not reimplement it.

## Upstream harness

- Source: https://github.com/QwenLM/qwen-code
- Headless entrypoint: **`qwen -p "..." --yolo`** (scripts/CI mode)
- Tools / file edits / shell: inside the official binary
- Model boundary: OpenAI-compatible Chat Completions (`POST /v1/chat/completions`)

## Qwen adapter

`worker/qwen/chat_server.py` is the **model/provider adapter**:

```
Qwen Code → HTTP 127.0.0.1:8091/v1/chat/completions → chat_server → llama.cpp → GGUF
```

Preserved:

- `worker/qwen/model_runner.py`
- `worker/qwen/inference.py`

No OpenAI/Claude/Gemini/OpenRouter cloud model for inference.

`worker/harness/workspace.py` only creates a temp dir and computes the **final diff** after Qwen Code mutates files.

## API / worker / CLI

Mongo jobs, SSE, worker WebSocket, atomic claim, CLI apply changes.

## Env

**API:** `MONGODB_URI`, `BLIPPY_WORKER_TOKEN`, `BLIPPY_CLIENT_TOKEN`
**Worker secrets:** `BLIPPY_API_URL`, `BLIPPY_WORKER_TOKEN`

## Local test

```bash
# API
cd api && pip install -r requirements.txt
export MONGODB_URI=... BLIPPY_WORKER_TOKEN=dev-worker-token
uvicorn app.main:app --port 8000

# Worker (downloads GGUF + Qwen Code binary)
pip install -r worker/requirements.txt
export BLIPPY_API_URL=http://localhost:8000 BLIPPY_WORKER_TOKEN=dev-worker-token
python -m worker.main

# CLI
python cli/blippy.py --empty "Build me a modern fashion landing page"
```

## Harness proof (runs on GHA, real GGUF)

```bash
# dispatched manually: Actions -> "Qwen Code E2E" -> Run workflow
# writes about.html (fashion About page) with real Qwen Code tools
```
