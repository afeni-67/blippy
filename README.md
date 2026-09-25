# Blippy

Remote coding agent: **actual OpenAI Codex harness** on a GitHub Actions worker, with **Qwen** only as the model backend.

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
        ├── official Codex CLI (`codex exec`)  ← ONLY agent loop / tools / apply_patch / shell
        └── Qwen Responses adapter → model_runner → llama.cpp → Qwen3-4B Q4_K_M GGUF
```

There is **one agent loop: Codex**. Blippy does not reimplement it.

## Upstream Codex

- Source: https://github.com/openai/codex  
- Pinned inspection commit: see `worker/codex/CODEX_UPSTREAM_COMMIT.txt`  
- Headless entrypoint: **`codex exec`** (`codex-rs/exec`, CLI subcommand in `codex-rs/cli`)  
- Tools / apply_patch / shell: Codex crates (`apply-patch`, `core`, sandbox/exec) inside the official binary  
- Model boundary: `ModelProviderInfo` + `wire_api = "responses"` only (`Chat` wire API removed upstream)

## Qwen adapter

Codex only speaks **Responses API** (`POST /v1/responses`).

`worker/qwen/responses_server.py` is the **model/provider adapter**:

```
Codex → HTTP 127.0.0.1:8090/v1/responses → responses_server → inference.chat → model_runner → llama.cpp → GGUF
```

Preserved:

- `worker/qwen/model_runner.py`
- `worker/qwen/inference.py`

No OpenAI/Claude/Gemini/OpenRouter cloud model for inference.

## Removed

- Custom Python agent loop (`worker/harness/agent.py`) — **deleted**
- Homemade tool dispatcher / fake apply_patch — **not used**

`worker/harness/workspace.py` only creates a temp dir and computes the **final diff** after Codex mutates files.

## API / worker / CLI

Unchanged Blippy infrastructure: Mongo jobs, SSE, worker WebSocket, atomic claim, CLI apply changes.

## Env

**API:** `MONGODB_URI`, `BLIPPY_WORKER_TOKEN`, `BLIPPY_CLIENT_TOKEN`  
**Worker secrets:** `BLIPPY_API_URL`, `BLIPPY_WORKER_TOKEN`

## Local test

```bash
# API
cd api && pip install -r requirements.txt
export MONGODB_URI=... BLIPPY_WORKER_TOKEN=dev-worker-token
uvicorn app.main:app --port 8000

# Worker (downloads GGUF + Codex binary)
pip install -r worker/requirements.txt
export BLIPPY_API_URL=http://localhost:8000 BLIPPY_WORKER_TOKEN=dev-worker-token
python -m worker.main

# CLI
python cli/blippy.py --empty "Build me a modern fashion landing page"
```
