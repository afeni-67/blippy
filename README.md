# Blippy

**Remote coding agent.** You type a task in your project; a GitHub Actions worker runs **Qwen locally on the runner**, streams progress, and returns file changes that the CLI applies on your machine.

> **Qwen inference runs remotely on the Blippy worker. Users do not install or run Qwen locally.**

> **GitHub Actions runner persistence/keep-alive is intentionally out of scope. The project owner will provide that separately** (e.g. ScholarReach-style 20-matrix waves, second wave at +3h).

## Architecture

```
USER DEVICE (Blippy CLI only)
        │ HTTPS + SSE
        ▼
   BLIPPY API (Render)
        │ WebSocket (worker outbound)
        ▼
GITHUB ACTIONS WORKER
   ├── Coding harness (tools)
   └── Qwen3-4B Q4_K_M (llama-cpp-python, CPU)
```

- User **never** downloads the GGUF.
- Runner **never** needs inbound ports; worker dials out to `BLIPPY_API_URL`.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/jobs` | Create coding job `{client_id, task, workspace}` |
| GET | `/jobs/:id` | Job status |
| GET | `/jobs/:id/events` | **SSE** progress stream |
| POST | `/workers/register` | Worker registration |
| POST | `/workers/heartbeat` | Worker health |
| WS | `/workers/ws?token=&worker_id=` | Persistent worker connection |

### Job states

`queued` → `assigned` → `running` → `completed` | `failed` | `cancelled`

Stale jobs from disconnected workers are requeued.

### SSE event types

`job_queued`, `worker_assigned`, `thinking`, `reading_file`, `file_created`, `file_modified`, `command_started`, `command_finished`, `job_completed`, `job_failed`

### Worker protocol (WebSocket JSON)

- Worker → API: `idle`, `heartbeat`, `event`, `job_result`
- API → Worker: `job_assigned` `{job_id, task, workspace}`

## Qwen integration

Reuses the proven stack from `local-llm-actions-benchmark`:

- Model: `Qwen/Qwen3-4B-GGUF` → `Qwen3-4B-Q4_K_M.gguf` (Apache-2.0)
- Runtime: `llama-cpp-python`, `n_gpu_layers=0`
- Code: `worker/qwen/model_runner.py` + `worker/qwen/inference.py`

## Multi-worker

- Workflow matrix **20 workers** per dispatch (`blippy-worker.yml`).
- API assigns one queued job per idle WebSocket connection (atomic Mongo claim).
- Busy workers skip; jobs wait in `queued` until an idle worker appears.

Wave / 3-hour overlap scheduling is **owner-side** (same idea as ScholarReach wake), not in this repo’s keep-alive.

## Environment variables

**API (Render)**

- `MONGODB_URI`
- `BLIPPY_WORKER_TOKEN`
- `BLIPPY_CLIENT_TOKEN`
- `REDIS_URL` (optional; MVP uses Mongo + in-process bus)

**Worker (Actions secrets)**

- `BLIPPY_API_URL` — e.g. `https://blippy-api.onrender.com`
- `BLIPPY_WORKER_TOKEN`

**CLI**

- `BLIPPY_API_URL`
- `BLIPPY_CLIENT_TOKEN`

## Local development

```bash
# API
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGODB_URI='mongodb+srv://...'
export BLIPPY_WORKER_TOKEN=dev-worker-token
uvicorn app.main:app --reload --port 8000

# Worker (needs Qwen download ~2.5GB + RAM)
cd ..
pip install -r worker/requirements.txt
export BLIPPY_API_URL=http://localhost:8000
export BLIPPY_WORKER_TOKEN=dev-worker-token
python -m worker.main

# CLI
pip install httpx
export BLIPPY_API_URL=http://localhost:8000
mkdir -p /tmp/demo && cd /tmp/demo
python /path/to/blippy/cli/blippy.py --empty "Build me a modern fashion landing page"
```

## One complete job test

1. Start API with MongoDB.
2. Start one worker (local or Actions).
3. Run CLI task with `--empty` in an empty folder.
4. Watch SSE lines: worker assigned → file_created → job_completed.
5. Confirm `index.html` / CSS / JS appeared locally.

## Production CLI remaining work

- Richer TUI, auth accounts, project ignore rules, diff review before apply, cancel job.

## Render deployment remaining work

- Create Web Service from `api/`, set env vars, attach MongoDB URI.
- Set Actions secrets `BLIPPY_API_URL` + `BLIPPY_WORKER_TOKEN`.
- Owner adds wake scheduler for continuous 20-worker waves.
