import os

MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb://localhost:27017/blippy",
)
REDIS_URL = os.environ.get("REDIS_URL", "")  # optional
WORKER_TOKEN = os.environ.get("BLIPPY_WORKER_TOKEN", "dev-worker-token")
CLIENT_TOKEN = os.environ.get("BLIPPY_CLIENT_TOKEN", "dev-client-token")
API_HOST = os.environ.get("HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", "8000"))
