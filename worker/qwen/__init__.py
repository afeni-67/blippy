"""Local Qwen inference — runs ONLY on the GitHub Actions worker."""
from .model_runner import download_model, load_llm, model_path
from .inference import chat
