#!/usr/bin/env bash
# Install official Qwen Code CLI (the agent harness).
# Headless usage: qwen -p "..." ; local models via OPENAI_BASE_URL/MODEL.
set -euo pipefail
if ! command -v node >/dev/null 2>&1; then
  echo "node not found; install Node.js 22+ first (e.g. actions/setup-node)"
  exit 1
fi
echo "node: $(node --version) npm: $(npm --version)"
npm install -g @qwen-code/qwen-code@latest
qwen --version
echo "Installed qwen CLI"
