#!/usr/bin/env bash
# Install official Codex CLI binary (actual OpenAI Codex harness).
set -euo pipefail
DEST="${CODEX_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$DEST"
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ASSET="codex-x86_64-unknown-linux-musl.tar.gz" ;;
  aarch64|arm64) ASSET="codex-aarch64-unknown-linux-musl.tar.gz" ;;
  *) echo "unsupported arch $ARCH"; exit 1 ;;
esac
URL="https://github.com/openai/codex/releases/latest/download/${ASSET}"
TMP=$(mktemp -d)
echo "Downloading $URL"
curl -fsSL "$URL" -o "$TMP/codex.tgz"
tar -xzf "$TMP/codex.tgz" -C "$TMP"
BIN=$(find "$TMP" -type f -name 'codex*' | head -1)
install -m 755 "$BIN" "$DEST/codex"
"$DEST/codex" --version || true
echo "Installed $DEST/codex"
