#!/usr/bin/env bash
# Install official Codex CLI binary (actual OpenAI Codex harness).
set -euo pipefail
DEST="${CODEX_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$DEST"
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ASSET="codex-x86_64-unknown-linux-musl.tar.gz"; MATCH="x86_64"; FILEMATCH="x86-64" ;;
  aarch64|arm64) ASSET="codex-aarch64-unknown-linux-musl.tar.gz"; MATCH="aarch64"; FILEMATCH="aarch64" ;;
  *) echo "unsupported arch $ARCH"; exit 1 ;;
esac
URL="https://github.com/openai/codex/releases/latest/download/${ASSET}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Downloading $URL"
curl -fsSL "$URL" -o "$TMP/codex.tgz"
tar -xzf "$TMP/codex.tgz" -C "$TMP"
echo "Tarball contents:"
find "$TMP" -type f | head -20
# The tarball may bundle several builds; select the ELF matching the host arch.
BIN=""
if command -v file >/dev/null 2>&1; then
  while IFS= read -r f; do
    if file "$f" | grep -q "ELF.*${FILEMATCH}"; then BIN="$f"; break; fi
  done < <(find "$TMP" -type f -executable)
fi
if [ -z "$BIN" ]; then
  BIN=$(find "$TMP" -type f -name "*${MATCH}*" | head -1)
fi
if [ -z "$BIN" ]; then
  BIN=$(find "$TMP" -type f -name 'codex*' | head -1)
fi
[ -n "$BIN" ] || { echo "no codex binary found in $ASSET"; exit 1; }
echo "Selected: $BIN"
if command -v file >/dev/null 2>&1; then file "$BIN"; fi
install -m 755 "$BIN" "$DEST/codex"
"$DEST/codex" --version
echo "Installed $DEST/codex"
