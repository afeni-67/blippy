"""Isolated temp workspace + final diff. Mutations are done by Codex tools, not Blippy."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List


class Workspace:
    def __init__(self, files: List[dict] | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="blippy-ws-"))
        self._initial: Dict[str, str] = {}
        for f in files or []:
            path = (f.get("path") or "").lstrip("/").replace("\\", "/")
            if not path or ".." in path.split("/"):
                continue
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = f.get("content") or ""
            dest.write_text(content, encoding="utf-8")
            self._initial[path] = content

    def snapshot_files(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in self.root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self.root))
                try:
                    out[rel] = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        return out

    def diff_changes(self) -> List[dict]:
        current = self.snapshot_files()
        changes = []
        for path, content in current.items():
            if path not in self._initial:
                changes.append(
                    {"type": "file_change", "operation": "create", "path": path, "content": content}
                )
            elif self._initial[path] != content:
                changes.append(
                    {"type": "file_change", "operation": "modify", "path": path, "content": content}
                )
        for path in self._initial:
            if path not in current:
                changes.append(
                    {"type": "file_change", "operation": "delete", "path": path, "content": ""}
                )
        return changes

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
