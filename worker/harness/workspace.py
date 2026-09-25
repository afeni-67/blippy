"""Isolated temporary workspace for one job."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List


class Workspace:
    def __init__(self, files: List[dict] | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="blippy-ws-"))
        self._initial: Dict[str, str] = {}
        for f in files or []:
            path = f.get("path") or ""
            content = f.get("content") or ""
            if not path or ".." in path:
                continue
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            self._initial[path] = content

    def resolve(self, rel: str) -> Path:
        rel = rel.lstrip("/").replace("\\", "/")
        if ".." in rel.split("/"):
            raise ValueError("path escape blocked")
        return self.root / rel

    def list_dir(self, rel: str = ".") -> List[str]:
        base = self.resolve(rel) if rel not in (".", "") else self.root
        if not base.exists():
            return []
        out = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return out[:200]

    def read_file(self, rel: str, max_chars: int = 12000) -> str:
        p = self.resolve(rel)
        if not p.is_file():
            return f"[missing file: {rel}]"
        data = p.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[:max_chars] + "\n...[truncated]"
        return data

    def write_file(self, rel: str, content: str) -> None:
        p = self.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def run_command(self, command: str, timeout: int = 60) -> dict:
        import subprocess

        # Block obviously dangerous patterns
        lowered = command.lower()
        for bad in ("rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot"):
            if bad in lowered:
                return {"exit_code": 1, "stdout": "", "stderr": "command blocked by policy"}
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
            return {
                "exit_code": r.returncode,
                "stdout": (r.stdout or "")[:8000],
                "stderr": (r.stderr or "")[:4000],
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}

    def diff_changes(self) -> List[dict]:
        """Return create/modify/delete ops vs initial snapshot."""
        current: Dict[str, str] = {}
        for p in self.root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self.root))
                try:
                    current[rel] = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        changes = []
        for path, content in current.items():
            if path not in self._initial:
                changes.append({"type": "file_change", "operation": "create", "path": path, "content": content})
            elif self._initial[path] != content:
                changes.append({"type": "file_change", "operation": "modify", "path": path, "content": content})
        for path in self._initial:
            if path not in current:
                changes.append({"type": "file_change", "operation": "delete", "path": path, "content": ""})
        return changes

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
