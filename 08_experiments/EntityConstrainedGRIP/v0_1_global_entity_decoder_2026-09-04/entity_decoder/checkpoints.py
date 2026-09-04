"""Frozen checkpoint registry with content-hash verification."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import sha256_file


def resolve_checkpoint_registry(entries: Iterable[dict], repo_root: Path, require_files: bool) -> list[dict]:
    resolved = []
    for raw in entries:
        entry = dict(raw)
        if entry["kind"] == "no_adapter":
            entry.update({"resolved_path": None, "observed_sha256": None, "status": "no_adapter"})
            resolved.append(entry)
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            if require_files:
                raise FileNotFoundError(f"missing checkpoint: {path}")
            entry.update({"resolved_path": str(path), "observed_sha256": None, "status": "missing"})
            resolved.append(entry)
            continue
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {entry['name']}: expected {entry['sha256']}, observed {observed}")
        entry.update({"resolved_path": str(path), "observed_sha256": observed, "status": "verified"})
        resolved.append(entry)
    return resolved
