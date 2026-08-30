"""Immutable run-directory and JSONL resume guards for the candidate probe."""
from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REQUIRED_RUN_FILES = ("config.json", "environment.txt", "run.log")


def make_run_id(prefix: str = "wsl3090_nell23k_candidate_energy") -> str:
    """Return a filesystem-safe UTC run id with process uniqueness."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{now}_{os.getpid()}"


def prepare_run_dir(root: Path, run_id: str, *, resume: bool = False) -> Path:
    """Create a run directory, refusing accidental reuse or overwrite.

    A run can only be resumed when its existing config has the same immutable
    ``run_id``. There is intentionally no overwrite mode: a new run id is the
    only way to replace an old experiment.
    """
    if not run_id or any(ch in run_id for ch in "/\\"):
        raise ValueError("run_id must be a non-empty single path component")
    run_dir = (root / run_id).resolve()
    if run_dir.exists():
        if not resume:
            raise FileExistsError(
                f"run directory already exists: {run_dir}; choose a new RUN_ID or pass --resume"
            )
        config = run_dir / "config.json"
        if not config.is_file():
            raise ValueError(f"cannot resume incomplete run without config.json: {run_dir}")
        stored = json.loads(config.read_text(encoding="utf-8"))
        if stored.get("run_id") != run_id:
            raise ValueError("resume guard failed: config.json run_id does not match directory")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
        run_dir.mkdir(parents=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    return run_dir


def completed_question_ids(path: Path) -> set[tuple[str, str, str, str]]:
    """Read completed prediction cells used by the resume guard."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, str, str]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            fields = ("question_id", "split", "adapter_control", "decoder_type")
            missing = [field for field in fields if field not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing {', '.join(missing)}")
            done.add(tuple(str(row[field]) for field in fields))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def git_commit(repository_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_environment(repository_dir: Path) -> dict[str, Any]:
    """Collect lightweight, serialisable provenance without requiring CUDA."""
    import platform
    import sys

    info: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit(repository_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # pragma: no cover - defensive provenance path
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def write_environment(path: Path, info: dict[str, Any]) -> None:
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
