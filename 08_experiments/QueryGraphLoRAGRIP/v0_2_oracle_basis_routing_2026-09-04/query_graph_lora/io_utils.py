"""Deterministic file and environment helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_info(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def environment_snapshot(repo_root: Path) -> dict:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": git_info(repo_root),
        "env": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "WSL_DISTRO_NAME")
        },
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["gpu_total_memory"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        payload["torch"] = "not-installed"
    try:
        import transformers

        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["transformers"] = "not-installed"
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
