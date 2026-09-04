"""Configuration validation for the Oracle Stage-2 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("format_version") != 1:
        raise ValueError("format_version must be 1")
    if not str(config.get("experiment_id", "")).endswith("2026-09-01"):
        raise ValueError("diagnostic config must be dated 2026-09-01")
    model = config["model"]
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    if model.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported model dtype")
    data = config["data"]
    for split in ("train", "validation", "test"):
        if split not in data:
            raise ValueError(f"missing data.{split}")
    training = config["training"]
    if int(training["seed"]) < 0:
        raise ValueError("training.seed must be non-negative")
    if int(training["stage2_epochs"]) < 2:
        raise ValueError("diagnostic requires at least two Stage-2 epochs")
    if training.get("stage2_scheduler") not in {"constant", "cosine"}:
        raise ValueError("stage2_scheduler must be constant or cosine")
    if int(config["lora"]["rank"]) <= 0 or float(config["lora"]["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    diagnostic = config["diagnostic"]
    if diagnostic["conditions"] != ["graph_free", "oracle_evidence"]:
        raise ValueError("diagnostic conditions must preserve graph_free/oracle_evidence order")
    if diagnostic["splits"] != ["validation", "test"]:
        raise ValueError("diagnostic splits must preserve validation/test order")
