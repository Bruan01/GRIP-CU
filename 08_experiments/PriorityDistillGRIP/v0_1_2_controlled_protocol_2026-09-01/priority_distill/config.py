"""Configuration validation for controlled direct-vs-two-stage experiments."""
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
        raise ValueError("controlled config must be dated 2026-09-01")
    if config.get("protocol") not in {"direct_answer_only", "oracle_two_stage"}:
        raise ValueError("protocol must be direct_answer_only or oracle_two_stage")
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
    if int(training["tokens_per_optimizer_step"]) <= 0:
        raise ValueError("tokens_per_optimizer_step must be positive")
    if int(training["stage2_epochs"]) != 8:
        raise ValueError("controlled protocol fixes stage2_epochs at 8")
    for key in ("first_segment_scheduler", "answer_only_scheduler"):
        if training.get(key) not in {"constant", "cosine"}:
            raise ValueError(f"{key} must be constant or cosine")
    for key in ("first_segment_learning_rate", "answer_only_learning_rate"):
        if float(training.get(key, 0.0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["lora"]["rank"]) <= 0 or float(config["lora"]["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    diagnostic = config["diagnostic"]
    if diagnostic["conditions"] != ["graph_free", "oracle_evidence"]:
        raise ValueError("diagnostic conditions must preserve graph_free/oracle_evidence order")
    if diagnostic["splits"] != ["validation", "test"]:
        raise ValueError("diagnostic splits must preserve validation/test order")
