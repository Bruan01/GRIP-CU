"""Configuration validation for v0.1.3 candidate-selection experiments."""
from __future__ import annotations

import json
from pathlib import Path

from .controlled import PROTOCOLS


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("format_version") != 1:
        raise ValueError("format_version must be 1")
    if not str(config.get("experiment_id", "")).endswith("2026-09-01"):
        raise ValueError("config must be dated 2026-09-01")
    if config.get("protocol") not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
    model = config["model"]
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    if model.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported model dtype")
    data = config["data"]
    for split in ("train", "validation", "test"):
        if split not in data:
            raise ValueError(f"missing data.{split}")
    candidates = config["candidates"]
    if int(candidates["distractor_count"]) != 3:
        raise ValueError("v0.1.3 fixes distractor_count at 3")
    if not candidates.get("same_depth_only") or not candidates.get("different_answer_only") or not candidates.get("train_split_only"):
        raise ValueError("candidate pools must be same-depth, different-answer, train-only")
    training = config["training"]
    if int(training["seed"]) < 0:
        raise ValueError("training.seed must be non-negative")
    if int(training["tokens_per_optimizer_step"]) <= 0:
        raise ValueError("tokens_per_optimizer_step must be positive")
    if int(training["stage2_epochs"]) != 8:
        raise ValueError("v0.1.3 fixes stage2_epochs at 8")
    replay = float(training.get("stage2_replay_ratio", PROTOCOLS[config["protocol"]]["stage2_replay_ratio"]))
    if not 0.0 <= replay < 1.0:
        raise ValueError("stage2_replay_ratio must be in [0, 1)")
    replay_protocols = {"candidate_selection_anti_copy_replay", "candidate_index_anti_copy_replay"}
    if config["protocol"] in replay_protocols and abs(replay - 0.2) > 1e-9:
        raise ValueError(f"{config['protocol']} fixes stage2_replay_ratio at 0.2")
    if config["protocol"] not in replay_protocols and replay != 0.0:
        raise ValueError("only candidate_selection_anti_copy_replay or candidate_index_anti_copy_replay may use Stage-2 evidence replay")
    for key in ("first_segment_scheduler", "answer_only_scheduler"):
        if training.get(key) not in {"constant", "cosine"}:
            raise ValueError(f"{key} must be constant or cosine")
    for key in ("first_segment_learning_rate", "answer_only_learning_rate"):
        if float(training.get(key, 0.0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["lora"]["rank"]) <= 0 or float(config["lora"]["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    diagnostic = config["diagnostic"]
    allowed_conditions = ["graph_free", "oracle_evidence"]
    explicit_conditions = ["graph_free", "candidate_selection", "candidate_selection_terminal_masked", "oracle_evidence"]
    if diagnostic["conditions"] not in (allowed_conditions, explicit_conditions):
        raise ValueError("diagnostic conditions must be graph_free/oracle_evidence or the explicit candidate diagnostics order")
    if config["protocol"].startswith("candidate_index_") and diagnostic["conditions"] != explicit_conditions:
        raise ValueError("candidate_index protocols require explicit candidate diagnostics")
    if diagnostic["splits"] != ["validation", "test"]:
        raise ValueError("diagnostic splits must preserve validation/test order")
