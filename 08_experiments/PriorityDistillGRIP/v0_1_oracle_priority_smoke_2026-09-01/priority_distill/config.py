"""Configuration loading and registered-design invariants."""

from __future__ import annotations

import json
from pathlib import Path

from .supervision import METHODS, STAGE1_METHODS


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("format_version") != 1:
        raise ValueError("format_version must be 1")
    methods = tuple(config["methods"])
    if methods != METHODS:
        raise ValueError(f"methods must preserve the registered order: {METHODS}")
    if config["training"]["stage1_reference_method"] not in STAGE1_METHODS:
        raise ValueError("stage1_reference_method must be a stage-1 method")
    if config["training"]["stage1_reference_method"] != "all_paths_equal_token":
        raise ValueError("v0.1 uses all_paths_equal_token as the maximum-information token reference")
    if not set(config["training"]["smoke_seeds"]).issubset(config["training"]["seeds"]):
        raise ValueError("smoke_seeds must be a subset of seeds")
    if int(config["candidates"]["distractor_count"]) != 3:
        raise ValueError("v0.1 registers exactly three distractors")
    if int(config["lora"]["rank"]) <= 0 or float(config["lora"]["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    for key in ("stage1_tokens_per_optimizer_step", "stage2_tokens_per_optimizer_step", "stage2_epochs"):
        if int(config["training"][key]) <= 0:
            raise ValueError(f"{key} must be positive")
    token_gap = float(config["gate"].get("max_stage1_token_relative_gap", 0.01))
    if not 0.0 <= token_gap < 1.0:
        raise ValueError("max_stage1_token_relative_gap must be in [0, 1)")
    truncation_rate = float(config["gate"].get("max_stage1_truncated_example_rate", 0.0))
    if not 0.0 <= truncation_rate <= 1.0:
        raise ValueError("max_stage1_truncated_example_rate must be in [0, 1]")
