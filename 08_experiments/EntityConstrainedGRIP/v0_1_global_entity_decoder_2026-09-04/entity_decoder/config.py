"""Configuration loader and validation-first graph-free protocol invariants."""
from __future__ import annotations

import json
import re
from pathlib import Path

ALLOWED_DECODERS = ("D0", "D1", "D2", "D3")
ALLOWED_PROMPTS = ("answer_only", "joint_answer_slot")
ALLOWED_ROLES = ("primary_direct", "reference_more_qa", "secondary_joint", "control")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("format_version") != 1:
        raise ValueError("format_version must be 1")
    required_data = {"train_graph", "train", "validation", "test"}
    if required_data - set(config.get("data", {})):
        raise ValueError("data must define train_graph/train/validation/test")
    if config["data"].get("vocabulary_source_split", "train") != "train":
        raise ValueError("entity vocabulary must come from the training graph")

    decoders = tuple(config.get("decoders", ()))
    if decoders != ("D0", "D1"):
        raise ValueError("initial Phase-A decoders must be exactly D0/D1")
    splits = tuple(config.get("evaluation", {}).get("splits", ()))
    if splits != ("validation",):
        raise ValueError("initial Phase-A evaluation must be validation-only")

    entries = config.get("checkpoints", [])
    if not entries:
        raise ValueError("checkpoints must be non-empty")
    names: set[str] = set()
    registry: dict[str, dict] = {}
    for entry in entries:
        name = entry["name"]
        if name in names:
            raise ValueError(f"duplicate checkpoint name: {name}")
        names.add(name)
        registry[name] = entry
        if entry["kind"] not in {"no_adapter", "adapter"}:
            raise ValueError("checkpoint kind must be no_adapter or adapter")
        if entry["prompt_protocol"] not in ALLOWED_PROMPTS:
            raise ValueError(f"prompt_protocol must be drawn from {ALLOWED_PROMPTS}")
        if entry.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"checkpoint role must be drawn from {ALLOWED_ROLES}")
        if entry["kind"] == "adapter":
            if not {"path", "sha256"}.issubset(entry):
                raise ValueError("adapter checkpoints require path and sha256")
            if not _SHA256.fullmatch(str(entry["sha256"])):
                raise ValueError(f"invalid checkpoint sha256: {name}")

    initial_names = list(config["evaluation"].get("initial_checkpoints", ()))
    if not initial_names or len(initial_names) != len(set(initial_names)):
        raise ValueError("initial_checkpoints must be unique and non-empty")
    unknown_initial = set(initial_names) - names
    if unknown_initial:
        raise ValueError(f"unknown initial checkpoints: {sorted(unknown_initial)}")
    for name in initial_names:
        source = registry[name].get("d0_predictions", {}).get("validation")
        if not source or not {"path", "sha256"}.issubset(source):
            raise ValueError(f"initial checkpoint requires validation D0 artifact: {name}")
        if not _SHA256.fullmatch(str(source["sha256"])):
            raise ValueError(f"invalid D0 artifact sha256: {name}")

    gate = config.get("gate", {})
    if gate.get("split") != "validation" or gate.get("decoder") != "D1":
        raise ValueError("gate must compare D0/D1 on validation")
    primary = list(gate.get("primary_checkpoints", ()))
    minimum = int(gate.get("minimum_primary_checkpoints", 2))
    if minimum < 2 or len(primary) < minimum or len(primary) != len(set(primary)):
        raise ValueError("gate requires at least two unique primary direct checkpoints")
    unknown_gate = set(primary + list(gate.get("reference_checkpoints", ()))) - names
    if unknown_gate:
        raise ValueError(f"unknown gate checkpoints: {sorted(unknown_gate)}")
    if any(registry[name].get("role") != "primary_direct" for name in primary):
        raise ValueError("all primary gate checkpoints must have role=primary_direct")
    references = list(gate.get("reference_checkpoints", ()))
    if not references or any(registry[name].get("role") != "reference_more_qa" for name in references):
        raise ValueError("gate requires at least one reference_more_qa checkpoint")
    if not set(primary + references).issubset(initial_names):
        raise ValueError("all gate checkpoints must be included in initial_checkpoints")

    candidates = config.get("evaluation", {}).get("d2_score_normalization_candidates", [])
    if not set(candidates).issubset({"sum", "mean"}) or not candidates:
        raise ValueError("D2 normalization candidates must be non-empty and drawn from sum/mean")
    follow_up = config.get("follow_up_policy", {})
    required_follow_up = {"D2", "D3", "test"}
    if required_follow_up - set(follow_up):
        raise ValueError("follow_up_policy must register D2, D3, and test handling")
