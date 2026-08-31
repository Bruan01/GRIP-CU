#!/usr/bin/env python3
"""Validate v0.2 without importing torch, transformers, or downloading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from structured_lora.config import load_config, routing_kwargs
from structured_lora.io_utils import sha256_file, write_json
from structured_lora.records import load_jsonl, validate_splits
from structured_lora.routing import build_route_masks, masks_are_nested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_prefix_smoke.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "artifacts/setup_audit.json")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    paths = {name: REPO_ROOT / config["data"][name] for name in ("train", "validation", "test")}
    splits = {name: load_jsonl(path) for name, path in paths.items()}
    split_audit = validate_splits(splits)
    routes = routing_kwargs(config)
    masks = {
        method: {
            str(depth): build_route_masks(method, [depth], groups=config["lora"]["groups"], **routes).forward[0]
            for depth in range(1, 5)
        }
        for method in config["methods"]
    }
    payload = {
        "status": "READY_FOR_WSL_GPU",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "data": {
            **split_audit,
            "sha256": {name: sha256_file(path) for name, path in paths.items()},
        },
        "methods": config["methods"],
        "route_masks": masks,
        "invariants": {
            "equal_total_rank": config["lora"]["groups"] * config["lora"]["group_rank"] == config["lora"]["total_rank"],
            "ordered_prefix_nested": masks_are_nested({int(k): v for k, v in masks["ordered_prefix"].items()}),
            "random_group_order_is_nested_symmetry": masks_are_nested({int(k): v for k, v in masks["random_group_order"].items()}),
            "non_nested_control_breaks_nesting": not masks_are_nested({int(k): v for k, v in masks["non_nested_random_masks"].items()}),
            "same_target_modules_as_grip": config["lora"]["target_modules"] == ["down_proj", "up_proj", "gate_proj"],
        },
    }
    if not all(payload["invariants"].values()):
        raise AssertionError(payload["invariants"])
    write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "methods": len(config["methods"]),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
