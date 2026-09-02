#!/usr/bin/env python3
"""Run post-hoc diagnostics on an existing adapter checkpoint.

This does not train or modify the checkpoint.  It answers three concrete
questions: did the joint model memorize the train set, does a supplied trace
help, and does exposing the terminal remove the bottleneck?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

import torch

from priority_distill.config import load_config
from priority_distill.experiment import generate_predictions, load_runtime
from priority_distill.io_utils import write_json, write_jsonl
from priority_distill.modules import load_adapter_state_dict
from priority_distill.records import load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    splits, _, split_audit = load_splits(REPO, config["data"])

    model, tokenizer, injection, device, dtype, resolved_model = load_runtime(config, args.model_name_or_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_adapter_state_dict(model, state)

    results: dict[str, dict] = {}
    # Same graph-free prompt as the test, now evaluated on training examples.
    train_predictions, train_metrics = generate_predictions(
        model, tokenizer, splits["train"], splits["train"], config, device, dtype,
        "graph_free", "stage2_epoch8_train_audit",
    )
    write_jsonl(output_dir / "predictions_train_graph_free.jsonl", train_predictions)
    results["graph_free_train"] = train_metrics

    for condition in ("gold_trace", "gold_trace_unmasked"):
        for split in ("validation", "test"):
            predictions, metrics = generate_predictions(
                model, tokenizer, splits[split], splits["train"], config, device, dtype,
                condition, f"stage2_epoch8_{condition}_{split}_audit",
            )
            write_jsonl(output_dir / f"predictions_{condition}_{split}.jsonl", predictions)
            results[f"{condition}_{split}"] = metrics

    report = {
        "format_version": 1,
        "purpose": "posthoc_joint_checkpoint_audit",
        "checkpoint": str(checkpoint_path),
        "model": resolved_model,
        "device": str(device),
        "dtype": str(dtype),
        "injection": {
            "target_modules": list(injection.target_modules),
            "trainable_parameters": injection.trainable_parameters,
        },
        "data": split_audit,
        "metrics": results,
        "interpretation": {
            "train_vs_test": "high train but low test indicates memorization/poor compositional generalization",
            "gold_trace_masked": "tests whether correct intermediate nodes are sufficient when terminal remains hidden",
            "gold_trace_unmasked": "oracle-only sanity check; visible terminal should make copying possible, not a graph-free score",
        },
    }
    write_json(output_dir / "checkpoint_audit.json", report)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
