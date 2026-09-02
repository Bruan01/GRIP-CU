#!/usr/bin/env python3
"""Re-evaluate existing checkpoints with a sufficient generation budget.

The v0.1.6 joint target is much longer than 24 generated tokens for most
examples.  This script does not retrain: it reuses ignored local adapters,
selects by corrected graph-free validation, then evaluates the selected
checkpoint on all diagnostic conditions.
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name-or-path", required=True)
    args = ap.parse_args()

    config_path = args.config.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    splits, _, data_audit = load_splits(REPO, config["data"])
    model, tokenizer, injection, device, dtype, resolved_model = load_runtime(config, args.model_name_or_path)

    names = ["stage1"] + [f"stage2_epoch{i}" for i in range(1, int(config["training"]["stage2_epochs"]) + 1)]
    curve = []
    for name in names:
        path = checkpoint_root / name / "adapter_model.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        load_adapter_state_dict(model, torch.load(path, map_location="cpu", weights_only=True))
        val_predictions, val_metrics = generate_predictions(
            model, tokenizer, splits["validation"], splits["train"], config, device, dtype,
            "graph_free", f"corrected_{name}_validation",
        )
        test_predictions, test_metrics = generate_predictions(
            model, tokenizer, splits["test"], splits["train"], config, device, dtype,
            "graph_free", f"corrected_{name}_test",
        )
        write_jsonl(out / f"predictions_{name}_graph_free_validation.jsonl", val_predictions)
        write_jsonl(out / f"predictions_{name}_graph_free_test.jsonl", test_predictions)
        curve.append({"checkpoint": name, "validation": val_metrics, "test": test_metrics})
        print(f"{name}: validation={val_metrics['accuracy']:.4f} test={test_metrics['accuracy']:.4f}")

    selected = max(curve, key=lambda item: (item["validation"]["accuracy"], -names.index(item["checkpoint"])))
    selected_name = selected["checkpoint"]
    selected_path = checkpoint_root / selected_name / "adapter_model.pt"
    load_adapter_state_dict(model, torch.load(selected_path, map_location="cpu", weights_only=True))
    selected_metrics = {}
    for condition in ("graph_free", "gold_trace", "gold_trace_unmasked", "oracle_evidence"):
        for split in ("validation", "test"):
            if condition == "graph_free":
                # Reuse the curve outputs/metrics to avoid a duplicate decode.
                selected_metrics[f"{condition}_{split}"] = selected[split]
                continue
            predictions, metrics = generate_predictions(
                model, tokenizer, splits[split], splits["train"], config, device, dtype,
                condition, f"corrected_{selected_name}_{condition}_{split}",
            )
            write_jsonl(out / f"predictions_{selected_name}_{condition}_{split}.jsonl", predictions)
            selected_metrics[f"{condition}_{split}"] = metrics

    report = {
        "format_version": 1,
        "purpose": "corrected_posthoc_checkpoint_curve",
        "config": str(config_path),
        "generation_budget": int(config["data"]["max_new_tokens"]),
        "checkpoint_root": str(checkpoint_root),
        "selected_checkpoint": selected_name,
        "selection_rule": "max corrected graph_free validation accuracy; test read after selection",
        "model": resolved_model,
        "device": str(device),
        "dtype": str(dtype),
        "data": data_audit,
        "curve": curve,
        "selected_metrics": selected_metrics,
        "diagnostic_warning": "gold_trace_unmasked is oracle-only and is not a graph-free score",
    }
    write_json(out / "corrected_checkpoint_curve.json", report)
    write_json(out / "corrected_selected_checkpoint_metrics.json", {
        "selected_checkpoint": selected_name,
        "generation_budget": int(config["data"]["max_new_tokens"]),
        "selection_rule": report["selection_rule"],
        "metrics": selected_metrics,
    })
    print(json.dumps({"selected_checkpoint": selected_name, "metrics": selected_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
