#!/usr/bin/env python3
"""Aggregate validation-selected metrics from a fair multi-seed sweep.

The script never reads adapter weights.  It consumes only JSON summaries and
optional post-hoc candidate-ranking reports, making the aggregate safe to
commit and reproducible on a machine without the local model cache.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from priority_distill.io_utils import write_json


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _metric(summary: dict, key: str) -> float | None:
    metrics = summary.get("metrics", {})
    item = metrics.get(key)
    if isinstance(item, dict) and "accuracy" in item:
        return float(item["accuracy"])
    return None


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std_sample": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_run(run_dir: Path) -> dict:
    run_summary = _read(run_dir / "run_summary.json")
    selected = _read(run_dir / "selected_checkpoint_metrics.json")
    row = {
        "run_dir": str(run_dir),
        "protocol": run_summary["protocol"],
        "seed": int(run_summary["seed"]),
        "seed_override": run_summary.get("seed_override"),
        "selected_checkpoint": run_summary["selected_checkpoint"],
        "max_new_tokens": int(run_summary.get("config", {}).get("data", {}).get("max_new_tokens", -1))
        if isinstance(run_summary.get("config"), dict) else None,
        "graph_free_validation": _metric(selected, "graph_free_validation"),
        "graph_free_test": _metric(selected, "graph_free_test"),
    }
    # Candidate diagnostics live below candidate_diagnostics_<checkpoint> so
    # the selected checkpoint remains validation-controlled.
    diag_roots = sorted(run_dir.glob("candidate_diagnostics_*/"))
    if diag_roots:
        diag = diag_roots[-1]
        deployment = diag / "deployment" / "deployment_candidate_ranking_test.json"
        constrained = diag / "constrained" / "constrained_entity_decoding.json"
        if deployment.is_file():
            payload = _read(deployment)
            row["deployment_sequence_rank1_test"] = payload.get("metrics", {}).get("rank1_accuracy")
        if constrained.is_file():
            payload = _read(constrained)
            row["constrained_test"] = payload.get("test", {}).get("accuracy")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(path.expanduser().resolve()) for path in args.run_dir]
    grouped: dict[str, list[dict]] = {}
    for row in runs:
        grouped.setdefault(row["protocol"], []).append(row)
    aggregate = {}
    for protocol, rows in sorted(grouped.items()):
        keys = sorted({key for row in rows for key, value in row.items() if key not in {"run_dir", "protocol", "seed", "seed_override", "selected_checkpoint"} and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)})
        aggregate[protocol] = {
            "seeds": [row["seed"] for row in sorted(rows, key=lambda item: item["seed"])],
            "runs": rows,
            "metrics": {key: _stats([float(row[key]) for row in rows if row.get(key) is not None]) for key in keys},
        }
    payload = {
        "format_version": 1,
        "purpose": "fair_direct_vs_joint_multi_seed_aggregate",
        "selection_rule": "each run selects checkpoint by validation only; test is read after selection",
        "runs": runs,
        "by_protocol": aggregate,
    }
    write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
