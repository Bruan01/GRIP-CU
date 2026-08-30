#!/usr/bin/env python3
"""Paired statistical analysis for RecurrentGRIP v1.1.1 diagnostic cross.

Uses only the Python standard library so the analysis is reproducible on macOS
and WSL without the training environment. It never mutates the source run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def paired_bootstrap_ci(a: list[int], b: list[int], *, seed: int = 20260830, samples: int = 20000) -> tuple[float, float]:
    """95% percentile CI for mean(a-b), resampling paired question rows."""
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    paired = [x - y for x, y in zip(a, b)]
    for _ in range(samples):
        diffs.append(sum(paired[rng.randrange(n)] for _ in range(n)) / n)
    diffs.sort()
    return percentile(diffs, 0.025), percentile(diffs, 0.975)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value using Binomial(n=b+c, p=.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * lower)


def norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def transition_metrics(row: dict) -> dict[str, float | None]:
    states = row.get("step_pooled_hidden_states") or []
    if len(states) < 2:
        return {"norm_ratio": None, "cosine": None, "relative_delta": None}
    x, y = states[0], states[-1]
    nx, ny = norm(x), norm(y)
    dot = sum(a * b for a, b in zip(x, y))
    delta = math.sqrt(sum((b - a) ** 2 for a, b in zip(x, y)))
    return {
        "norm_ratio": ny / nx if nx else None,
        "cosine": dot / (nx * ny) if nx and ny else None,
        "relative_delta": delta / nx if nx else None,
    }


def row_key(row: dict) -> tuple[str, int, int, str, str]:
    return (
        row["metadata"]["split"],
        int(row["recurrent_train_k"]),
        int(row["recurrence_k"]),
        row["adapter_control"],
        row["question_id"],
    )


def condition_key(row: dict) -> tuple[str, int, int, str]:
    k = row_key(row)
    return k[:4]


def compare(
    label: str,
    left_rows: list[dict],
    right_rows: list[dict],
    *,
    left_name: str,
    right_name: str,
) -> dict:
    left = {r["question_id"]: r for r in left_rows}
    right = {r["question_id"]: r for r in right_rows}
    if set(left) != set(right):
        raise ValueError(f"Unpaired comparison {label}: question ids differ")
    qids = sorted(left)
    a = [int(bool(left[q]["correct"])) for q in qids]
    b = [int(bool(right[q]["correct"])) for q in qids]
    both_correct = sum(x and y for x, y in zip(a, b))
    left_only = sum(x and not y for x, y in zip(a, b))
    right_only = sum(not x and y for x, y in zip(a, b))
    both_wrong = len(a) - both_correct - left_only - right_only
    delta = mean(a) - mean(b)
    lo, hi = paired_bootstrap_ci(a, b)
    return {
        "comparison": label,
        "left": left_name,
        "right": right_name,
        "n": len(a),
        "left_correct": sum(a),
        "right_correct": sum(b),
        "left_accuracy": mean(a),
        "right_accuracy": mean(b),
        "delta_accuracy": delta,
        "delta_pp": 100 * delta,
        "bootstrap_95ci_low_pp": 100 * lo,
        "bootstrap_95ci_high_pp": 100 * hi,
        "both_correct": both_correct,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 768:
        raise ValueError(f"Expected 768 predictions, found {len(rows)}")
    indexed = {row_key(r): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate condition/question rows found")

    groups: dict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[condition_key(row)].append(row)

    expected_counts = {"validation": 32, "test": 64}
    for (split, train_k, eval_k, adapter), rs in groups.items():
        if len(rs) != expected_counts[split]:
            raise ValueError(f"Bad count for {(split, train_k, eval_k, adapter)}: {len(rs)}")

    condition_rows = []
    for (split, train_k, eval_k, adapter), rs in sorted(groups.items()):
        dyn = [transition_metrics(r) for r in rs]
        condition_rows.append({
            "split": split,
            "train_k": train_k,
            "eval_k": eval_k,
            "adapter": adapter,
            "n": len(rs),
            "correct": sum(bool(r["correct"]) for r in rs),
            "accuracy": mean(int(bool(r["correct"])) for r in rs),
            "candidate_hit": sum(bool(r["response_in_candidates"]) for r in rs),
            "candidate_hit_rate": mean(int(bool(r["response_in_candidates"])) for r in rs),
            "eos_rate": mean(int(bool(r["ended_with_eos"])) for r in rs),
            "mean_generated_tokens": mean(float(r["generated_token_count"]) for r in rs),
            "mean_latency_seconds": mean(float(r["latency_seconds"]) for r in rs),
            "mean_norm_ratio": mean(float(d["norm_ratio"]) for d in dyn if d["norm_ratio"] is not None),
            "mean_cosine": mean(float(d["cosine"]) for d in dyn if d["cosine"] is not None),
            "mean_relative_delta": mean(float(d["relative_delta"]) for d in dyn if d["relative_delta"] is not None),
        })
    write_csv(out / "condition_metrics.csv", condition_rows)

    comparisons = []
    # Adapter correct vs none, per split and train/eval depth.
    for split in ("validation", "test"):
        for train_k in (1, 2):
            for eval_k in (1, 2):
                comparisons.append(compare(
                    f"adapter_effect|split={split}|train_k={train_k}|eval_k={eval_k}",
                    groups[(split, train_k, eval_k, "correct")],
                    groups[(split, train_k, eval_k, "none")],
                    left_name="correct_adapter",
                    right_name="no_adapter",
                ))
    # Eval depth effect K2 vs K1.
    for split in ("validation", "test"):
        for train_k in (1, 2):
            for adapter in ("correct", "none"):
                comparisons.append(compare(
                    f"eval_depth_effect|split={split}|train_k={train_k}|adapter={adapter}",
                    groups[(split, train_k, 2, adapter)],
                    groups[(split, train_k, 1, adapter)],
                    left_name="eval_k2",
                    right_name="eval_k1",
                ))
    # Train depth effect K2 vs K1 at fixed eval depth.
    for split in ("validation", "test"):
        for eval_k in (1, 2):
            for adapter in ("correct", "none"):
                comparisons.append(compare(
                    f"train_depth_effect|split={split}|eval_k={eval_k}|adapter={adapter}",
                    groups[(split, 2, eval_k, adapter)],
                    groups[(split, 1, eval_k, adapter)],
                    left_name="train_k2",
                    right_name="train_k1",
                ))
    write_csv(out / "paired_comparisons.csv", comparisons)
    (out / "mcnemar_tests.json").write_text(json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # K1 -> K2 transitions and representative failure/new-solve examples.
    transition_rows = []
    examples = []
    for split in ("validation", "test"):
        for train_k in (1, 2):
            for adapter in ("correct", "none"):
                k1 = {r["question_id"]: r for r in groups[(split, train_k, 1, adapter)]}
                k2 = {r["question_id"]: r for r in groups[(split, train_k, 2, adapter)]}
                counts = Counter()
                local_examples = defaultdict(list)
                for qid in sorted(k1):
                    a, b = bool(k1[qid]["correct"]), bool(k2[qid]["correct"])
                    name = ("correct" if a else "wrong") + "_to_" + ("correct" if b else "wrong")
                    counts[name] += 1
                    if name in {"correct_to_wrong", "wrong_to_correct"} and len(local_examples[name]) < 8:
                        d = transition_metrics(k2[qid])
                        local_examples[name].append({
                            "split": split,
                            "train_k": train_k,
                            "adapter": adapter,
                            "transition": name,
                            "question_id": qid,
                            "true_hop": k1[qid].get("true_hop"),
                            "target": k1[qid]["target"],
                            "k1_response": k1[qid]["response"],
                            "k2_response": k2[qid]["response"],
                            "k1_in_candidates": k1[qid]["response_in_candidates"],
                            "k2_in_candidates": k2[qid]["response_in_candidates"],
                            **d,
                        })
                transition_rows.append({
                    "split": split,
                    "train_k": train_k,
                    "adapter": adapter,
                    "n": len(k1),
                    **{name: counts[name] for name in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")},
                    "net_k2_gain": counts["wrong_to_correct"] - counts["correct_to_wrong"],
                    "mcnemar_exact_p": exact_mcnemar_p(counts["correct_to_wrong"], counts["wrong_to_correct"]),
                })
                for vals in local_examples.values():
                    examples.extend(vals)
    write_csv(out / "k1_k2_transitions.csv", transition_rows)
    with (out / "failure_transition_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Per-transition state metrics: does a larger state move identify K1->K2 failures?
    state_by_transition = []
    for split in ("validation", "test"):
        for train_k in (1, 2):
            for adapter in ("correct", "none"):
                k1 = {r["question_id"]: r for r in groups[(split, train_k, 1, adapter)]}
                k2 = {r["question_id"]: r for r in groups[(split, train_k, 2, adapter)]}
                buckets = defaultdict(list)
                for qid in k1:
                    a, b = bool(k1[qid]["correct"]), bool(k2[qid]["correct"])
                    name = ("correct" if a else "wrong") + "_to_" + ("correct" if b else "wrong")
                    buckets[name].append(transition_metrics(k2[qid]))
                for name, vals in sorted(buckets.items()):
                    state_by_transition.append({
                        "split": split,
                        "train_k": train_k,
                        "adapter": adapter,
                        "transition": name,
                        "n": len(vals),
                        "mean_norm_ratio": mean(v["norm_ratio"] for v in vals if v["norm_ratio"] is not None),
                        "mean_cosine": mean(v["cosine"] for v in vals if v["cosine"] is not None),
                        "mean_relative_delta": mean(v["relative_delta"] for v in vals if v["relative_delta"] is not None),
                    })
    write_csv(out / "state_by_transition.csv", state_by_transition)

    audit = json.loads((run_dir / "cross_run_audit.json").read_text(encoding="utf-8"))
    summary = {
        "source_run": str(run_dir),
        "audit_status": audit.get("status"),
        "prediction_count": len(rows),
        "condition_count": len(groups),
        "selection_sha256": audit.get("selection_sha256"),
        "condition_metrics_file": "condition_metrics.csv",
        "paired_comparisons_file": "paired_comparisons.csv",
        "transition_file": "k1_k2_transitions.csv",
        "state_by_transition_file": "state_by_transition.csv",
        "bootstrap_samples": 20000,
        "bootstrap_seed": 20260830,
    }
    (out / "analysis_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
