"""Analyze candidate-energy probe predictions without changing frozen settings."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not {"question_id", "split", "adapter_control", "decoder_type", "correct"}.issubset(row):
                    raise ValueError(f"line {number} missing required prediction fields")
                rows.append(row)
    if not rows:
        raise ValueError(f"empty predictions file: {path}")
    return rows


def accuracy(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {"n": len(rows), "correct": sum(bool(r["correct"]) for r in rows), "accuracy": sum(bool(r["correct"]) for r in rows) / len(rows) if rows else 0.0}


def _binom_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    p = min(1.0, 2.0 * tail / (2**n))
    return p


def mcnemar(left: list[bool], right: list[bool]) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("paired vectors must have equal length")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    return {
        "n": len(left),
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant": discordant,
        "p_value_exact": _binom_two_sided_p(min(left_only, right_only), discordant),
    }


def bootstrap_ci(values: list[float], seed: int = 2026, samples: int = 10000) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "seed": seed, "samples": 0}
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return {"n": n, "mean": sum(values) / n, "ci95_low": lo, "ci95_high": hi, "seed": seed, "samples": samples}


def paired_effect(rows: list[dict[str, Any]], left_key: tuple[str, str, str], right_key: tuple[str, str, str]) -> dict[str, Any]:
    def index(key: tuple[str, str, str]) -> dict[tuple[str, str], bool]:
        split, control, decoder = key
        return {(str(r["question_id"]), str(r["split"])): bool(r["correct"]) for r in rows if (r["split"], r["adapter_control"], r["decoder_type"]) == key}
    left = index(left_key)
    right = index(right_key)
    keys = sorted(set(left) & set(right))
    lv, rv = [left[k] for k in keys], [right[k] for k in keys]
    mc = mcnemar(lv, rv)
    ci = bootstrap_ci([float(b) - float(a) for a, b in zip(lv, rv)])
    return {"left": ".".join(left_key), "right": ".".join(right_key), "question_count": len(keys), "left_accuracy": sum(lv) / len(lv) if lv else 0.0, "right_accuracy": sum(rv) / len(rv) if rv else 0.0, "delta_right_minus_left": (sum(rv) - sum(lv)) / len(lv) if lv else 0.0, "mcnemar": mc, "bootstrap_delta_ci95": ci}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["adapter_control"]), str(row["decoder_type"]))].append(row)
    table = []
    for (split, control, decoder), group in sorted(grouped.items()):
        metric = accuracy(group)
        table.append({"split": split, "adapter_control": control, "decoder_type": decoder, **metric, "mean_latency_seconds": sum(float(r.get("latency_seconds", 0.0)) for r in group) / len(group), "mean_peak_memory_bytes": sum(int(r.get("peak_memory_bytes", 0)) for r in group) / len(group)})
    write_csv(out / "decoder_accuracy.csv", table, ["split", "adapter_control", "decoder_type", "n", "correct", "accuracy", "mean_latency_seconds", "mean_peak_memory_bytes"])

    effects: list[dict[str, Any]] = []
    splits = sorted({str(r["split"]) for r in rows})
    controls = sorted({str(r["adapter_control"]) for r in rows})
    decoders = sorted({str(r["decoder_type"]) for r in rows})
    for split in splits:
        for decoder in decoders:
            if "correct" in controls and "none" in controls:
                effects.append(paired_effect(rows, (split, "none", decoder), (split, "correct", decoder)))
            if "free" in decoders and decoder == "score":
                for control in controls:
                    effects.append(paired_effect(rows, (split, control, "free"), (split, control, "score")))
    (out / "paired_decoder_effects.csv").write_text("", encoding="utf-8")
    effect_rows = []
    for effect in effects:
        mc, ci = effect.pop("mcnemar"), effect.pop("bootstrap_delta_ci95")
        effect_rows.append({**effect, "mcnemar_p_value": mc["p_value_exact"], "discordant": mc["discordant"], "bootstrap_ci95_low": ci["ci95_low"], "bootstrap_ci95_high": ci["ci95_high"], "bootstrap_seed": ci["seed"]})
    if effect_rows:
        write_csv(out / "paired_decoder_effects.csv", effect_rows, list(effect_rows[0]))

    # Calibration is intentionally compact: score margin and correctness for score rows.
    calibration = []
    for row in rows:
        if row["decoder_type"] != "score":
            continue
        summary = row.get("candidate_score_summary") or {}
        scores = row.get("scores") or []
        selected = summary.get("best_score")
        ranked = sorted((float(s.get("norm_logprob", float("-inf"))) for s in scores), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
        calibration.append({"question_id": row["question_id"], "split": row["split"], "adapter_control": row["adapter_control"], "correct": bool(row["correct"]), "best_score": selected, "margin": margin})
    (out / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def acc_for(split: str, control: str, decoder: str) -> float | None:
        g = grouped.get((split, control, decoder), [])
        return float(accuracy(g)["accuracy"]) if g else None
    test_correct_score = acc_for("test", "correct", "score")
    test_none_score = acc_for("test", "none", "score")
    test_free = acc_for("test", "correct", "free")
    validation_correct_score = acc_for("validation", "correct", "score")
    validation_none_score = acc_for("validation", "none", "score")
    validation_free = acc_for("validation", "correct", "free")
    score_generation_delta = (test_correct_score - test_free) if test_correct_score is not None and test_free is not None else None
    correct_none_delta = (test_correct_score - test_none_score) if test_correct_score is not None and test_none_score is not None else None
    val_correct_none = (validation_correct_score - validation_none_score) if validation_correct_score is not None and validation_none_score is not None else None
    val_score_generation = (validation_correct_score - validation_free) if validation_correct_score is not None and validation_free is not None else None
    directions = [x for x in ((correct_none_delta, val_correct_none), (score_generation_delta, val_score_generation)) if x[0] is not None and x[1] is not None]
    direction_consistent = all((test >= 0) == (validation >= 0) for test, validation in directions) if directions else False
    gate = {
        "correct_none_gain_ge_5pp": correct_none_delta is not None and correct_none_delta >= 0.05,
        "score_vs_free_gain_ge_10pp": score_generation_delta is not None and score_generation_delta >= 0.10,
        "validation_test_direction_consistent": direction_consistent,
        "wrong_or_shuffled_below_correct": False,
        "candidate_order_permutation_stable": all(bool(r.get("metadata", {}).get("order_permutation_stable", True)) for r in rows),
        "go": False,
        "reason": "STOP: shuffled adapter unavailable or one or more Go-gate conditions failed",
    }
    if any(c in controls for c in ("wrong_depth", "shuffled")):
        bad = [c for c in ("wrong_depth", "shuffled") if c in controls and acc_for("test", c, "score") is not None and test_correct_score is not None and acc_for("test", c, "score") >= test_correct_score]
        gate["wrong_or_shuffled_below_correct"] = not bad
    gate["go"] = all(bool(gate[key]) for key in ("correct_none_gain_ge_5pp", "score_vs_free_gain_ge_10pp", "validation_test_direction_consistent", "wrong_or_shuffled_below_correct", "candidate_order_permutation_stable"))
    if gate["go"]:
        gate["reason"] = "GO: all pre-registered candidate-decoder conditions passed"
    summary = {
        "count": len(rows), "groups": table, "selected_normalization": next((r.get("candidate_score_summary", {}).get("normalization") for r in rows if r.get("decoder_type") == "score"), None),
        "validation_correct_none_delta": val_correct_none, "validation_score_generation_delta": val_score_generation, "test_correct_none_delta": correct_none_delta, "test_score_generation_delta": score_generation_delta, "go_gate": gate,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(load_jsonl(args.input_file), args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
