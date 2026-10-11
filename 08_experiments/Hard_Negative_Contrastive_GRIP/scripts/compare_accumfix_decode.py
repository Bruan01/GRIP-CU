"""Validate full decode inputs and compare matched predictions for two fixed adapters."""
import argparse
import json
from pathlib import Path

import numpy as np

HNG = Path(__file__).resolve().parents[1]
VARIANTS = ("random_k", "calibrated")


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate question IDs: {path}")
    return indexed


def samples():
    payload = json.loads((HNG / "data/nell23k/recurrent_relation_prediction_full.aligned.json").read_text())
    record = payload[0] if isinstance(payload, list) else payload
    rows = [s for s in record["recurrent_questions"] if s["split"] in ("test", "validation")]
    indexed = {s["question_id"]: s for s in rows}
    if len(rows) != 9895 or len(indexed) != len(rows):
        raise ValueError("Expected 9895 unique official evaluation questions")
    return indexed


def validate(rows, expected):
    if rows.keys() != expected.keys():
        raise ValueError("Incomplete or unexpected evaluation question IDs")
    for qid, row in rows.items():
        sample = expected[qid]
        gold = sample["answer"]
        gold = gold if isinstance(gold, list) else [gold]
        if (row["question"] != sample["question"] or row["split"] != sample["split"]
                or row["target"] != gold or not isinstance(row["correct"], bool)):
            raise ValueError(f"Prediction/input mismatch: {qid}")


def preflight(expected, b1):
    from safetensors import safe_open
    for variant in VARIANTS:
        adapter = HNG / f"results/runs/20261007_accumfix_{variant}_full/listed/adapter"
        with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
            count = len(list(weights.keys()))
            if count == 0:
                raise ValueError(f"Empty adapter: {adapter}")
        print(f"[preflight] {variant}: {count} weight tensors", flush=True)
    for filename in ("predictions_correct.jsonl", "predictions_closed_set.jsonl"):
        validate(read_rows(b1 / "b1" / filename), expected)
    print("[preflight] B1 predictions match all 9895 prompts, targets and IDs", flush=True)


def comparison(left, right, ids, rng):
    a = np.array([left[q]["correct"] for q in ids], dtype=int)
    b = np.array([right[q]["correct"] for q in ids], dtype=int)
    gains = int(((a == 0) & (b == 1)).sum())
    regressions = int(((a == 1) & (b == 0)).sum())
    n = len(ids)
    # Equivalent to paired resampling of the per-question differences in {-1,0,1}.
    draws = rng.multinomial(n, [regressions / n, (n - gains - regressions) / n, gains / n], size=10000)
    interval = np.quantile((draws[:, 2] - draws[:, 0]) / n, [0.025, 0.975]).tolist()
    return {"count": n, "left_em": float(a.mean()), "right_em": float(b.mean()),
            "delta_right_minus_left": float((b - a).mean()), "gains": gains,
            "regressions": regressions, "paired_question_bootstrap_ci95": interval,
            "left_wrong_in_list": sum(not left[q]["correct"] and left[q]["in_list"] for q in ids),
            "right_wrong_in_list": sum(not right[q]["correct"] and right[q]["in_list"] for q in ids),
            "left_out_of_list": sum(not left[q]["in_list"] for q in ids),
            "right_out_of_list": sum(not right[q]["in_list"] for q in ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b1-run", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    expected = samples()
    if args.preflight:
        preflight(expected, args.b1_run)
        return
    report = {"primary": "test generate calibrated minus random_k",
              "baseline_caveat": "B1 predates accumulation normalization repair; historical reference only, not a matched post-fix control.",
              "uncertainty_caveat": "Question bootstrap describes this fixed run under question resampling; it does not measure training-seed variation or graph dependence.",
              "results": {}}
    rng = np.random.default_rng(20261011)
    for mode, filename in (("generate", "predictions_correct.jsonl"), ("closed_set", "predictions_closed_set.jsonl")):
        rows = {v: read_rows(args.output_dir / v / "listed" / filename) for v in VARIANTS}
        rows["b1"] = read_rows(args.b1_run / "b1" / filename)
        for values in rows.values():
            validate(values, expected)
        report["results"][mode] = {}
        for left, right in (("random_k", "calibrated"), ("b1", "random_k"), ("b1", "calibrated")):
            result = {}
            for split in ("test", "validation", "all"):
                ids = [q for q, s in expected.items() if split == "all" or s["split"] == split]
                result[split] = comparison(rows[left], rows[right], ids, rng)
            report["results"][mode][f"{right}_vs_{left}"] = result
    text = json.dumps(report, indent=2)
    (args.output_dir / "paired_comparison.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
