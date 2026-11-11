#!/usr/bin/env python3
"""Run WN18RR with the mean weight from the matched Calibrated 1.0 arm."""
import argparse
import json
import math
from pathlib import Path
import statistics

from analyze_ccfb_seed2026 import analyze
from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
from run_ccfb_top9 import HNG, RUNS, argument, hpo, manifest_rows, run, sprint

PRIMARY = RUNS / "wn18rr_three_configs_seed2026"
ROOT = RUNS / "ccfb_wn_fixed_lambda_seed2026"
JOB = ROOT / "wn18rr"


def prepare():
    original = hpo.read_json(PRIMARY / "experiment_plan.json")
    train_receipt = hpo.read_json(PRIMARY / "completed/calibrated_1.json")
    test_receipt = hpo.read_json(PRIMARY / "completed/test_calibrated_1.json")
    train = train_receipt["command"]
    test = test_receipt["command"]
    if train != original["training_commands"]["calibrated_1"] or any(
        argument(train, flag) != expected for flag, expected in {
            "--seed": "2026", "--involve_qa_epochs": "3", "--eval_splits": "validation",
        }.items()
    ):
        raise ValueError("Original WN calibrated training configuration differs.")
    task, evaluation, source, s1, rel = (
        Path(argument(train, flag)) for flag in (
            "--input_file", "--eval_file", "--score_hard_manifest", "--s1_adapter", "--relation_text_file",
        )
    )
    if str(evaluation) != argument(test, "--eval_file") or argument(test, "--splits") != "test":
        raise ValueError("The WN control must use the original test inputs and split.")
    for path, digest in {**train_receipt["outputs"], **test_receipt["outputs"]}.items():
        if hpo.sha256(path) != digest:
            raise ValueError(f"Original WN result changed: {path}")
    sprint.verify_data({
        "dataset": "wn18rr", "data": str(evaluation.parent), "raw": argument(train, "--raw_dir"),
        "evaluation": str(evaluation), "spec": {"validation_count": 3034, "test_count": 3134},
        "initial": True,
    })
    source_rows = manifest_rows(source)
    eligible = [row for row in source_rows.values() if not row.get("contrastive_excluded")]
    excluded = [row for row in source_rows.values() if row.get("contrastive_excluded")]
    weights = [row["lambda_candidate"] for row in eligible]
    if (len(eligible), len(excluded)) != (2697, 79) or any(
        not math.isfinite(w) or w <= 0 for w in weights
    ) or any(row["lambda_candidate"] or row["negative_relations"] for row in excluded):
        raise ValueError("Original WN weights or exclusion coverage differs.")
    mean = sum(weights) / len(weights)
    manifest = JOB / "negatives.jsonl"
    hpo.freeze_text(manifest, "".join(
        json.dumps(row if row.get("contrastive_excluded") else {**row, "lambda_candidate": mean},
                   ensure_ascii=False) + "\n"
        for row in source_rows.values()
    ))
    wiring = verify_listed_manifest_wiring(
        task_path=task, manifest_paths={"calibrated": manifest},
        raw_dir=Path(argument(train, "--raw_dir")), expected_listed=len(eligible),
        relation_text_file=rel,
    )
    wiring.pop("question_ids")
    variant = wiring["variants"]["calibrated"]
    if (variant["lambda_min"], variant["lambda_max"], variant["n_generation_only_excluded"]) != (
            mean, mean, len(excluded)):
        raise ValueError("The constant weights did not reach the eligible training examples.")
    inputs = [task, evaluation, source, manifest, rel, s1 / "s1_optimizer.pt",
              s1 / "adapter_config.json", s1 / "adapter_model.safetensors"]
    plan = {
        "train": sprint.option(sprint.option(train, "--output_dir", JOB), "--score_hard_manifest", manifest),
        "test": sprint.option(sprint.option(test, "--output_dir", JOB / "test"),
                              "--adapter_dir", JOB / "listed/adapter"),
        "test_count": 3134, "fixed_lambda": mean, "original_lambda_std": statistics.pstdev(weights),
        "wiring": wiring, "inputs": {str(p): hpo.sha256(p) for p in inputs},
        "reference_train": train, "reference_test": test,
    }
    hpo.freeze_text(ROOT / "plan.json", json.dumps({"wn18rr": plan}, indent=2) + "\n")
    print("WN", mean, plan["original_lambda_std"], wiring["n_qa"], len(eligible), len(excluded))


def compare():
    plan = hpo.read_json(ROOT / "plan.json")["wn18rr"]
    reference = PRIMARY / "test/calibrated_1"
    rows = analyze(
        "wn18rr", PRIMARY, Path(argument(plan["test"], "--eval_file")), reference, JOB / "test",
        plan["test_count"], (hpo.read_json(reference / "summary.json")["test"]["official_em"],
                             hpo.read_json(JOB / "test_result.json")["official_em"]),
        (PRIMARY / "calibrated_1/listed/adapter", JOB / "listed/adapter"),
        methods=("calibrated_1", "fixed_mean"),
    )
    rows["validation_em"] = {
        label: hpo.read_json(path / "listed/summary.json")["validation"]["official_em"]
        for label, path in (("calibrated_1", PRIMARY / "calibrated_1"), ("fixed_mean", JOB))
    }
    rows["fixed_lambda"] = plan["fixed_lambda"]
    hpo.freeze_text(ROOT / "comparison.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run(ROOT)
        compare()
