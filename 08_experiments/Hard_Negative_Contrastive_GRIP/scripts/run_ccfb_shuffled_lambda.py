#!/usr/bin/env python3
"""Permute the frozen Calibrated 1.0 weights independently within NELL and FB."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

from analyze_ccfb_seed2026 import DATA, analyze, read_predictions
from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
from run_ccfb_top9 import FB, HNG, NELL, NELL_HPO, RUNS, argument, hpo, manifest_rows, run, sprint

ROOT = RUNS / "ccfb_shuffled_lambda_seed2026"
TRAIN_SEED = 2026
SHUFFLE_SEED = 2026
SPECS = {
    "nell23k": (NELL, NELL_HPO / "trial_003", NELL_HPO / "test_comparison/trial_003",
                19808, 3030, 4951, 4944, "official_harness_em"),
    "fb15k237": (FB, FB / "calibrated_1", FB / "test/calibrated_1",
                 19845, 3021, 10000, 10000, "official_em"),
}


def verify_hashes(expected):
    for path, sha in expected.items():
        if hpo.sha256(path) != sha:
            raise ValueError(f"Frozen input or reference changed: {path}")


def permute_weights(source, target, count):
    original = manifest_rows(source)
    eligible = [qid for qid, row in original.items() if not row.get("contrastive_excluded")]
    weights = [original[qid]["lambda_candidate"] for qid in eligible]
    if len(eligible) != count or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"Unexpected eligible weights in {source}")
    if any(len(original[qid]["negative_relations"]) != 9 for qid in eligible):
        raise ValueError(f"Unexpected candidate budget in {source}")
    if any(row["lambda_candidate"] != 0 or row["negative_relations"]
           for row in original.values() if row.get("contrastive_excluded")):
        raise ValueError(f"Nonzero contrastive payload on excluded QA in {source}")

    shuffled = weights.copy()
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    replacement = dict(zip(eligible, shuffled, strict=True))
    text = "".join(
        json.dumps({**row, "lambda_candidate": replacement[qid]} if qid in replacement else row,
                   ensure_ascii=False) + "\n"
        for qid, row in original.items()
    )
    hpo.freeze_text(target, text)
    actual = manifest_rows(target)
    changed = sum(actual[qid]["lambda_candidate"] != original[qid]["lambda_candidate"] for qid in eligible)
    if (list(actual) != list(original) or not changed
            or Counter(actual[qid]["lambda_candidate"] for qid in eligible) != Counter(weights)
            or any({**actual[qid], "lambda_candidate": row["lambda_candidate"]} != row
                   for qid, row in original.items())):
        raise ValueError(f"Shuffled manifest did not preserve the original QA and weight distribution: {target}")
    return {
        "seed": SHUFFLE_SEED, "eligible": count, "excluded": len(original) - count,
        "changed_weights": changed, "min": min(weights), "max": max(weights),
        "mean": statistics.mean(weights), "std": statistics.pstdev(weights),
    }


def historical_commands(dataset, trial, reference, test_count):
    if dataset == "nell23k":
        frozen_trial = hpo.read_json(trial / "trial.json")
        config = hpo.read_json(reference / "test_config.json")
        train = frozen_trial["command"]
        adapter = trial / "listed/adapter"
        eval_file = DATA / dataset / "recurrent_relation_prediction_full.aligned.json"
        if (frozen_trial["params"] != {"method": "calibrated", "lambda_scale": 1.0, "epochs": 3}
                or config["selection"]["number"] != frozen_trial["number"]
                or config["adapter"] != str(adapter) or config["eval_file"] != str(eval_file)
                or config["splits"] != ["test"] or config["test_count"] != test_count
                or not config["official_drop_last_line"] or "--official_drop_last_line" not in train
                or config["model_name"] != argument(train, "--model_name")
                or str(config["gen_max_length"]) != argument(train, "--gen_max_length")):
            raise ValueError("NELL test receipt does not identify the matched Calibrated 1.0 model.")
        verify_hashes({
            eval_file: config["eval_sha256"],
            adapter / "adapter_model.safetensors": config["adapter_sha256"],
            adapter / "adapter_config.json": config["adapter_config_sha256"],
        })
        frozen_search = hpo.read_json(NELL_HPO / "search_config.json")["files"]
        s1 = Path(argument(train, "--s1_adapter"))
        matched_inputs = [Path(argument(train, "--input_file")), Path(train[1]), s1 / "adapter_config.json",
                          s1 / "adapter_model.safetensors", s1 / "s1_optimizer.pt"]
        verify_hashes({p: frozen_search[str(p)] for p in matched_inputs})
        # The evaluator uses the same generate path. The shared runner applies
        # NELL's original 4,943-question harness to the full prediction file.
        test = [
            train[0], str(HNG / "scripts/eval_listed_decode.py"),
            "--adapter_dir", str(adapter), "--output_dir", str(reference), "--eval_file", str(eval_file),
            "--splits", "test", "--model_name", config["model_name"],
            "--model_cache_dir", argument(train, "--model_cache_dir"), "--decode", "generate",
            "--gen_max_length", str(config["gen_max_length"]), "--progress_every", "100",
        ]
    else:
        plan = hpo.read_json(FB / "experiment_plan.json")
        fitted = hpo.read_json(FB / "completed/calibrated_1.json")
        tested = hpo.read_json(FB / "completed/test_calibrated_1.json")
        train, test = fitted["command"], tested["command"]
        if (train != plan["training_commands"]["calibrated_1"]
                or hpo.read_json(FB / "run_status.json")["status"] != "COMPLETE"
                or argument(test, "--adapter_dir") != str(trial / "listed/adapter")
                or argument(test, "--output_dir") != str(reference)
                or argument(test, "--eval_file") != argument(train, "--eval_file")):
            raise ValueError("FB completion receipts do not identify the matched Calibrated 1.0 model.")
        verify_hashes({**fitted["outputs"], **tested["outputs"]})
        verify_hashes({p: plan["sources"][p] for p in (train[1], test[1])})
    return train, test


def prepare():
    plan = {}
    for dataset, (primary, trial, reference, n_qa, n_listed, n_val, n_test, metric) in SPECS.items():
        train, test = historical_commands(dataset, trial, reference, n_test)
        validation = sprint.validation(hpo.read_json(trial / "listed/summary.json"), {
            "dataset": dataset, "spec": {"validation_count": n_val},
        })
        test_summary = hpo.read_json(reference / "summary.json")["test"]
        prediction, _ = read_predictions(reference, n_test)
        if (test_summary["count"] != n_test or argument(test, "--splits") != "test"
                or argument(test, "--decode") != "generate"
                or any(argument(train, flag) != value for flag, value in {
                    "--seed": str(TRAIN_SEED), "--stage": "listed", "--involve_qa_epochs": "3",
                    "--lambda_candidate": "1", "--s2_optimizer": "inherit_s1",
                    "--listed_negative_source": "score_hard", "--eval_splits": "validation",
                }.items())):
            raise ValueError(f"{dataset}: reference training or evaluation protocol differs")
        metadata = hpo.read_json(trial / "listed/adapter/run_metadata.json")
        if (metadata["trainer_global_step"], metadata["trainer_epoch"], metadata["qa_samples"],
                metadata["candidate_forwards"]) != (117, 3.0, n_qa, 30 * n_listed):
            raise ValueError(f"{dataset}: reference did not complete the expected QA/candidate budget")
        task, raw, source, s1, val_file = (Path(argument(train, flag)) for flag in (
            "--input_file", "--raw_dir", "--score_hard_manifest", "--s1_adapter", "--eval_file",
        ))
        rel = Path(argument(train, "--relation_text_file")) if "--relation_text_file" in train else None
        eval_file = Path(argument(test, "--eval_file"))
        sprint.verify_data({
            "dataset": dataset, "data": str(eval_file.parent), "raw": str(raw), "evaluation": str(eval_file),
            "spec": {"validation_count": n_val, "test_count": n_test}, "initial": True,
        })

        job = ROOT / dataset
        manifest = job / "negatives.jsonl"
        weights = permute_weights(source, manifest, n_listed)
        wiring = verify_listed_manifest_wiring(
            task_path=task, manifest_paths={"calibrated": manifest}, raw_dir=raw,
            relation_text_file=rel, expected_listed=n_listed,
        )
        wiring.pop("question_ids")
        wiring.pop("_negatives_by_id", None)
        variant = wiring["variants"]["calibrated"]
        if (wiring["n_qa"], variant["n_lambda"], variant["k_min"], variant["k_max"],
                variant["n_generation_only_excluded"], variant["lambda_min"], variant["lambda_max"]) != (
                n_qa, n_listed, 9, 9, weights["excluded"], weights["min"], weights["max"]):
            raise ValueError(f"{dataset}: weights or candidate lists did not reach the training QA rows")

        inputs = [task, val_file, eval_file, raw / "train.txt", source, manifest,
                  s1 / "adapter_model.safetensors", s1 / "adapter_config.json", s1 / "s1_optimizer.pt",
                  prediction, reference / "summary.json", trial / "listed/summary.json", Path(train[1]), Path(test[1]),
                  HNG / "src/hard_negative_grip/score_hard.py", HNG / "src/hard_negative_grip/listed_training.py",
                  HNG / "src/hard_negative_grip/decode_io.py", Path(__file__).resolve()]
        if rel is not None:
            inputs.append(rel)
        plan[dataset] = {
            "train": sprint.option(sprint.option(train, "--output_dir", job), "--score_hard_manifest", manifest),
            "test": sprint.option(sprint.option(test, "--output_dir", job / "test"),
                                  "--adapter_dir", job / "listed/adapter"),
            "reference_train": train, "reference_test": test, "test_count": n_test,
            "reference_scores": {"validation": validation, "test": test_summary[metric]},
            "weights": weights, "wiring": wiring,
            "inputs": {str(p): hpo.sha256(p) for p in dict.fromkeys(inputs)},
        }
        print(f"{dataset}: {n_qa} QA, {n_listed} contrastive, {weights['changed_weights']} weights changed; "
              f"old lambda mean={weights['mean']:.8f}, sd={weights['std']:.8f}", flush=True)
    hpo.freeze_text(ROOT / "plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def compare():
    if hpo.read_json(ROOT / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Both shuffled-weight controls must complete before comparison.")
    plan = hpo.read_json(ROOT / "plan.json")
    comparisons = {}
    for dataset, (primary, trial, reference, n_qa, n_listed, n_val, n_test, metric) in SPECS.items():
        config = plan[dataset]
        job = ROOT / dataset
        adapter = job / "listed/adapter"
        metadata = hpo.read_json(adapter / "run_metadata.json")
        if (metadata["trainer_global_step"], metadata["trainer_epoch"], metadata["qa_samples"],
                metadata["candidate_forwards"]) != (117, 3.0, n_qa, 30 * n_listed):
            raise ValueError(f"{dataset}: shuffled run did not match the reference training budget")
        reference_prediction = str(reference / "predictions_correct.jsonl")
        verify_hashes({reference_prediction: config["inputs"][reference_prediction]})
        row = analyze(
            dataset, primary, Path(argument(config["test"], "--eval_file")), reference, job / "test", n_test,
            (config["reference_scores"]["test"], hpo.read_json(job / "test_result.json")[metric]),
            (trial / "listed/adapter", adapter), methods=("calibrated_1", "shuffled_lambda"),
        )
        row["validation_em"] = {
            "calibrated_1": config["reference_scores"]["validation"],
            "shuffled_lambda": sprint.validation(hpo.read_json(job / "listed/summary.json"), {
                "dataset": dataset, "spec": {"validation_count": n_val},
            }),
        }
        row["weights"] = config["weights"]
        comparisons[dataset] = row
    output = {"seed": TRAIN_SEED, "shuffle_seed": SHUFFLE_SEED,
              "left": "calibrated_1", "right": "shuffled_lambda", "datasets": comparisons}
    hpo.freeze_text(ROOT / "comparison.json", json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true", help="Freeze both controls and verify weight wiring on CPU.")
    modes.add_argument("--compare", action="store_true", help="Compare the completed predictions without running the GPU.")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        if not args.compare:
            run(ROOT)
        compare()
