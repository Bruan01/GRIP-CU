#!/usr/bin/env python3
"""Prepare and run the matched NELL/FB Top-9 Hard controls in one tmux pane."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_ccfb_sprint as sprint
from hard_negative_grip.shared_pool_samplers import (
    freeze_shared_pool_samplers,
    verify_listed_manifest_wiring,
)

HNG, CODE, hpo = sprint.HNG, sprint.CODE, sprint.hpo
RUNS = HNG / "results/runs"
ROOT = RUNS / "ccfb_top9_seed2026"
NELL = RUNS / "paper_v2_nell23k"
NELL_HPO = NELL / "hpo_budget60_seed2026"
FB = RUNS / "fb15k237_three_configs_seed2026"
GIB = 1024**3


def argument(command, flag):
    return command[command.index(flag) + 1]


def manifest_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["question_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate QA in {path}")
    return by_id


def prepare():
    plan = {}
    for dataset, primary, original, count, test_count, eval_file in (
        ("nell23k", NELL, hpo.read_json(NELL_HPO / "trial_001/trial.json")["command"],
         3030, 4944, Path(hpo.read_json(NELL_HPO / "selected_test/test_config.json")["eval_file"])),
        ("fb15k237", FB, hpo.read_json(FB / "experiment_plan.json")["training_commands"]["soft_mix_1"],
         3021, 10000, HNG / "data/fb15k237/recurrent_relation_prediction_full.json"),
    ):
        root = ROOT / dataset
        source = primary / "confusion"
        scores = source / "candidate_scores.jsonl"
        metadata_path = source / "metadata.json"
        policy = hpo.read_json(primary / "sampler_seed2026/policy.json")
        metadata = hpo.read_json(metadata_path)
        task = Path(argument(original, "--input_file"))
        if (metadata["filter_splits"] != ["train"] or metadata["split"] != "train" or
                metadata["number_of_qa"] != count or Path(metadata["dataset"]).resolve() != task.resolve() or
                policy["source_scores_sha256"] != hpo.sha256(scores)):
            raise ValueError(f"{dataset}: teacher dump differs from the Soft-Mix source")
        if any(argument(original, flag) != value for flag, value in {
            "--stage": "listed", "--seed": "2026", "--involve_qa_epochs": "3",
            "--lambda_candidate": "1", "--eval_splits": "validation",
        }.items()):
            raise ValueError(f"{dataset}: unexpected historical training budget")
        if dataset == "nell23k":
            test_config = hpo.read_json(NELL_HPO / "selected_test/test_config.json")
            if hpo.sha256(eval_file) != test_config["eval_sha256"]:
                raise ValueError("NELL test inputs changed")
        else:
            sprint.verify_data({
                "dataset": dataset, "data": str(eval_file.parent), "raw": argument(original, "--raw_dir"),
                "evaluation": str(eval_file), "spec": {"validation_count": 10000, "test_count": 10000},
                "initial": True,
            })

        manifest = root / "sampler/top_k_hard.jsonl"
        if not manifest.exists():
            freeze_shared_pool_samplers(scores_path=scores, metadata_path=metadata_path,
                                        output_dir=manifest.parent, k_fixed=9, seed=2026,
                                        lambda_0=1.0, variants=("top_k_hard",))
        top = manifest_rows(manifest)
        soft = manifest_rows(Path(argument(original, "--score_hard_manifest")))
        if top.keys() != soft.keys() or len(top) != count:
            raise ValueError(f"{dataset}: the eligible QA sets differ")
        for qid, row in top.items():
            old = soft[qid]
            if (row["positive_relation"] != old["positive_relation"] or
                    row["n_valid_negatives"] != old["n_valid_negatives"] or
                    any(len(item["negative_relations"]) != 9 or item["lambda_candidate"] != 1
                        for item in (row, old))):
                raise ValueError(f"{dataset}: candidate budget drift at {qid}")
        wiring = verify_listed_manifest_wiring(
            task_path=task, manifest_paths={"top_k_hard": manifest},
            raw_dir=Path(argument(original, "--raw_dir")), expected_listed=count,
            relation_text_file=Path(argument(original, "--relation_text_file")) if
            "--relation_text_file" in original else None,
        )
        wiring.pop("question_ids")
        train = sprint.option(sprint.option(original, "--output_dir", root), "--score_hard_manifest", manifest)
        test = [
            sys.executable, str(HNG / "scripts/eval_listed_decode.py"),
            "--adapter_dir", str(root / "listed/adapter"), "--output_dir", str(root / "test"),
            "--eval_file", str(eval_file), "--splits", "test", "--model_name", argument(original, "--model_name"),
            "--model_cache_dir", argument(original, "--model_cache_dir"),
            "--decode", "generate", "--gen_max_length", argument(original, "--gen_max_length"),
            "--progress_every", "100",
        ]
        s1 = Path(argument(original, "--s1_adapter"))
        inputs = [task, eval_file, Path(argument(original, "--eval_file")), manifest,
                  s1 / "adapter_model.safetensors", s1 / "s1_optimizer.pt", s1 / "adapter_config.json"]
        plan[dataset] = {
            "train": train, "test": test, "test_count": test_count, "wiring": wiring,
            "inputs": {str(p): hpo.sha256(p) for p in dict.fromkeys(inputs)},
            "teacher_scores_sha256": policy["source_scores_sha256"],
            "soft_mix_train": original,
            "soft_mix_overlap_mean": sum(len(set(top[q]["negative_relations"]) &
                                             set(soft[q]["negative_relations"])) for q in top) / count,
        }
        print(dataset, wiring["n_qa"], wiring["n_listed"], flush=True)
    hpo.freeze_text(ROOT / "plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def run():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Start inside the dedicated tmux session.")
    with (RUNS / "ccfb_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sprint.check_idle_gpu()
        status = ROOT / "run_status.json"
        plan = hpo.read_json(ROOT / "plan.json")
        try:
            execute(plan, status, lock)
        except (Exception, KeyboardInterrupt) as exc:
            current = hpo.read_json(status) if status.is_file() else {}
            hpo.write_json(status, {**current, "status": "INTERRUPTED", "error": str(exc)})
            raise


def execute(plan, status, lock):
    for dataset, minimum_gib in (("nell23k", 20), ("fb15k237", 22)):
        job = plan[dataset]
        root = ROOT / dataset
        for path, expected in job["inputs"].items():
            if hpo.sha256(path) != expected:
                raise ValueError(f"Input changed: {path}")
        for stage in ("train", "test"):
            receipt = root / f"{stage}_complete.json"
            if receipt.is_file():
                if any(hpo.sha256(path) != sha for path, sha in hpo.read_json(receipt).items()):
                    raise ValueError(f"Completed output changed: {receipt}")
                continue
            if shutil.disk_usage(root).free < minimum_gib * GIB:
                raise RuntimeError(f"{dataset} needs {minimum_gib} GiB free before {stage}")
            hpo.write_json(status, {"status": "RUNNING", "dataset": dataset, "stage": stage})
            print(f"Starting {dataset} {stage}", flush=True)
            command = list(job[stage])
            adapter = root / "listed/adapter"
            if stage == "train" and all((adapter / p).is_file() for p in
                                         ("adapter_model.safetensors", "adapter_config.json")):
                command.append("--skip_train")
            with (root / f"{stage}.log").open("a") as log:
                subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                               check=True, pass_fds=(lock.fileno(),))
            if stage == "train":
                summary = adapter.parent / "summary.json"
                sprint.validation(hpo.read_json(summary), {
                    "dataset": dataset, "spec": {"validation_count": 4951 if dataset == "nell23k" else 10000},
                })
                outputs = [summary, adapter / "adapter_model.safetensors"]
            else:
                summary = root / "test/summary.json"
                predictions = root / "test/predictions_correct.jsonl"
                result = hpo.read_json(summary)["test"]
                if result["count"] != job["test_count"]:
                    raise ValueError(f"Incomplete test split for {dataset}")
                if dataset == "nell23k":
                    result.update(sprint.nell_harness(
                        [json.loads(row) for row in predictions.read_text().splitlines()],
                        split="test", expected=job["test_count"],
                    ))
                hpo.write_json(root / "test_result.json", result)
                outputs = [summary, predictions, root / "test_result.json"]
            hpo.write_json(receipt, {str(path): hpo.sha256(path) for path in outputs})
    hpo.write_json(status, {"status": "COMPLETE", "datasets": list(plan)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run()
