#!/usr/bin/env python3
"""Serial, restartable Optuna search over S2; never decode or select on test."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

HNG = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = HNG / "scripts/train_listed_contrastive.py"
EXPECTED_VALIDATION_COUNT = 4951


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze_text(path, text):
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing frozen artifact differs: {path}; use a new search_dir")
    if not path.exists():
        atomic_text(path, text)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input_file", "eval_file", "s1_adapter", "calibrated_manifest",
                 "raw_dir", "model_cache_dir", "search_dir", "pipeline_root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n_trials", type=int, default=8, help="Total budget, not extra trials per restart")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true", help="Print plan without models or artifact writes")
    args = parser.parse_args(argv)
    if args.n_trials < 4:
        parser.error("n_trials must be >= 4 (four predeclared anchor trials)")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def required_files(args):
    return [args.input_file, args.eval_file, args.calibrated_manifest,
            args.raw_dir / "train.txt", args.s1_adapter / "adapter_config.json",
            args.s1_adapter / "adapter_model.safetensors", args.s1_adapter / "s1_optimizer.pt"]


def training_command(args, params, trial_dir):
    # Per-QA manifest weights override --lambda_candidate in the trainer.
    command = [sys.executable, str(TRAIN_SCRIPT),
               "--input_file", str(args.input_file), "--eval_file", str(args.eval_file),
               "--eval_splits", "validation", "--official_drop_last_line",
               "--output_dir", str(trial_dir), "--stage", "listed",
               "--s1_adapter", str(args.s1_adapter), "--model_name", "qwen-7b",
               "--model_cache_dir", str(args.model_cache_dir), "--raw_dir", str(args.raw_dir),
               "--lora_r", "4", "--lora_alpha", "8",
               "--target_modules", "down_proj", "up_proj", "gate_proj",
               "--num_train_epochs", "1", "--involve_qa_epochs", str(params["epochs"]),
               "--qa_loss_mask", "answer_only", "--per_device_train_batch_size", "1",
               "--gradient_accumulation_steps", "512", "--s1_gradient_checkpointing",
               "--learning_rate", "0.001", "--weight_decay", "0.0001", "--adam_epsilon", "0.0001",
               "--max_grad_norm", "1", "--s2_optimizer", "inherit_s1", "--gen_max_length", "1000",
               "--save_steps", "10", "--save_total_limit", "2", "--seed", str(args.seed),
               "--lambda_candidate", "1", "--temperature", "1", "--memory_size", "0",
               "--listed_negative_source", "score_hard",
               "--score_hard_manifest", str(trial_dir / "negatives.jsonl")]
    if (trial_dir / "listed/adapter/adapter_config.json").is_file() and (
        trial_dir / "listed/adapter/adapter_model.safetensors"
    ).is_file():
        command.append("--skip_train")
    return command


def freeze_manifest(source, output, params):
    rows = []
    seen = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["question_id"]
        negatives = row["negative_relations"]
        if qid in seen or not negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"Invalid/duplicate frozen negatives for {qid}")
        if row["positive_relation"] in negatives or row.get("shared_pool") != "train_only_valid_negatives":
            raise ValueError(f"Expected train-only negatives without gold for {qid}")
        if row.get("negative_source") != "shared_valid_pool_calibrated" or row.get("variant", "calibrated") != "calibrated":
            raise ValueError("Source must be the frozen calibrated manifest, not a fixed-lambda copy")
        provenance = list(dict.fromkeys(row["hard_negative_relations"] + row["uniform_negative_relations"]))
        if negatives != provenance:
            raise ValueError(f"Negative provenance mismatch for {qid}")
        seen.add(qid)
        original = float(row["lambda_candidate"])
        if not math.isfinite(original) or not 0 < original <= 1:
            raise ValueError(f"Invalid calibrated weight for {qid}")
        row["hpo_source_lambda"] = original
        row["lambda_candidate"] = params["lambda_scale"] * (
            original if params["method"] == "calibrated" else 1.0
        )
        row["variant"] = "hpo_" + params["method"]
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")
    if not rows:
        raise ValueError("Empty calibrated manifest")
    freeze_text(output, "".join(rows))


def validation_record(path):
    payload = read_json(path)
    record = dict(payload[0] if isinstance(payload, list) else payload)
    samples = [row for row in record["recurrent_questions"] if row.get("split") == "validation"]
    ids = [row["question_id"] for row in samples]
    indices = {int(str(qid).rsplit(":", 1)[-1]) for qid in ids}
    if len(samples) < 2 or len(set(ids)) != len(ids) or indices != set(range(len(samples))):
        raise ValueError("Full validation with unique sequential question IDs is required")
    record["recurrent_questions"] = samples
    return record


def validation_value(path, expected_count):
    summary = read_json(path)
    if "test" in summary:
        raise ValueError("Search summary contains test results")
    validation = summary["validation"]
    if validation["count"] != expected_count or validation["official_harness_count"] != expected_count - 1:
        raise ValueError("Incomplete validation summary")
    value = float(validation["official_harness_em"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid validation metric")
    return value


def experiment_config(args):
    sources = required_files(args) + [TRAIN_SCRIPT, Path(__file__).resolve()]
    sources += sorted((HNG / "src/hard_negative_grip").glob("*.py"))
    import optuna

    return {"version": 1, "seed": args.seed, "n_trials": args.n_trials,
            "optuna_version": optuna.__version__, "model_cache_dir": str(args.model_cache_dir),
            "python": sys.executable, "pipeline_root": str(args.pipeline_root),
            "space": {"method": ["soft_mix", "calibrated"], "lambda_scale": [0.1, 1.0],
                      "lambda_log": True, "epochs": [1, 3, 5, 10]},
            "metric": "validation.official_harness_em", "train_loss_stopping": False,
            "files": {str(path): sha256(path) for path in sources}}


def suggest(trial):
    return {"method": trial.suggest_categorical("method", ["soft_mix", "calibrated"]),
            "lambda_scale": trial.suggest_float("lambda_scale", 0.1, 1.0, log=True),
            "epochs": trial.suggest_categorical("epochs", [1, 3, 5, 10])}


def next_trial(study, seed, budget):
    import optuna

    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1:
        raise ValueError("Multiple RUNNING trials: study was modified by another runner")
    if running:
        study.sampler = optuna.samplers.TPESampler(seed=seed + running[0].number, n_startup_trials=4)
        return optuna.trial.Trial(study, running[0]._trial_id)
    issued = sum(trial.state != optuna.trial.TrialState.WAITING for trial in study.trials)
    if issued >= budget:
        return None
    # A per-trial seed prevents restart from resetting one shared RNG stream.
    study.sampler = optuna.samplers.TPESampler(seed=seed + issued, n_startup_trials=4)
    return study.ask()


def export_results(study, directory):
    records = [{"number": t.number, "state": t.state.name, "params": t.params,
                "validation_em": t.value, "details": t.user_attrs} for t in study.trials]
    write_json(directory / "trials.json", records)
    complete = [row for row in records if row["state"] == "COMPLETE"]
    best = max(complete, key=lambda row: (row["validation_em"], -row["number"])) if complete else None
    write_json(directory / "best.json", {"metric": "validation.official_harness_em", "best": best,
               "best_by_method": {method: max(
                   (row for row in complete if row["params"]["method"] == method),
                   key=lambda row: (row["validation_em"], -row["number"]), default=None)
                   for method in ("soft_mix", "calibrated")}})
    lines = ["# Validation HPO Variants\n\n", "Test is never evaluated by this search.\n\n",
             "| Trial | State | Parameters | Validation EM | Selected |\n",
             "| --- | --- | --- | --- | --- |\n"]
    for row in records:
        value = "" if row["validation_em"] is None else f'{100 * row["validation_em"]:.4f}%'
        lines.append(f'| {row["number"]} | {row["state"]} | {json.dumps(row["params"], sort_keys=True)} '
                     f'| {value} | {"yes" if row == best else ""} |\n')
    atomic_text(directory / "VARIANTS.md", "".join(lines))


def run_search(args, execute=None):
    import optuna

    missing = [str(path) for path in required_files(args) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Finish S1/score/freeze first; missing: " + ", ".join(missing))
    args.search_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_fds = []
        # Share the existing paper-pipeline lock, even across tmux sessions.
        for path in (args.pipeline_root / "pipeline.lock", args.search_dir / "search.lock"):
            handle = stack.enter_context(path.open("a"))
            lock_fds.append(handle.fileno())
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"GPU workflow already running; lock busy: {path}") from exc
        config = experiment_config(args)
        freeze_text(args.search_dir / "search_config.json", json.dumps(config, sort_keys=True, indent=2) + "\n")
        record = validation_record(args.eval_file)
        if len(record["recurrent_questions"]) != EXPECTED_VALIDATION_COUNT:
            raise ValueError(f"NELL23K full validation requires {EXPECTED_VALIDATION_COUNT} questions")
        validation_file = args.search_dir / "validation_only.json"
        freeze_text(validation_file, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        study = optuna.create_study(study_name="nell23k_validation_s2", direction="maximize",
                                    storage=f"sqlite:///{args.search_dir / 'study.sqlite3'}", load_if_exists=True)
        if not study.trials:
            for method in ("soft_mix", "calibrated"):
                for scale in (1.0, 0.5):
                    study.enqueue_trial({"method": method, "lambda_scale": scale, "epochs": 10})
        while (trial := next_trial(study, args.seed, args.n_trials)) is not None:
            params = suggest(trial)
            trial_dir = args.search_dir / f"trial_{trial.number:03d}"
            trial_dir.mkdir(exist_ok=True)
            freeze_manifest(args.calibrated_manifest, trial_dir / "negatives.jsonl", params)
            child_args = argparse.Namespace(**vars(args))
            child_args.eval_file = validation_file
            command = training_command(child_args, params, trial_dir)
            write_json(trial_dir / "trial.json", {"number": trial.number, "params": params, "command": command})
            trial.set_user_attr("directory", str(trial_dir))
            export_results(study, args.search_dir)
            print(f"[hpo] trial={trial.number} params={params}\n{shlex.join(command)}", flush=True)
            try:
                if execute is None:
                    with (trial_dir / "run.log").open("a") as log:
                        # Keep locks held by training even if the search parent is killed.
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(lock_fds))
                else:
                    execute(command, trial_dir)
                value = validation_value(trial_dir / "listed/summary.json", len(record["recurrent_questions"]))
            except Exception as exc:
                trial.set_user_attr("last_error", str(exc))
                errors = list(trial.user_attrs.get("attempt_errors", []))
                errors.append(str(exc))
                trial.set_user_attr("attempt_errors", errors)
                export_results(study, args.search_dir)
                raise  # Keep RUNNING: recover the same trial/checkpoint on restart.
            trial.set_user_attr("last_error", None)
            study.tell(trial, value)
            export_results(study, args.search_dir)
            print(f"[hpo] trial={trial.number} validation_em={value:.6f}", flush=True)
        export_results(study, args.search_dir)
        print(f"[hpo] finished; best configuration: {args.search_dir / 'best.json'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        params = {"method": "soft_mix", "lambda_scale": 1.0, "epochs": 10}
        print(json.dumps({"n_trials_total": args.n_trials, "missing_prerequisites": [
            str(path) for path in required_files(args) if not path.is_file()],
            "example_command": training_command(args, params, args.search_dir / "trial_000"),
            "metric": "validation.official_harness_em", "test_evaluation": False}, indent=2))
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Launch long GPU searches through configs/run_validation_hpo.sh inside tmux")
    run_search(args)


if __name__ == "__main__":
    main()
