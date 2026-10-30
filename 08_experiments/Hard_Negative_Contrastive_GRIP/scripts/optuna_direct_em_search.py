#!/usr/bin/env python3
"""Optuna multi-fidelity search for free-generation EM in SC-CL runs.

The objective is standard GRIP free-generation EM on a fixed validation screen.
Candidate-reranking EM is deliberately not used as the optimization target.
Training runs use the existing staged runner so checkpoints, environment,
commands, and manifests remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HNG = Path(__file__).resolve().parents[1]
REPO = HNG.parents[1]
BASELINE = REPO / "13_base_method" / "grip-exp"
RUNNER = HNG / "configs" / "run_error_memory_wsl.sh"
EVALUATOR = HNG / "scripts" / "evaluate_adapter.py"
PYTHON_DEFAULT = Path("/home/kieran/miniconda3/envs/guardenv/bin/python")
MODEL_CACHE_DEFAULT = Path("/mnt/e/GRIP-CU/model_cache")
SEED = 2026
PAPER_EM = {"nell23k": 0.8774, "fb15k237_2": 0.8875}
RUNG_STEPS = (1, 2, 4, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def dataset_paths(dataset: str) -> dict[str, Path | str | int]:
    data = REPO / "04_benchmark" / "data" / dataset
    if dataset == "nell23k":
        return {
            "paper_name": "NELL23K",
            "data_root": data,
            "input": data / "tasks" / "grip_nell23k_tasks.json",
            "valid": data / "processed" / "processed_val.json",
            "test": data / "processed" / "processed_test.json",
            "raw": data / "raw",
            "adapter": BASELINE / "artifacts" / "nell23k_paper_7b" / "lora_adapter",
            "sampler": HNG
            / "results"
            / "hpo"
            / "fb_nell_direct_em_20261027"
            / "nell23k"
            / "k9_from_k15",
            "expected_listed": 3253,
        }
    return {
        "paper_name": "FB15K237-2",
        "data_root": data,
        "input": data / "tasks" / "grip_fb15k237_2_tasks.json",
        "valid": data / "processed" / "processed_val.json",
        "test": data / "processed" / "processed_test.json",
        "raw": data / "raw",
        "adapter": None,
        "sampler": None,
        "expected_listed": 3109,
    }


def prepare_validation_screen(source: Path, destination: Path, n: int, seed: int) -> list[int]:
    if destination.exists():
        manifest_path = destination.with_suffix(".manifest.json")
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))["selected_indices"]
    record = json.loads(source.read_text(encoding="utf-8"))
    questions = record.get("questions")
    answers = record.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list) or len(questions) != len(answers):
        raise ValueError(f"Unsupported validation graph record schema: {source}")
    count = min(n, len(questions))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(questions)), count))
    screen = dict(record)
    screen["questions"] = [questions[i] for i in indices]
    screen["answers"] = [answers[i] for i in indices]
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, screen)
    write_json(
        destination.with_suffix(".manifest.json"),
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "seed": seed,
            "count": count,
            "selected_indices": indices,
            "selection": "uniform random sample without replacement; fixed across all trials",
        },
    )
    return indices


def validate_assets(
    dataset: str,
    paths: dict[str, Path | str | int],
    adapter: Path,
    sampler: Path,
    python: Path,
    model_cache: Path,
) -> None:
    required_files = [
        Path(paths["input"]),
        Path(paths["valid"]),
        Path(paths["test"]),
        Path(paths["raw"]) / "train.txt",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        sampler / "counterfactual_bank.jsonl",
        sampler / "metadata.json",
        sampler / "wiring.json",
        model_cache / "Qwen--Qwen2.5-7B-Instruct" / "config.json",
    ]
    if dataset == "fb15k237_2":
        required_files.append(Path(paths["raw"]) / "rel2text.json")
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "HPO prerequisites are incomplete; do not substitute historical candidate scores. "
            "Missing: " + ", ".join(missing)
        )
    if not python.is_file():
        raise FileNotFoundError(f"Required guardenv Python is unavailable: {python}")
    shard = model_cache / "Qwen--Qwen2.5-7B-Instruct" / "model-00004-of-00004.safetensors"
    if not shard.is_file():
        raise FileNotFoundError(f"Incomplete Qwen2.5-7B model cache: {shard}")
    wiring = json.loads((sampler / "wiring.json").read_text(encoding="utf-8"))
    if wiring.get("train_only") is not True:
        raise ValueError(f"Candidate bank is not certified train-only: {sampler / 'wiring.json'}")
    metadata = json.loads((sampler / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("requested_k", -1)) != 9:
        raise ValueError(f"This HPO pass is fixed at K=9; bank metadata says {metadata.get('requested_k')}")


def run_logged(command: list[str], env: dict[str, str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def train_rung(
    *,
    python: Path,
    dataset: str,
    paths: dict[str, Path | str | int],
    adapter: Path,
    sampler: Path,
    model_cache: Path,
    trial_dir: Path,
    trial_number: int,
    rung_step: int,
    fresh: bool,
    resume_checkpoint: Path | None,
    params: dict[str, float],
) -> None:
    run_dir = trial_dir / "train_run"
    # The loaded Stage-1 adapter determines the actual LoRA shape. In
    # particular, FB's matched paper adapter uses rank 24 rather than NELL's 4.
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    for key, value in {
        "PYTHON": str(python),
        "PYTHONPATH": os.pathsep.join(
            [str(BASELINE), str(HNG / "src"), str(HNG / "scripts"), env.get("PYTHONPATH", "")]
        ),
        "GRIP_BASELINE_DIR": str(BASELINE),
        "STEP": "train",
        "ALLOW_TRAIN": "1",
        "SKIP_TMUX": "1",
        "NO_RESUME": "1" if fresh else "0",
        "DEFER_EVAL": "1",
        "DATASET": dataset,
        "NEGATIVE_SOURCE": "structure_counterfactual",
        "SEED": str(SEED),
        "LISTED_NEGATIVE_K": "9",
        "LAMBDA_CANDIDATE": str(params["lambda_candidate"]),
        "CONTRASTIVE_TEMPERATURE": str(params["temperature"]),
        "TRAIN_LEARNING_RATE": str(params["learning_rate"]),
        "TRAIN_WEIGHT_DECAY": str(params["weight_decay"]),
        "TRAIN_MAX_GRAD_NORM": "1.0",
        "INVOLVE_QA_EPOCHS": str(paths.get("epochs", 10)),
        "TRAIN_LORA_R": str(adapter_config["r"]),
        "TRAIN_LORA_ALPHA": str(adapter_config["lora_alpha"]),
        "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
        "GRADIENT_ACCUMULATION_STEPS": "512",
        "SAVE_STEPS": str(paths.get("save_steps", 1)),
        "SAVE_TOTAL_LIMIT": "2",
        "STOP_AFTER_STEP": str(rung_step),
        "TORCH_DTYPE": "bfloat16",
        "MODEL_CACHE_DIR": str(model_cache),
        "INPUT_FILE": str(paths["input"]),
        "EVAL_FILE": str(paths["valid"]),
        "RAW_DIR": str(paths["raw"]),
        "S1_ADAPTER": str(adapter),
        "B1_ADAPTER": str(paths.get("b1_adapter", adapter)),
        "SAMPLER_DIR": str(sampler),
        "MANIFEST": str(sampler / "counterfactual_bank.jsonl"),
        "COUNTERFACTUAL_METADATA": str(sampler / "metadata.json"),
        "EXPECTED_LISTED": str(paths["expected_listed"]),
        "ARTIFACT_ROOT": str(trial_dir / "training_assets"),
        "RUN_ID": f"{dataset}_direct_em_optuna_t{trial_number:03d}",
        "RUN_DIR": str(run_dir),
        "RUN_VARIANT": f"optuna_rung_{rung_step}",
    }.items():
        env[key] = value
    if dataset in {"fb15k237_2", "codexm"}:
        env["RELATION_TEXT_FILE"] = str(Path(paths["raw"]) / "rel2text.json")
    if paths.get("entity_text_file"):
        env["ENTITY_TEXT_FILE"] = str(paths["entity_text_file"])
    if paths.get("relation_eligibility"):
        env["RELATION_ELIGIBILITY"] = str(paths["relation_eligibility"])
    if resume_checkpoint is None:
        env.pop("RESUME_FROM_CHECKPOINT", None)
    else:
        env["RESUME_FROM_CHECKPOINT"] = str(resume_checkpoint)
    run_logged(["bash", str(RUNNER)], env, trial_dir / f"train_rung_{rung_step}.log", HNG)


def evaluate(
    *,
    python: Path,
    adapter: Path,
    eval_file: Path,
    output_dir: Path,
    model_cache: Path,
    label: str,
    progress_every: int,
) -> float:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("adapter") == str(adapter) and summary.get("eval_file") == str(eval_file):
            return float(summary["all"]["em"])
    command = [
        str(python),
        str(EVALUATOR),
        "--adapter",
        str(adapter),
        "--eval_file",
        str(eval_file),
        "--output_dir",
        str(output_dir),
        "--model_name",
        "qwen-7b",
        "--model_cache_dir",
        str(model_cache),
        "--dtype",
        "bfloat16",
        "--gen_max_length",
        "32",
        "--progress_every",
        str(progress_every),
        "--label",
        label,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BASELINE), str(HNG / "src"), str(HNG / "scripts"), env.get("PYTHONPATH", "")]
    )
    env["GRIP_BASELINE_DIR"] = str(BASELINE)
    env["GRIP_TORCH_DTYPE"] = "bfloat16"
    run_logged(command, env, output_dir / "evaluation.log", HNG)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["all"]["em"])


def write_study_exports(study: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs"))
    frame.to_csv(output_dir / "trials.csv", index=False)
    payload = []
    for trial in study.trials:
        payload.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "intermediate_values": trial.intermediate_values,
                "user_attrs": trial.user_attrs,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
        )
    write_json(output_dir / "trials.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nell23k", "fb15k237_2"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--model-cache-dir", type=Path, default=MODEL_CACHE_DEFAULT)
    parser.add_argument("--s1-adapter", type=Path, default=None,
                        help="For FB, the complete matched B1 adapter is required.")
    parser.add_argument("--sampler-dir", type=Path, default=None,
                        help="For FB, the current train-only SC-CL K=9 bank is required.")
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--validation-screen-size", type=int, default=300)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--skip-final-test", action="store_true",
                        help="Only for a validation-only search; never use test metrics for trial selection.")
    args = parser.parse_args()

    try:
        import optuna
        import torch
    except ImportError as error:
        raise SystemExit(
            "Optuna and PyTorch must be available in the configured guardenv. "
            "Install Optuna there with: guardenv/bin/python -m pip install optuna"
        ) from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit(
            "guardenv cannot see a CUDA device. Run this command in the approved WSL GPU context."
        )

    paths = dataset_paths(args.dataset)
    adapter = args.s1_adapter or paths["adapter"]
    sampler = args.sampler_dir or paths["sampler"]
    if adapter is None or sampler is None:
        raise SystemExit(
            "FB15K237-2 HPO requires a current matched B1 adapter and train-only SC-CL bank. "
            "The historical InfoNCE-i0 candidate scores are not interchangeable."
        )
    adapter, sampler = Path(adapter), Path(sampler)
    model_cache = args.model_cache_dir
    validate_assets(args.dataset, paths, adapter, sampler, args.python, model_cache)

    output_root = args.output_root.resolve() / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    screen = output_root / f"valid_screen_{args.validation_screen_size}.json"
    indices = prepare_validation_screen(
        Path(paths["valid"]), screen, args.validation_screen_size, SEED
    )
    study_name = args.study_name or f"{args.dataset}_sccl_direct_em_seed{SEED}"
    db_path = output_root / f"{study_name}.sqlite3"
    storage = f"sqlite:///{db_path.as_posix()}"
    sampler_optuna = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3, multivariate=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=RUNG_STEPS[0], reduction_factor=2
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler_optuna,
        pruner=pruner,
        load_if_exists=True,
    )
    contract = {
        "research_question": "Can Stage-2 SC-CL hyperparameters improve standard free-generation EM?",
        "objective": "validation free-generation exact match (EM), maximized",
        "candidate_reranking_em_is_objective": False,
        "dataset": args.dataset,
        "study_name": study_name,
        "paper_em_reference": PAPER_EM[args.dataset],
        "seed": SEED,
        "negative_source": "structure_counterfactual",
        "listed_negative_k": 9,
        "training_input": str(paths["input"]),
        "validation_file": str(paths["valid"]),
        "validation_screen": str(screen),
        "validation_screen_sha256": sha256_file(screen),
        "validation_screen_indices": indices,
        "test_file": str(paths["test"]),
        "test_used_during_search": False,
        "baseline_adapter": str(adapter),
        "train_only_candidate_bank": str(sampler),
        "candidate_bank_sha256": sha256_file(sampler / "counterfactual_bank.jsonl"),
        "model_cache": str(model_cache),
        "python": str(args.python),
        "trial_parameters": {
            "lambda_candidate": "log-uniform [0.05, 2.0]",
            "temperature": "log-uniform [0.5, 2.0]",
            "learning_rate": "log-uniform [0.0001, 0.001]",
            "weight_decay": "uniform [0.0, 0.001]",
        },
        "multi_fidelity_optimizer_steps": list(RUNG_STEPS),
        "n_trials_requested": args.n_trials,
        "full_training_budget_epochs": 10,
        "promotion_policy": "Optuna SuccessiveHalving; promote by validation EM",
    }
    write_json(output_root / "run_contract.json", contract)
    write_json(
        output_root / "environment_snapshot.json",
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "optuna": optuna.__version__,
            "python": sys.version,
        },
    )

    baseline_path = output_root / "baseline_validation"
    baseline_em = None
    summary_path = baseline_path / "summary.json"
    predictions_path = baseline_path / "predictions.jsonl"
    if summary_path.is_file() and predictions_path.is_file():
        try:
            cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cached_metric = cached_summary.get("all", cached_summary.get("test", {}))
            cached_adapter = Path(cached_summary.get("adapter", "")).resolve()
            cached_eval_file = Path(cached_summary.get("eval_file", "")).resolve()
            prediction_count = sum(1 for _ in predictions_path.open(encoding="utf-8"))
            cache_matches = (
                cached_summary.get("label") == "b1_free_generation_validation_screen"
                and int(cached_metric.get("count", -1)) == len(indices)
                and prediction_count == len(indices)
                and cached_adapter == adapter.resolve()
                and cached_eval_file == screen.resolve()
                and "em" in cached_metric
            )
            if cache_matches:
                baseline_em = float(cached_metric["em"])
                print(
                    f"[hpo] reuse verified B1 validation cache: EM={baseline_em:.4f}, "
                    f"n={prediction_count}",
                    flush=True,
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            baseline_em = None
    if baseline_em is None:
        baseline_em = evaluate(
            python=args.python,
            adapter=adapter,
            eval_file=screen,
            output_dir=baseline_path,
            model_cache=model_cache,
            label="b1_free_generation_validation_screen",
            progress_every=50,
        )
    write_json(
        baseline_path / "paper_comparison.json",
        {
            "baseline_validation_em": baseline_em,
            "paper_test_em_reference": PAPER_EM[args.dataset],
            "note": "Different split; this validation value is for within-dataset HPO only.",
        },
    )
    print(
        f"[hpo] {args.dataset}: baseline val-screen EM={baseline_em:.4f}; "
        f"paper test EM reference={PAPER_EM[args.dataset]:.4f}; K=9; screen n={len(indices)}",
        flush=True,
    )

    def objective(trial: Any) -> float:
        params = {
            "lambda_candidate": trial.suggest_float("lambda_candidate", 0.05, 2.0, log=True),
            "temperature": trial.suggest_float("temperature", 0.5, 2.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        }
        trial_dir = output_root / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            trial_dir / "trial_config.json",
            {
                "trial_number": trial.number,
                "dataset": args.dataset,
                "seed": SEED,
                "params": params,
                "fixed": {
                    "K": 9,
                    "negative_source": "structure_counterfactual",
                    "epochs": 10,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 512,
                },
            },
        )
        last_em = float("nan")
        previous = 0
        resume_checkpoint = None
        for rung in RUNG_STEPS:
            train_rung(
                python=args.python,
                dataset=args.dataset,
                paths=paths,
                adapter=adapter,
                sampler=sampler,
                model_cache=model_cache,
                trial_dir=trial_dir,
                trial_number=trial.number,
                rung_step=rung,
                fresh=(previous == 0),
                resume_checkpoint=resume_checkpoint,
                params=params,
            )
            previous = rung
            checkpoints = list((trial_dir / "train_run" / "trainer_listed").glob("checkpoint-*"))
            if not checkpoints:
                raise FileNotFoundError(f"No resumable rung checkpoint under {trial_dir / 'train_run'}")
            resume_checkpoint = max(
                checkpoints,
                key=lambda path: int(path.name.rsplit("-", 1)[1]),
            )
            current_adapter = trial_dir / "train_run" / "listed" / "adapter"
            last_em = evaluate(
                python=args.python,
                adapter=current_adapter,
                eval_file=screen,
                output_dir=trial_dir / f"validation_rung_{rung:03d}",
                model_cache=model_cache,
                label=f"optuna_trial_{trial.number}_step_{rung}",
                progress_every=100,
            )
            trial.set_user_attr("last_train_dir", str(trial_dir / "train_run"))
            trial.set_user_attr("last_adapter", str(current_adapter))
            trial.set_user_attr("baseline_validation_em", baseline_em)
            trial.report(last_em, rung)
            print(
                f"[hpo] trial={trial.number} step={rung} val_em={last_em:.4f} "
                f"delta_vs_b1={last_em - baseline_em:+.4f} params={params}",
                flush=True,
            )
            write_study_exports(study, output_root)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_step", rung)
                raise optuna.TrialPruned(f"pruned at step={rung}, validation_em={last_em:.4f}")
        trial.set_user_attr("final_search_validation_em", last_em)
        return last_em

    remaining_trials = max(0, args.n_trials - len(study.trials))
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    write_study_exports(study, output_root)
    if not study.best_trials:
        raise RuntimeError("Optuna completed without a promotable trial.")
    best = study.best_trial
    trial_dir = output_root / "trials" / f"trial_{best.number:04d}"
    run_dir = trial_dir / "train_run"
    adapter_dir = run_dir / "listed" / "adapter"
    write_json(
        output_root / "selected_candidate.json",
        {
            "trial_number": best.number,
            f"validation_em_at_{RUNG_STEPS[-1]}_steps": best.value,
            "baseline_validation_em": baseline_em,
            "delta_vs_baseline": float(best.value) - baseline_em,
            "params": best.params,
            "trial_dir": str(trial_dir),
            "adapter_dir": str(adapter_dir),
            "promotion": "continue this checkpoint to the full 10-epoch training budget before test evaluation",
        },
    )

    if float(best.value) <= baseline_em:
        write_json(
            output_root / "promotion_decision.json",
            {
                "decision": "no_go",
                "reason": f"best {RUNG_STEPS[-1]}-step validation free-generation EM did not exceed the matched B1 validation screen",
                "best_trial": best.number,
                "best_validation_em": best.value,
                "b1_validation_em": baseline_em,
                "full_test_evaluated": False,
            },
        )
        print(
            f"[hpo] no-go: best validation EM={best.value:.4f} did not exceed "
            f"B1={baseline_em:.4f}; skipped full training and test evaluation",
            flush=True,
        )
        print(f"[hpo] best trial={best.number} value={best.value:.4f} params={best.params}", flush=True)
        return

    if not args.skip_final_test:
        final_dir = output_root / f"winner_trial_{best.number:04d}"
        # Continue the exact winning trial's optimizer/scheduler checkpoint to the
        # original full budget. This does not restart training from a partial model.
        final_env = os.environ.copy()
        final_env.update(
            {
                "PYTHON": str(args.python),
                "PYTHONPATH": os.pathsep.join(
                    [str(BASELINE), str(HNG / "src"), str(HNG / "scripts"), final_env.get("PYTHONPATH", "")]
                ),
                "GRIP_BASELINE_DIR": str(BASELINE),
                "STEP": "train",
                "ALLOW_TRAIN": "1",
                "SKIP_TMUX": "1",
                "NO_RESUME": "0",
                "DEFER_EVAL": "1",
                "DATASET": args.dataset,
                "NEGATIVE_SOURCE": "structure_counterfactual",
                "SEED": str(SEED),
                "LISTED_NEGATIVE_K": "9",
                "LAMBDA_CANDIDATE": str(best.params["lambda_candidate"]),
                "CONTRASTIVE_TEMPERATURE": str(best.params["temperature"]),
                "TRAIN_LEARNING_RATE": str(best.params["learning_rate"]),
                "TRAIN_WEIGHT_DECAY": str(best.params["weight_decay"]),
                "TRAIN_MAX_GRAD_NORM": "1.0",
                "INVOLVE_QA_EPOCHS": "10",
                "TRAIN_LORA_R": "4",
                "TRAIN_LORA_ALPHA": "8",
                "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
                "GRADIENT_ACCUMULATION_STEPS": "512",
                "SAVE_STEPS": "10",
                "SAVE_TOTAL_LIMIT": "2",
                "STOP_AFTER_STEP": "0",
                "TORCH_DTYPE": "bfloat16",
                "MODEL_CACHE_DIR": str(model_cache),
                "INPUT_FILE": str(paths["input"]),
                "EVAL_FILE": str(paths["valid"]),
                "RAW_DIR": str(paths["raw"]),
                "S1_ADAPTER": str(adapter),
                "B1_ADAPTER": str(adapter),
                "SAMPLER_DIR": str(sampler),
                "MANIFEST": str(sampler / "counterfactual_bank.jsonl"),
                "COUNTERFACTUAL_METADATA": str(sampler / "metadata.json"),
                "EXPECTED_LISTED": str(paths["expected_listed"]),
                "ARTIFACT_ROOT": str(trial_dir / "training_assets"),
                "RUN_ID": f"{args.dataset}_direct_em_optuna_t{best.number:03d}",
                "RUN_DIR": str(run_dir),
                "RUN_VARIANT": "optuna_promoted_full_budget",
            }
        )
        final_env.pop("RESUME_FROM_CHECKPOINT", None)
        if not (run_dir / "run_config.json").is_file():
            raise FileNotFoundError(f"Promoted trial is missing its training run: {run_dir}")
        checkpoints = list((run_dir / "trainer_listed").glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"Promoted trial has no resumable checkpoint: {run_dir}")
        latest_checkpoint = max(
            checkpoints,
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        final_env["RESUME_FROM_CHECKPOINT"] = str(latest_checkpoint)
        run_logged(
            ["bash", str(RUNNER)],
            final_env,
            final_dir / "continue_to_full_budget.log",
            HNG,
        )
        adapter_dir = run_dir / "listed" / "adapter"
        final_screen_em = evaluate(
            python=args.python,
            adapter=adapter_dir,
            eval_file=screen,
            output_dir=final_dir / "validation_screen",
            model_cache=model_cache,
            label="promoted_full_budget_validation_screen",
            progress_every=50,
        )
        test_em = evaluate(
            python=args.python,
            adapter=adapter_dir,
            eval_file=Path(paths["test"]),
            output_dir=final_dir / "test",
            model_cache=model_cache,
            label="promoted_full_budget_test_free_generation",
            progress_every=100,
        )
        write_json(
            final_dir / "final_result.json",
            {
                "dataset": args.dataset,
                "trial_number": best.number,
                "params": best.params,
                "baseline_validation_screen_em": baseline_em,
                "winner_validation_screen_em": final_screen_em,
                "winner_test_free_generation_em": test_em,
                "paper_test_em_reference": PAPER_EM[args.dataset],
                "delta_vs_paper_test_em": test_em - PAPER_EM[args.dataset],
                "test_split_used_for_hpo": False,
                "test_prediction_file": str(final_dir / "test" / "predictions.jsonl"),
                "adapter_dir": str(adapter_dir),
                "train_run_dir": str(run_dir),
                "seconds_utc": int(time.time()),
            },
        )
        write_json(
            output_root / "promotion_decision.json",
            {
                "decision": "promoted",
                "reason": f"best {RUNG_STEPS[-1]}-step validation free-generation EM exceeded the matched B1 screen",
                "best_trial": best.number,
                "best_validation_em": best.value,
                "b1_validation_em": baseline_em,
                "full_test_evaluated": True,
                "full_test_em": test_em,
                "paper_test_em_reference": PAPER_EM[args.dataset],
                "exceeds_paper_test_reference": test_em > PAPER_EM[args.dataset],
            },
        )
    print(f"[hpo] best trial={best.number} value={best.value:.4f} params={best.params}", flush=True)


if __name__ == "__main__":
    main()
