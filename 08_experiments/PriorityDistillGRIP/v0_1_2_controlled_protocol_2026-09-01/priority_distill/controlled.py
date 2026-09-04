"""Controlled direct answer-only versus oracle two-stage experiment."""
from __future__ import annotations
import math
import time
from pathlib import Path
import torch
from tqdm import tqdm
from .candidates import build_candidate_pools
from .experiment import (
    evaluate_checkpoint, load_runtime, save_checkpoint, train_rows_once,
)
from .io_utils import environment_snapshot, set_seed, sha256_file, write_json, write_jsonl
from .records import load_splits
from .runtime_data import balance_rows_to_token_budget, tokenize_supervision_rows
from .supervision import build_method_supervision


def _prepare_rows(train_rows, tokenizer, config, seed):
    candidate_seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(train_rows, int(config["candidates"]["distractor_count"]), candidate_seed)
    answer = tokenize_supervision_rows(
        build_method_supervision(train_rows, pools, "answer_only", candidate_seed),
        tokenizer, int(config["data"]["max_length"]),
    )
    oracle = tokenize_supervision_rows(
        build_method_supervision(train_rows, pools, "oracle_priority_equal_token", candidate_seed),
        tokenizer, int(config["data"]["max_length"]),
    )
    reference = tokenize_supervision_rows(
        build_method_supervision(train_rows, pools, "all_paths_equal_token", candidate_seed),
        tokenizer, int(config["data"]["max_length"]),
    )
    reference_tokens = sum(int(row["input_token_count"]) for row in reference)
    stage1_rows, stage1_budget = balance_rows_to_token_budget(oracle, reference_tokens, seed)
    stage2_segment_tokens = sum(int(row["input_token_count"]) for row in answer)
    stage2_segments = int(config["training"]["stage2_epochs"])
    segments = []
    if config["protocol"] == "oracle_two_stage":
        segments.append(("stage1", "oracle_priority_equal_token", stage1_rows, stage1_budget))
    else:
        direct_rows, direct_budget = balance_rows_to_token_budget(answer, reference_tokens, seed)
        segments.append(("direct_warm_budget", "answer_only", direct_rows, direct_budget))
    for epoch in range(1, stage2_segments + 1):
        rows, budget = balance_rows_to_token_budget(answer, stage2_segment_tokens, seed + epoch)
        segments.append((f"stage2_epoch{epoch}", "answer_only", rows, budget))
    audit = {
        "protocol": config["protocol"],
        "fairness_unit": "tokenized_input_tokens",
        "reference_method": "all_paths_equal_token",
        "reference_input_tokens": reference_tokens,
        "stage1_oracle_input_tokens": stage1_budget["input_tokens"],
        "answer_only_segment_input_tokens": stage2_segment_tokens,
        "segments": [
            {"name": name, "method": method, "input_tokens": budget["input_tokens"], "examples": budget["examples"], "overshoot_tokens": budget["overshoot_tokens"]}
            for name, method, _, budget in segments
        ],
        "total_input_tokens": sum(budget["input_tokens"] for _, _, _, budget in segments),
        "stage2_epochs": stage2_segments,
        "stage1_is_skipped": config["protocol"] == "direct_answer_only",
    }
    return segments, audit


def run_controlled(*, repo_root: Path, config_path: Path, config: dict, output_dir: Path, model_override: str | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    splits, paths, split_audit = load_splits(repo_root, config["data"])
    split_audit["paths"] = {split: str(path.relative_to(repo_root)) for split, path in paths.items()}
    split_audit["sha256"] = {split: sha256_file(path) for split, path in paths.items()}
    write_json(output_dir / "environment.json", environment_snapshot(repo_root))
    write_json(output_dir / "data_audit.json", split_audit)
    started = time.perf_counter()
    model, tokenizer, injection, device, dtype, resolved_model = load_runtime(config, model_override)
    segments, budget_audit = _prepare_rows(splits["train"], tokenizer, config, seed)
    write_json(output_dir / "controlled_budget_audit.json", budget_audit)
    diagnostics = [{"checkpoint": "initial", "stage": "before_training", "epoch": 0,
                    "metrics": evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, "initial")}]
    first_segment_steps = max(1, math.ceil(segments[0][3]["input_tokens"] / int(config["training"]["tokens_per_optimizer_step"])))
    answer_segment_steps = max(1, math.ceil(segments[1][3]["input_tokens"] / int(config["training"]["tokens_per_optimizer_step"])))
    estimated_steps = first_segment_steps + answer_segment_steps * (len(segments) - 1)
    from .experiment import _new_optimizer
    parameters, optimizer, scheduler = _new_optimizer(
        model, config, float(config["training"]["first_segment_learning_rate"]), first_segment_steps, config["training"]["first_segment_scheduler"]
    )
    segment_stats = []
    checkpoints = []
    segment_progress = tqdm(
        enumerate(segments),
        total=len(segments),
        desc=f"{config['protocol']} overall",
        unit="segment",
        disable=bool(config.get("runtime", {}).get("disable_progress", False)),
    )
    for index, (name, method, rows, budget) in segment_progress:
        if index == 1:
            # Both protocols use the same optimizer reset and answer-only schedule
            # after the matched first segment; only the first-segment rows differ.
            parameters, optimizer, scheduler = _new_optimizer(
                model, config, float(config["training"]["answer_only_learning_rate"]),
                answer_segment_steps * (len(segments) - 1), config["training"]["answer_only_scheduler"]
            )
        tagged_rows = [{**row, "task_id": f"{row['task_id']}:{name}"} for row in rows]
        stats = train_rows_once(
            model, tokenizer, tagged_rows, stage_name=name, epoch=index, seed=seed + 1000 + index,
            config=config, device=device, dtype=dtype,
            tokens_per_optimizer_step=int(config["training"]["tokens_per_optimizer_step"]),
            parameters=parameters, optimizer=optimizer, scheduler=scheduler,
        )
        logs = stats.pop("logs")
        write_jsonl(output_dir / f"training_log_{name}.jsonl", logs)
        stats["method"] = method
        stats["budget"] = budget
        segment_stats.append(stats)
        checkpoint = save_checkpoint(model, output_dir, name, stage=name, epoch=index)
        checkpoints.append(checkpoint)
        metrics = evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, name)
        diagnostics.append({"checkpoint": name, "stage": name, "epoch": index, "metrics": metrics})
        segment_progress.set_postfix(
            checkpoint=name,
            val=f"{metrics['graph_free_validation']['accuracy']:.3f}",
            test=f"{metrics['graph_free_test']['accuracy']:.3f}",
            refresh=False,
        )
    write_jsonl(output_dir / "controlled_metrics.jsonl", diagnostics)
    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol": config["protocol"],
        "seed": seed,
        "model": resolved_model,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "data": split_audit,
        "budget": budget_audit,
        "injection": {"target_modules": list(injection.target_modules), "replaced_modules": list(injection.replaced_modules),
                      "trainable_parameters": injection.trainable_parameters, "total_parameters": injection.total_parameters},
        "training": {"segments": segment_stats, "estimated_optimizer_steps": estimated_steps,
                     "actual_optimizer_steps": sum(item["optimizer_steps"] for item in segment_stats),
                     "total_input_tokens": sum(item["input_tokens"] for item in segment_stats)},
        "checkpoints": checkpoints,
        "metrics_file": "controlled_metrics.jsonl",
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "selection_rule": "max graph_free_validation accuracy; test read only from selected checkpoint",
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
