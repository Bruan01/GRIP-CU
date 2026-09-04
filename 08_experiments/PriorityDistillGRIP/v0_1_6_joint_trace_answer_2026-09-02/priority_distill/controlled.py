"""Controlled candidate-selection and Stage-2 replay experiment."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from tqdm import tqdm

from .candidates import build_candidate_pools
from .experiment import evaluate_checkpoint, load_adapter_state_dict, load_runtime, save_checkpoint, train_rows_once
from .io_utils import environment_snapshot, set_seed, sha256_file, write_json, write_jsonl
from .records import load_splits
from .runtime_data import balance_rows_to_token_budget, tokenize_supervision_rows
from .supervision import build_method_supervision


PROTOCOLS = {
    "direct_answer_only": {"stage1_method": "answer_only", "replay_method": None, "stage2_replay_ratio": 0.0},
    "oracle_two_stage": {"stage1_method": "oracle_priority_equal_token", "replay_method": None, "stage2_replay_ratio": 0.0},
    "candidate_selection_two_stage": {"stage1_method": "all_paths_equal_token", "replay_method": None, "stage2_replay_ratio": 0.0},
    "candidate_selection_anti_copy": {"stage1_method": "all_paths_terminal_masked_equal_token", "replay_method": None, "stage2_replay_ratio": 0.0},
    "candidate_selection_anti_copy_replay": {"stage1_method": "all_paths_terminal_masked_equal_token", "replay_method": "all_paths_terminal_masked_equal_token", "stage2_replay_ratio": 0.2},
    "candidate_index_two_stage": {"stage1_method": "explicit_path_selection", "replay_method": None, "stage2_replay_ratio": 0.0},
    "candidate_index_anti_copy": {"stage1_method": "explicit_path_selection_terminal_masked", "replay_method": None, "stage2_replay_ratio": 0.0},
    "candidate_index_anti_copy_replay": {"stage1_method": "explicit_path_selection_terminal_masked", "replay_method": "explicit_path_selection_terminal_masked", "stage2_replay_ratio": 0.2},
    # Trace supervision has no graph/candidate evidence in its input.  The
    # intermediate nodes are present only in the supervised target.
    "graph_free_trace": {
        "stage1_method": "graph_free_trace_terminal_masked",
        "stage2_primary_method": "answer_only",
        "replay_method": None,
        "stage2_replay_ratio": 0.0,
    },
    # Jointly supervise the structured trace and final answer in every phase.
    # The trace tokens are down-weighted (0.25) while answer tokens retain
    # weight 1.0; this prevents the longer trace from dominating the loss.
    "graph_free_trace_joint": {
        "stage1_method": "graph_free_trace_answer_joint",
        "stage2_primary_method": "graph_free_trace_answer_joint",
        "replay_method": None,
        "stage2_replay_ratio": 0.0,
    },
    # First learn to select an anti-copy candidate; then use graph-free trace
    # targets as the main Stage-2 objective while retaining 20% explicit
    # selection replay.
    "candidate_index_anti_copy_bridge": {
        "stage1_method": "explicit_path_selection_terminal_masked",
        "stage2_primary_method": "graph_free_trace_terminal_masked",
        "replay_method": "explicit_path_selection_terminal_masked",
        "stage2_replay_ratio": 0.2,
    },
}


def _tokenize(rows: list[dict], tokenizer, config: dict) -> list[dict]:
    return tokenize_supervision_rows(rows, tokenizer, int(config["data"]["max_length"]))


def _prepare_rows(train_rows, tokenizer, config, seed):
    protocol = config["protocol"]
    protocol_spec = PROTOCOLS[protocol]
    candidate_seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(
        train_rows,
        int(config["candidates"]["distractor_count"]),
        candidate_seed,
    )
    answer = _tokenize(
        build_method_supervision(train_rows, pools, "answer_only", candidate_seed),
        tokenizer,
        config,
    )
    all_paths = _tokenize(
        build_method_supervision(train_rows, pools, "all_paths_equal_token", candidate_seed),
        tokenizer,
        config,
    )
    oracle = _tokenize(
        build_method_supervision(train_rows, pools, "oracle_priority_equal_token", candidate_seed),
        tokenizer,
        config,
    )
    anti_copy = _tokenize(
        build_method_supervision(train_rows, pools, "all_paths_terminal_masked_equal_token", candidate_seed),
        tokenizer,
        config,
    )
    explicit = _tokenize(
        build_method_supervision(train_rows, pools, "explicit_path_selection", candidate_seed),
        tokenizer,
        config,
    )
    explicit_anti_copy = _tokenize(
        build_method_supervision(train_rows, pools, "explicit_path_selection_terminal_masked", candidate_seed),
        tokenizer,
        config,
    )
    trace = _tokenize(
        build_method_supervision(train_rows, pools, "graph_free_trace_terminal_masked", candidate_seed),
        tokenizer,
        config,
    )
    joint = _tokenize(
        build_method_supervision(train_rows, pools, "graph_free_trace_answer_joint", candidate_seed),
        tokenizer,
        config,
    )

    reference_tokens = sum(int(row["input_token_count"]) for row in explicit)
    answer_tokens = sum(int(row["input_token_count"]) for row in answer)
    trace_tokens = sum(int(row["input_token_count"]) for row in trace)
    stage1_method = protocol_spec["stage1_method"]
    stage1_source = {
        "answer_only": answer,
        "oracle_priority_equal_token": oracle,
        "all_paths_equal_token": all_paths,
        "all_paths_terminal_masked_equal_token": anti_copy,
        "explicit_path_selection": explicit,
        "explicit_path_selection_terminal_masked": explicit_anti_copy,
        "graph_free_trace_terminal_masked": trace,
        "graph_free_trace_answer_joint": joint,
    }[stage1_method]
    stage1_rows, stage1_budget = balance_rows_to_token_budget(stage1_source, reference_tokens, seed)

    replay_ratio = float(config["training"].get("stage2_replay_ratio", protocol_spec["stage2_replay_ratio"]))
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("training.stage2_replay_ratio must be in [0, 1)")
    stage2_epochs = int(config["training"]["stage2_epochs"])
    stage2_segments = []
    stage2_primary_method = protocol_spec.get("stage2_primary_method", "answer_only")
    stage2_primary_source = {
        "answer_only": answer,
        "graph_free_trace_terminal_masked": trace,
        "graph_free_trace_answer_joint": joint,
    }[stage2_primary_method]
    # Keep every Stage-2 protocol on the same tokenized input budget as the
    # answer-only baseline.  Trace targets are longer, so they intentionally
    # receive fewer examples, but not more total input-token updates.
    stage2_native_tokens = {
        "answer_only": answer_tokens,
        "graph_free_trace_terminal_masked": trace_tokens,
        "graph_free_trace_answer_joint": sum(int(row["input_token_count"]) for row in joint),
    }[stage2_primary_method]
    stage2_primary_tokens = answer_tokens
    for epoch in range(1, stage2_epochs + 1):
        if replay_ratio == 0.0:
            rows, budget = balance_rows_to_token_budget(
                stage2_primary_source, stage2_primary_tokens, seed + epoch
            )
            stage2_segments.append((f"stage2_epoch{epoch}", stage2_primary_method, rows, budget))
            continue
        # The primary objective receives (1 - replay_ratio) of the fixed
        # per-epoch token budget; the auxiliary objective receives the rest.
        primary_target = max(1, int(round(stage2_primary_tokens * (1.0 - replay_ratio))))
        auxiliary_target = max(1, stage2_primary_tokens - primary_target)
        primary_rows, primary_budget = balance_rows_to_token_budget(
            stage2_primary_source, primary_target, seed + epoch
        )
        replay_method = protocol_spec["replay_method"]
        replay_source = {
            "all_paths_equal_token": all_paths,
            "all_paths_terminal_masked_equal_token": anti_copy,
            "explicit_path_selection": explicit,
            "explicit_path_selection_terminal_masked": explicit_anti_copy,
            "graph_free_trace_terminal_masked": trace,
            "graph_free_trace_answer_joint": joint,
            "answer_only": answer,
        }[replay_method]
        replay_rows, replay_budget = balance_rows_to_token_budget(
            replay_source, auxiliary_target, seed + epoch + 50000
        )
        rows = primary_rows + replay_rows
        budget = {
            "target_input_tokens": stage2_primary_tokens,
            "input_tokens": primary_budget["input_tokens"] + replay_budget["input_tokens"],
            "overshoot_tokens": primary_budget["overshoot_tokens"] + replay_budget["overshoot_tokens"],
            "examples": len(rows),
            "cycles": max(primary_budget["cycles"], replay_budget["cycles"]),
            "supervised_tokens": primary_budget["supervised_tokens"] + replay_budget["supervised_tokens"],
            "truncated_examples": primary_budget["truncated_examples"] + replay_budget["truncated_examples"],
            "truncated_prompt_tokens": primary_budget["truncated_prompt_tokens"] + replay_budget["truncated_prompt_tokens"],
            "primary_method": stage2_primary_method,
            "primary_input_tokens": primary_budget["input_tokens"],
            "auxiliary_method": replay_method,
            "auxiliary_input_tokens": replay_budget["input_tokens"],
            # Keep the legacy field names for old reports, but explicitly
            # label the generic primary/auxiliary fields above.
            "answer_only_input_tokens": primary_budget["input_tokens"] if stage2_primary_method == "answer_only" else replay_budget["input_tokens"] if replay_method == "answer_only" else 0,
            "evidence_replay_input_tokens": replay_budget["input_tokens"] if stage2_primary_method == "answer_only" else 0,
            "answer_only_examples": primary_budget["examples"] if stage2_primary_method == "answer_only" else replay_budget["examples"] if replay_method == "answer_only" else 0,
            "evidence_replay_examples": replay_budget["examples"] if stage2_primary_method == "answer_only" else 0,
        }
        stage2_segments.append((f"stage2_epoch{epoch}", f"{stage2_primary_method}_plus_{replay_method}", rows, budget))

    segments = []
    if protocol == "direct_answer_only":
        direct_rows, direct_budget = balance_rows_to_token_budget(answer, reference_tokens, seed)
        segments.append(("direct_warm_budget", "answer_only", direct_rows, direct_budget))
    else:
        segments.append(("stage1", stage1_method, stage1_rows, stage1_budget))
    segments.extend(stage2_segments)
    audit = {
        "protocol": protocol,
        "fairness_unit": "tokenized_input_tokens",
        "reference_method": "all_paths_equal_token",
        "reference_input_tokens": reference_tokens,
        "answer_only_input_tokens_per_epoch": answer_tokens,
        "stage2_primary_method": stage2_primary_method,
        "stage2_primary_input_tokens_per_epoch": stage2_primary_tokens,
        "stage2_primary_native_input_tokens_per_epoch": stage2_native_tokens,
        "graph_free_trace_input_tokens_per_epoch": trace_tokens,
        "stage1_method": stage1_method,
        "stage1_input_tokens": segments[0][3]["input_tokens"],
        "stage1_supervised_tokens": segments[0][3]["supervised_tokens"],
        "stage2_replay_ratio": replay_ratio,
        "segments": [
            {
                "name": name,
                "method": method,
                "input_tokens": budget["input_tokens"],
                "supervised_tokens": budget["supervised_tokens"],
                "examples": budget["examples"],
                "overshoot_tokens": budget["overshoot_tokens"],
                "primary_method": budget.get("primary_method", method),
                "primary_input_tokens": budget.get("primary_input_tokens", budget["input_tokens"]),
                "auxiliary_method": budget.get("auxiliary_method"),
                "auxiliary_input_tokens": budget.get("auxiliary_input_tokens"),
                "answer_only_input_tokens": budget.get("answer_only_input_tokens"),
                "evidence_replay_input_tokens": budget.get("evidence_replay_input_tokens"),
            }
            for name, method, _, budget in segments
        ],
        "total_input_tokens": sum(budget["input_tokens"] for _, _, _, budget in segments),
        "total_supervised_tokens": sum(budget["supervised_tokens"] for _, _, _, budget in segments),
        "stage2_epochs": stage2_epochs,
        "stage1_is_skipped": protocol == "direct_answer_only",
        "candidate_selection_requires_matching": stage1_method in {
            "all_paths_equal_token",
            "all_paths_terminal_masked_equal_token",
            "explicit_path_selection",
            "explicit_path_selection_terminal_masked",
        },
        "terminal_masked": stage1_method in {
            "all_paths_terminal_masked_equal_token",
            "explicit_path_selection_terminal_masked",
        },
        "replay_method": protocol_spec["replay_method"],
    }
    return segments, audit


def run_controlled(*, repo_root: Path, config_path: Path, config: dict, output_dir: Path, model_override: str | None, seed_override: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the registered JSON config immutable, while allowing a reproducible
    # seed sweep to reuse the exact same protocol and data definition.
    if seed_override is not None:
        config = copy.deepcopy(config)
        if int(seed_override) < 0:
            raise ValueError("seed_override must be non-negative")
        config["training"]["seed"] = int(seed_override)
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
    all_conditions = list(config["diagnostic"]["conditions"])
    graph_free_only = ["graph_free"]
    diagnostics = []
    if config["diagnostic"].get("evaluate_initial", False):
        diagnostics.append({
            "checkpoint": "initial",
            "stage": "before_training",
            "epoch": 0,
            "metrics": evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, "initial", conditions=graph_free_only),
        })
    first_segment_steps = max(1, math.ceil(segments[0][3]["input_tokens"] / int(config["training"]["tokens_per_optimizer_step"])))
    stage2_steps = max(1, math.ceil(segments[1][3]["input_tokens"] / int(config["training"]["tokens_per_optimizer_step"])))
    estimated_steps = first_segment_steps + stage2_steps * (len(segments) - 1)
    from .experiment import _new_optimizer
    parameters, optimizer, scheduler = _new_optimizer(
        model,
        config,
        float(config["training"]["first_segment_learning_rate"]),
        first_segment_steps,
        config["training"]["first_segment_scheduler"],
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
            parameters, optimizer, scheduler = _new_optimizer(
                model,
                config,
                float(config["training"]["answer_only_learning_rate"]),
                stage2_steps * (len(segments) - 1),
                config["training"]["answer_only_scheduler"],
            )
        tagged_rows = [{**row, "task_id": f"{row['task_id']}:{name}"} for row in rows]
        stats = train_rows_once(
            model,
            tokenizer,
            tagged_rows,
            stage_name=name,
            epoch=index,
            seed=seed + 1000 + index,
            config=config,
            device=device,
            dtype=dtype,
            tokens_per_optimizer_step=int(config["training"]["tokens_per_optimizer_step"]),
            parameters=parameters,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logs = stats.pop("logs")
        write_jsonl(output_dir / f"training_log_{name}.jsonl", logs)
        stats["method"] = method
        stats["budget"] = budget
        segment_stats.append(stats)
        checkpoint = save_checkpoint(model, output_dir, name, stage=name, epoch=index)
        checkpoints.append(checkpoint)
        # Graph-free validation is the only checkpoint-selection signal.  The
        # more expensive candidate diagnostics are run at stage 1 and again
        # only for the validation-selected checkpoint below.
        evaluate_stage1_all = bool(config.get("diagnostic", {}).get("evaluate_stage1_all_conditions", True))
        eval_conditions = all_conditions if (index == 0 and evaluate_stage1_all) else graph_free_only
        metrics = evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, name, conditions=eval_conditions)
        diagnostics.append({"checkpoint": name, "stage": name, "epoch": index, "metrics": metrics})
        segment_progress.set_postfix(
            checkpoint=name,
            val=f"{metrics['graph_free_validation']['accuracy']:.3f}",
            test=f"{metrics['graph_free_test']['accuracy']:.3f}",
            refresh=False,
        )
    trained = [item for item in diagnostics if item["checkpoint"] != "initial"]
    selected = max(trained, key=lambda item: (item["metrics"]["graph_free_validation"]["accuracy"], -item["epoch"]))
    selected_checkpoint = next(item for item in checkpoints if item["checkpoint"] == selected["checkpoint"])
    state = __import__("torch").load(output_dir / selected_checkpoint["adapter_path"], map_location="cpu", weights_only=True)
    load_adapter_state_dict(model, state)
    selected_full_metrics = evaluate_checkpoint(
        model, tokenizer, splits, config, device, dtype, output_dir, selected["checkpoint"], conditions=all_conditions
    )
    selected["metrics"] = selected_full_metrics
    for item in diagnostics:
        if item["checkpoint"] == selected["checkpoint"]:
            item["metrics"] = selected_full_metrics
    write_jsonl(output_dir / "controlled_metrics.jsonl", diagnostics)
    write_json(output_dir / "selected_checkpoint_metrics.json", {
        "selection_rule": "max graph_free_validation accuracy; test read only from selected checkpoint",
        "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": selected["epoch"],
        "metrics": selected_full_metrics,
    })
    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol": config["protocol"],
        "seed": seed,
        "seed_override": seed_override,
        "model": resolved_model,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "data": split_audit,
        "budget": budget_audit,
        "injection": {
            "target_modules": list(injection.target_modules),
            "replaced_modules": list(injection.replaced_modules),
            "trainable_parameters": injection.trainable_parameters,
            "total_parameters": injection.total_parameters,
        },
        "training": {
            "segments": segment_stats,
            "estimated_optimizer_steps": estimated_steps,
            "actual_optimizer_steps": sum(item["optimizer_steps"] for item in segment_stats),
            "total_input_tokens": sum(item["input_tokens"] for item in segment_stats),
            "total_supervised_tokens": sum(item["supervised_tokens"] for item in segment_stats),
        },
        "checkpoints": checkpoints,
        "metrics_file": "controlled_metrics.jsonl",
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "selection_rule": "max graph_free_validation accuracy; test read only from selected checkpoint",
        "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": selected["epoch"],
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
