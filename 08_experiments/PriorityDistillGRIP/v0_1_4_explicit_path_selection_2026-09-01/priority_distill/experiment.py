"""Oracle-only diagnostic with checkpoint curves and dual evaluation."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .candidates import build_candidate_pool_for_query, build_candidate_pools
from .io_utils import environment_snapshot, set_seed, sha256_file, write_json, write_jsonl
from .metrics import parse_explicit_output, score_candidate_selection_predictions, score_predictions
from .modules import adapter_state_dict, inject_lora, load_adapter_state_dict
from .records import build_evaluation_prompt, load_splits
from .runtime_data import CausalCollator, TokenizedDataset, balance_rows_to_token_budget, tokenize_supervision_rows
from .supervision import build_method_supervision, build_oracle_evidence_prompt, _explicit_selection_prompt


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _device_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_runtime(config: dict, model_override: str | None = None):
    model_config = config["model"]
    model_name = model_override or model_config["name_or_path"]
    device = torch.device(model_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the config but is not available")
    dtype = _resolve_dtype(model_config["dtype"])
    load_kwargs = {
        "local_files_only": bool(model_config.get("local_files_only", False)),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "use_fast": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=bool(model_config.get("local_files_only", False)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", "eager"),
    )
    model.to(device)
    training = config["training"]
    if training.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    report = inject_lora(model, **config["lora"])
    counted = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if report.trainable_parameters != counted:
        raise AssertionError("LoRA trainable-parameter report is inconsistent")
    return model, tokenizer, report, device, dtype, model_name


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _scheduler(optimizer, estimated_steps: int, warmup_ratio: float, kind: str):
    if kind == "constant":
        warmup = max(1, math.ceil(estimated_steps * warmup_ratio))

        def factor(step: int) -> float:
            return max(min((step + 1) / warmup, 1.0), 1e-3)
    elif kind == "cosine":
        warmup = max(1, math.ceil(estimated_steps * warmup_ratio))

        def factor(step: int) -> float:
            if step < warmup:
                return max((step + 1) / warmup, 1e-3)
            progress = (step - warmup) / max(estimated_steps - warmup, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))), 0.0)
    else:
        raise ValueError(f"unknown scheduler: {kind}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _new_optimizer(model, config: dict, learning_rate: float, estimated_steps: int, scheduler_kind: str):
    training = config["training"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(training["weight_decay"]))
    scheduler = _scheduler(optimizer, estimated_steps, float(training["warmup_ratio"]), scheduler_kind)
    return parameters, optimizer, scheduler


def _empty_stage_stats(stage_name: str, epoch: int | None, tokens_per_optimizer_step: int, learning_rate: float) -> dict:
    return {
        "stage": stage_name,
        "epoch": epoch,
        "examples": 0,
        "input_tokens": 0,
        "supervised_tokens": 0,
        "padded_tokens": 0,
        "micro_batches": 0,
        "optimizer_steps": 0,
        "mean_token_weighted_loss": None,
        "elapsed_seconds": 0.0,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "learning_rate": learning_rate,
        "truncated_examples": 0,
        "truncated_prompt_tokens": 0,
    }


def train_rows_once(
    model,
    tokenizer,
    rows: list[dict],
    *,
    stage_name: str,
    epoch: int | None,
    seed: int,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_optimizer_step: int,
    parameters: list[torch.nn.Parameter],
    optimizer,
    scheduler,
) -> dict:
    if not rows:
        return _empty_stage_stats(stage_name, epoch, tokens_per_optimizer_step, optimizer.defaults["lr"])
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TokenizedDataset(rows),
        batch_size=int(training["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        pin_memory=device.type == "cuda",
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    model.config.use_cache = False
    started = time.perf_counter()
    accumulated_tokens = 0
    input_tokens = 0
    supervised_tokens = 0
    padded_tokens = 0
    micro_batches = 0
    optimizer_steps = 0
    loss_weighted_sum = 0.0
    logs = []

    def optimizer_update() -> None:
        nonlocal accumulated_tokens, optimizer_steps
        torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulated_tokens = 0

    progress = tqdm(
        loader,
        desc=f"{stage_name} train",
        unit="batch",
        leave=False,
        disable=bool(config.get("runtime", {}).get("disable_progress", False)),
    )
    for batch in progress:
        batch = _move_batch(batch, device)
        batch_input_tokens = int(batch.pop("input_token_count"))
        batch_supervised_tokens = int(batch.pop("supervised_token_count"))
        batch.pop("task_ids")
        batch.pop("depths")
        padded_tokens += int(batch["input_ids"].numel())
        with _device_context(device, dtype):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {stage_name} epoch={epoch}: {loss.item()}")
        (loss * (batch_input_tokens / tokens_per_optimizer_step)).backward()
        accumulated_tokens += batch_input_tokens
        input_tokens += batch_input_tokens
        supervised_tokens += batch_supervised_tokens
        micro_batches += 1
        loss_weighted_sum += float(loss.detach().item()) * batch_input_tokens
        if accumulated_tokens >= tokens_per_optimizer_step:
            optimizer_update()
            progress.set_postfix(
                loss=f"{loss.detach().item():.3f}",
                opt_steps=optimizer_steps,
                refresh=False,
            )
            if optimizer_steps % int(training["log_every_steps"]) == 0:
                logs.append({
                    "stage": stage_name,
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "input_tokens": input_tokens,
                    "mean_token_weighted_loss": loss_weighted_sum / input_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
    if accumulated_tokens > 0:
        optimizer_update()
    return {
        "stage": stage_name,
        "epoch": epoch,
        "examples": len(rows),
        "planned_input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "padded_tokens": padded_tokens,
        "micro_batches": micro_batches,
        "optimizer_steps": optimizer_steps,
        "mean_token_weighted_loss": loss_weighted_sum / input_tokens,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "learning_rate": optimizer.defaults["lr"],
        "last_learning_rate": scheduler.get_last_lr()[0],
        "truncated_examples": sum(int(row.get("truncated_prompt_tokens", 0)) > 0 for row in rows),
        "truncated_prompt_tokens": sum(int(row.get("truncated_prompt_tokens", 0)) for row in rows),
        "logs": logs,
    }


def prepare_training_rows(train_rows: list[dict], tokenizer, config: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    candidate_seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(train_rows, int(config["candidates"]["distractor_count"]), candidate_seed)
    oracle = build_method_supervision(train_rows, pools, "oracle_priority_equal_token", candidate_seed)
    oracle_tokenized = tokenize_supervision_rows(oracle, tokenizer, int(config["data"]["max_length"]))
    all_paths = build_method_supervision(train_rows, pools, "all_paths_equal_token", candidate_seed)
    all_paths_tokenized = tokenize_supervision_rows(all_paths, tokenizer, int(config["data"]["max_length"]))
    reference_tokens = sum(row["input_token_count"] for row in all_paths_tokenized)
    stage1_rows, budget = balance_rows_to_token_budget(oracle_tokenized, reference_tokens, seed)
    answers = build_method_supervision(train_rows, pools, "answer_only", candidate_seed)
    answer_tokenized = tokenize_supervision_rows(answers, tokenizer, int(config["data"]["max_length"]))
    audit = {
        "diagnostic_method": "oracle_priority_equal_token",
        "reference_method": "all_paths_equal_token",
        "reference_input_tokens": reference_tokens,
        "unbalanced_oracle_input_tokens": sum(row["input_token_count"] for row in oracle_tokenized),
        "selected_stage1_budget": budget,
        "stage1_truncated_examples": sum(int(row.get("truncated_prompt_tokens", 0)) > 0 for row in stage1_rows),
        "stage1_truncated_prompt_tokens": sum(int(row.get("truncated_prompt_tokens", 0)) for row in stage1_rows),
        "stage2_examples_per_epoch": len(answer_tokenized),
        "stage2_input_tokens_per_epoch": sum(row["input_token_count"] for row in answer_tokenized),
        "stage2_epochs": int(config["training"]["stage2_epochs"]),
        "stage2_total_input_tokens": sum(row["input_token_count"] for row in answer_tokenized) * int(config["training"]["stage2_epochs"]),
        "answer_prompt_is_same_as_stage2": True,
    }
    return stage1_rows, answer_tokenized, audit


def _prompt_for_condition(row: dict, condition: str, train_rows: list[dict], config: dict) -> tuple[str, dict | None]:
    if condition == "graph_free":
        return build_evaluation_prompt(row["text"]), None
    if condition == "oracle_evidence":
        return build_oracle_evidence_prompt(row), None
    if condition in {"candidate_selection", "candidate_selection_terminal_masked"}:
        pool = build_candidate_pool_for_query(
            row, train_rows, int(config["candidates"]["distractor_count"]), int(config["candidates"]["seed"])
        )
        prompt, gold_position = _explicit_selection_prompt(
            row["text"], pool, int(config["candidates"]["seed"]), row["task_id"],
            terminal_masked=condition == "candidate_selection_terminal_masked",
        )
        return prompt, {
            "selection_target": gold_position + 1,
            "candidate_count": len(pool),
            "uses_candidate_paths": True,
        }
    raise ValueError(f"unknown evaluation condition: {condition}")


def generate_predictions(model, tokenizer, rows: list[dict], train_rows: list[dict], config: dict, device: torch.device, dtype: torch.dtype, condition: str, checkpoint_name: str) -> tuple[list[dict], dict]:
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    batch_size = int(config["data"]["evaluation_batch_size"])
    predictions: list[dict] = []
    with torch.inference_mode():
        starts = range(0, len(rows), batch_size)
        progress = tqdm(
            starts, total=(len(rows) + batch_size - 1) // batch_size,
            desc=f"{condition} eval {checkpoint_name}", unit="batch", leave=False,
            disable=bool(config.get("runtime", {}).get("disable_progress", False)),
        )
        for start in progress:
            batch_rows = rows[start : start + batch_size]
            prompt_info = [_prompt_for_condition(row, condition, train_rows, config) for row in batch_rows]
            prompts = [item[0] for item in prompt_info]
            if condition == "graph_free" and any("Candidate" in prompt for prompt in prompts):
                raise AssertionError("graph-free evaluation prompt leaked training-time evidence")
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=int(config["data"]["max_length"]),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with _device_context(device, dtype):
                generated = model.generate(
                    **encoded, max_new_tokens=int(config["data"]["max_new_tokens"]), do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            generated_only = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            for row, prompt, prompt_item, prediction_text in zip(batch_rows, prompts, prompt_info, texts):
                info = prompt_item[1]
                parsed = parse_explicit_output(prediction_text) if condition.startswith("candidate_selection") else {}
                prediction = {
                    "task_id": row["task_id"], "depth_label": int(row["depth_label"]), "answer": row["answer"],
                    "prediction_text": prediction_text, "evaluation_prompt": prompt, "condition": condition,
                    "uses_gold_path": condition == "oracle_evidence",
                    "uses_candidate_paths": bool(info and info.get("uses_candidate_paths", False)),
                }
                if info:
                    prediction.update(info)
                if parsed:
                    prediction.update({"predicted_selection": parsed["selected_path"], "prediction_answer": parsed["answer"]})
                predictions.append(prediction)
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    if condition.startswith("candidate_selection"):
        return predictions, score_candidate_selection_predictions(predictions)
    return predictions, score_predictions(predictions)


def save_checkpoint(model, output_dir: Path, checkpoint_name: str, *, stage: str, epoch: int | None) -> dict:
    checkpoint_dir = output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / "adapter_model.pt"
    torch.save(adapter_state_dict(model), adapter_path)
    metadata = {
        "checkpoint": checkpoint_name,
        "stage": stage,
        "epoch": epoch,
        "adapter_path": str(adapter_path.relative_to(output_dir)),
        "adapter_size_bytes": adapter_path.stat().st_size,
        "adapter_sha256": sha256_file(adapter_path),
        "git_policy": "ignored weight artifact",
    }
    return metadata


def evaluate_checkpoint(model, tokenizer, splits: dict[str, list[dict]], config: dict, device: torch.device, dtype: torch.dtype, output_dir: Path, checkpoint_name: str, conditions: list[str] | None = None) -> dict:
    metrics = {}
    for condition in conditions or config["diagnostic"]["conditions"]:
        for split in config["diagnostic"]["splits"]:
            prediction_rows, split_metrics = generate_predictions(model, tokenizer, splits[split], splits["train"], config, device, dtype, condition, checkpoint_name)
            filename = f"predictions_{checkpoint_name}_{condition}_{split}.jsonl"
            write_jsonl(output_dir / filename, prediction_rows)
            metrics[f"{condition}_{split}"] = split_metrics
    return metrics


def run_one(*, repo_root: Path, config_path: Path, config: dict, output_dir: Path, model_override: str | None) -> dict:
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
    stage1_rows, stage2_rows, budget_audit = prepare_training_rows(splits["train"], tokenizer, config, seed)
    write_json(output_dir / "token_budget_audit.json", budget_audit)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    diagnostics = []
    if config["diagnostic"].get("evaluate_initial", True):
        diagnostics.append({"checkpoint": "initial", "stage": "before_training", "epoch": 0, "metrics": evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, "initial")})

    stage1_estimated = max(1, math.ceil(sum(row["input_token_count"] for row in stage1_rows) / int(config["training"]["stage1_tokens_per_optimizer_step"])))
    parameters, stage1_optimizer, stage1_scheduler = _new_optimizer(model, config, float(config["training"]["stage1_learning_rate"]), stage1_estimated, "cosine")
    stage1 = train_rows_once(
        model, tokenizer, stage1_rows, stage_name="stage1", epoch=0, seed=seed, config=config, device=device, dtype=dtype,
        tokens_per_optimizer_step=int(config["training"]["stage1_tokens_per_optimizer_step"]), parameters=parameters, optimizer=stage1_optimizer, scheduler=stage1_scheduler,
    )
    write_jsonl(output_dir / "training_log_stage1.jsonl", stage1.pop("logs"))
    checkpoints = [save_checkpoint(model, output_dir, "stage1_end", stage="stage1", epoch=0)]
    if config["diagnostic"].get("evaluate_after_stage1", True):
        diagnostics.append({"checkpoint": "stage1_end", "stage": "stage1", "epoch": 0, "metrics": evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, "stage1_end")})

    stage2_epochs = int(config["training"]["stage2_epochs"])
    per_epoch_tokens = sum(int(row["input_token_count"]) for row in stage2_rows)
    stage2_estimated = max(1, math.ceil(per_epoch_tokens * stage2_epochs / int(config["training"]["stage2_tokens_per_optimizer_step"])))
    parameters, stage2_optimizer, stage2_scheduler = _new_optimizer(model, config, float(config["training"]["stage2_learning_rate"]), stage2_estimated, config["training"]["stage2_scheduler"])
    stage2_epochs_stats = []
    for epoch in range(1, stage2_epochs + 1):
        epoch_rows = [{**row, "task_id": f"{row['task_id']}:stage2_epoch{epoch}"} for row in stage2_rows]
        stats = train_rows_once(
            model, tokenizer, epoch_rows, stage_name="stage2", epoch=epoch, seed=seed + 1000 + epoch, config=config, device=device, dtype=dtype,
            tokens_per_optimizer_step=int(config["training"]["stage2_tokens_per_optimizer_step"]), parameters=parameters, optimizer=stage2_optimizer, scheduler=stage2_scheduler,
        )
        logs = stats.pop("logs")
        write_jsonl(output_dir / f"training_log_stage2_epoch{epoch}.jsonl", logs)
        stage2_epochs_stats.append(stats)
        checkpoint_name = f"stage2_epoch{epoch}"
        checkpoints.append(save_checkpoint(model, output_dir, checkpoint_name, stage="stage2", epoch=epoch))
        if config["diagnostic"].get("evaluate_after_each_stage2_epoch", True):
            diagnostics.append({"checkpoint": checkpoint_name, "stage": "stage2", "epoch": epoch, "metrics": evaluate_checkpoint(model, tokenizer, splits, config, device, dtype, output_dir, checkpoint_name)})

    write_jsonl(output_dir / "diagnostic_metrics.jsonl", diagnostics)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "method": "oracle_priority_equal_token",
        "seed": seed,
        "model": resolved_model,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "data": split_audit,
        "token_budget": budget_audit,
        "injection": {
            "target_modules": list(injection.target_modules),
            "replaced_modules": list(injection.replaced_modules),
            "replaced_module_count": len(injection.replaced_modules),
            "trainable_parameters": injection.trainable_parameters,
            "total_parameters": injection.total_parameters,
        },
        "training": {
            "stage1": stage1,
            "stage2_epochs": stage2_epochs_stats,
            "stage2_total_input_tokens": sum(item["input_tokens"] for item in stage2_epochs_stats),
            "stage2_total_optimizer_steps": sum(item["optimizer_steps"] for item in stage2_epochs_stats),
            "stage2_scheduler": config["training"]["stage2_scheduler"],
        },
        "checkpoints": checkpoints,
        "diagnostic_metrics_file": "diagnostic_metrics.jsonl",
        "peak_gpu_memory_bytes": peak_memory,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "inference_graph_access": False,
        "oracle_evidence_condition_is_diagnostic_only": True,
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
