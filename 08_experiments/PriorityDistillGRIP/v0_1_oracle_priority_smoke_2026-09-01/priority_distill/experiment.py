"""Two-stage PriorityDistill training and graph-free evaluation."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .candidates import build_candidate_pools
from .io_utils import environment_snapshot, set_seed, sha256_file, write_json, write_jsonl
from .metrics import score_predictions
from .modules import adapter_state_dict, inject_lora
from .records import build_evaluation_prompt, load_splits
from .runtime_data import CausalCollator, TokenizedDataset, balance_rows_to_token_budget, tokenize_supervision_rows
from .supervision import STAGE1_METHODS, build_method_supervision


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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=bool(model_config.get("local_files_only", False)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=True,
    )
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
    if config["training"].get("gradient_checkpointing", False):
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


def _scheduler(optimizer, estimated_steps: int, warmup_ratio: float):
    warmup = max(1, math.ceil(estimated_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(estimated_steps - warmup, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_stage(
    model,
    tokenizer,
    rows: list[dict],
    *,
    stage_name: str,
    seed: int,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_optimizer_step: int,
    learning_rate: float,
    output_dir: Path,
) -> dict:
    if not rows:
        return {
            "stage": stage_name,
            "examples": 0,
            "input_tokens": 0,
            "supervised_tokens": 0,
            "padded_tokens": 0,
            "micro_batches": 0,
            "optimizer_steps": 0,
            "elapsed_seconds": 0.0,
            "tokens_per_optimizer_step": tokens_per_optimizer_step,
            "learning_rate": learning_rate,
            "truncated_examples": 0,
            "truncated_prompt_tokens": 0,
        }
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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(training["weight_decay"]))
    planned_input_tokens = sum(int(row["input_token_count"]) for row in rows)
    estimated_steps = max(1, math.ceil(planned_input_tokens / tokens_per_optimizer_step))
    scheduler = _scheduler(optimizer, estimated_steps, float(training["warmup_ratio"]))
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
    for batch in loader:
        batch = _move_batch(batch, device)
        batch_input_tokens = int(batch.pop("input_token_count"))
        batch_supervised_tokens = int(batch.pop("supervised_token_count"))
        batch.pop("task_ids")
        batch.pop("depths")
        padded_tokens += int(batch["input_ids"].numel())
        with _device_context(device, dtype):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {stage_name} after {micro_batches} micro-batches: {loss.item()}")
        weight = batch_input_tokens / tokens_per_optimizer_step
        (loss * weight).backward()
        accumulated_tokens += batch_input_tokens
        input_tokens += batch_input_tokens
        supervised_tokens += batch_supervised_tokens
        micro_batches += 1
        loss_weighted_sum += float(loss.detach().item()) * batch_input_tokens
        if accumulated_tokens >= tokens_per_optimizer_step:
            torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulated_tokens = 0
            if optimizer_steps % int(training["log_every_steps"]) == 0:
                logs.append({"stage": stage_name, "optimizer_step": optimizer_steps, "input_tokens": input_tokens, "mean_token_weighted_loss": loss_weighted_sum / input_tokens, "learning_rate": scheduler.get_last_lr()[0]})
    if accumulated_tokens > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    write_jsonl(output_dir / f"training_log_{stage_name}.jsonl", logs)
    return {
        "stage": stage_name,
        "examples": len(rows),
        "planned_input_tokens": planned_input_tokens,
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "padded_tokens": padded_tokens,
        "micro_batches": micro_batches,
        "optimizer_steps": optimizer_steps,
        "estimated_optimizer_steps": estimated_steps,
        "mean_token_weighted_loss": loss_weighted_sum / input_tokens,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "learning_rate": learning_rate,
        "truncated_examples": sum(int(row.get("truncated_prompt_tokens", 0)) > 0 for row in rows),
        "truncated_prompt_tokens": sum(int(row.get("truncated_prompt_tokens", 0)) for row in rows),
    }


def prepare_training_rows(train_rows: list[dict], tokenizer, config: dict, method: str, seed: int) -> tuple[list[dict], list[dict], dict]:
    candidate_seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(train_rows, int(config["candidates"]["distractor_count"]), candidate_seed)
    stage1_tokenized: dict[str, list[dict]] = {}
    for registered_method in STAGE1_METHODS:
        rows = build_method_supervision(train_rows, pools, registered_method, candidate_seed)
        stage1_tokenized[registered_method] = tokenize_supervision_rows(rows, tokenizer, int(config["data"]["max_length"]))
    reference = config["training"]["stage1_reference_method"]
    reference_tokens = sum(row["input_token_count"] for row in stage1_tokenized[reference])
    method_totals = {name: sum(row["input_token_count"] for row in rows) for name, rows in stage1_tokenized.items()}
    method_truncation = {
        name: {
            "examples": sum(int(row.get("truncated_prompt_tokens", 0)) > 0 for row in rows),
            "prompt_tokens": sum(int(row.get("truncated_prompt_tokens", 0)) for row in rows),
        }
        for name, rows in stage1_tokenized.items()
    }
    if method == "answer_only":
        stage1_rows = []
        stage1_budget = {"target_input_tokens": reference_tokens, "input_tokens": 0, "overshoot_tokens": 0, "examples": 0, "cycles": 0, "supervised_tokens": 0, "truncated_examples": 0, "truncated_prompt_tokens": 0}
    else:
        stage1_rows, stage1_budget = balance_rows_to_token_budget(stage1_tokenized[method], reference_tokens, seed)
    answer_rows = build_method_supervision(train_rows, pools, "answer_only", candidate_seed)
    answer_tokenized = tokenize_supervision_rows(answer_rows, tokenizer, int(config["data"]["max_length"]))
    stage2_rows = []
    for epoch in range(int(config["training"]["stage2_epochs"])):
        for row in answer_tokenized:
            stage2_rows.append({**row, "task_id": f"{row['task_id']}:stage2_epoch{epoch}"})
    audit = {
        "reference_method": reference,
        "reference_input_tokens": reference_tokens,
        "unbalanced_stage1_input_tokens": method_totals,
        "unbalanced_stage1_truncation": method_truncation,
        "selected_method_budget": stage1_budget,
        "stage2_epochs": int(config["training"]["stage2_epochs"]),
        "stage2_input_tokens": sum(row["input_token_count"] for row in stage2_rows),
        "candidate_pool_count": len(pools),
    }
    return stage1_rows, stage2_rows, audit


def generate_predictions(model, tokenizer, rows: list[dict], config: dict, device: torch.device, dtype: torch.dtype) -> tuple[list[dict], dict]:
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    batch_size = int(config["data"]["evaluation_batch_size"])
    predictions: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = [build_evaluation_prompt(row["text"]) for row in batch_rows]
            if any("Candidate evidence:" in prompt for prompt in prompts):
                raise AssertionError("evaluation prompt leaked training-time evidence")
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(config["data"]["max_length"]))
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with _device_context(device, dtype):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(config["data"]["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_only = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            for row, prompt, prediction_text in zip(batch_rows, prompts, texts):
                predictions.append({"task_id": row["task_id"], "depth_label": int(row["depth_label"]), "answer": row["answer"], "prediction_text": prediction_text, "evaluation_prompt": prompt})
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    return predictions, score_predictions(predictions)


def run_one(
    *,
    repo_root: Path,
    config_path: Path,
    config: dict,
    method: str,
    seed: int,
    output_dir: Path,
    model_override: str | None,
) -> dict:
    if method not in config["methods"]:
        raise ValueError(f"method not registered: {method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    splits, paths, split_audit = load_splits(repo_root, config["data"])
    split_audit["paths"] = {split: str(path.relative_to(repo_root)) for split, path in paths.items()}
    split_audit["sha256"] = {split: sha256_file(path) for split, path in paths.items()}
    write_json(output_dir / "environment.json", environment_snapshot(repo_root))
    write_json(output_dir / "data_audit.json", split_audit)
    started = time.perf_counter()
    model, tokenizer, injection, device, dtype, resolved_model = load_runtime(config, model_override)
    stage1_rows, stage2_rows, budget_audit = prepare_training_rows(splits["train"], tokenizer, config, method, seed)
    write_json(output_dir / "token_budget_audit.json", budget_audit)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stage1 = train_stage(
        model, tokenizer, stage1_rows, stage_name="stage1", seed=seed, config=config, device=device, dtype=dtype,
        tokens_per_optimizer_step=int(config["training"]["stage1_tokens_per_optimizer_step"]),
        learning_rate=float(config["training"]["stage1_learning_rate"]), output_dir=output_dir,
    )
    stage2 = train_stage(
        model, tokenizer, stage2_rows, stage_name="stage2", seed=seed + 1000, config=config, device=device, dtype=dtype,
        tokens_per_optimizer_step=int(config["training"]["stage2_tokens_per_optimizer_step"]),
        learning_rate=float(config["training"]["stage2_learning_rate"]), output_dir=output_dir,
    )
    adapter_path = output_dir / "adapter_model.pt"
    torch.save(adapter_state_dict(model), adapter_path)
    metrics = {}
    for split in ("validation", "test"):
        prediction_rows, split_metrics = generate_predictions(model, tokenizer, splits[split], config, device, dtype)
        write_jsonl(output_dir / f"predictions_{split}.jsonl", prediction_rows)
        metrics[split] = split_metrics
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "method": method,
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
        "training": {"stage1": stage1, "stage2": stage2, "total_input_tokens": stage1["input_tokens"] + stage2["input_tokens"], "total_optimizer_steps": stage1["optimizer_steps"] + stage2["optimizer_steps"]},
        "metrics": metrics,
        "peak_gpu_memory_bytes": peak_memory,
        "adapter_artifact": {"path": adapter_path.name, "size_bytes": adapter_path.stat().st_size, "sha256": sha256_file(adapter_path), "git_policy": "ignored weight artifact"},
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "inference_graph_access": False,
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
