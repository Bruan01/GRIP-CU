"""Training, generation, and mechanism probes for one StructuredLoRA run."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import routing_kwargs
from .io_utils import append_jsonl, environment_snapshot, set_seed, sha256_file, write_json
from .metrics import normalize_entity, score_predictions
from .modules import adapter_state_dict, inject_lora
from .records import build_prompt, load_jsonl, records_by_depth, validate_splits
from .runtime_data import CausalCollator, ExactHopDataset


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]


def _device_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _count_trainable(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_runtime(config: dict, method: str, model_override: str | None = None):
    model_config = config["model"]
    model_name = model_override or model_config["name_or_path"]
    device = torch.device(model_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the config but torch.cuda.is_available() is false")
    dtype = _resolve_dtype(model_config["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=bool(model_config.get("local_files_only", False)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
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

    lora = config["lora"]
    routes = routing_kwargs(config)
    controller, report = inject_lora(
        model,
        method=method,
        target_modules=lora["target_modules"],
        total_rank=int(lora["total_rank"]),
        groups=int(lora["groups"]),
        group_rank=int(lora["group_rank"]),
        alpha=float(lora["alpha"]),
        dropout=float(lora["dropout"]),
        orthogonal_init=bool(lora["orthogonal_init"]),
        **routes,
    )
    if report.trainable_parameters != _count_trainable(model):
        raise AssertionError("injection trainable-parameter report is inconsistent")
    return model, tokenizer, controller, report, device, dtype, model_name


def load_splits(repo_root: Path, config: dict) -> tuple[dict[str, list[dict]], dict]:
    paths = {name: repo_root / config["data"][name] for name in ("train", "validation", "test")}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} split: {path}")
    splits = {name: load_jsonl(path) for name, path in paths.items()}
    audit = validate_splits(splits)
    expected = config["data"].get("expected_sizes", {})
    for name, expected_size in expected.items():
        if len(splits[name]) != int(expected_size):
            raise ValueError(f"{name} size changed: expected {expected_size}, found {len(splits[name])}")
    audit["sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    audit["paths"] = {name: str(path.relative_to(repo_root)) for name, path in paths.items()}
    return splits, audit


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_model(model, tokenizer, controller, rows: list[dict], config: dict, device, dtype) -> dict:
    training = config["training"]
    dataset = ExactHopDataset(rows, tokenizer, max_length=int(config["data"]["max_length"]))
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    loader = DataLoader(
        dataset,
        batch_size=int(training["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        drop_last=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_steps = int(training["max_optimizer_steps"])
    warmup_steps = max(1, round(max_steps * float(training["warmup_ratio"])))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max_steps - step
        return max(0.0, float(remaining) / float(max(1, max_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    accumulation = int(training["gradient_accumulation_steps"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    iterator = iter(loader)
    optimizer_step = 0
    micro_step = 0
    tokens_seen = 0
    loss_window: list[float] = []
    history: list[dict] = []

    while optimizer_step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        depths = batch.pop("depths")
        batch.pop("task_ids")
        batch.pop("answers")
        controller.set_batch(depths.detach().cpu().tolist())
        tokens_seen += int(batch["attention_mask"].sum().item())
        with _device_context(device, dtype):
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / accumulation
        loss.backward()
        loss_window.append(float(raw_loss.detach().cpu()))
        micro_step += 1
        if micro_step % accumulation != 0:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        if optimizer_step == 1 or optimizer_step % int(training["log_every_steps"]) == 0:
            record = {
                "optimizer_step": optimizer_step,
                "mean_loss": sum(loss_window) / len(loss_window),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "tokens_seen": tokens_seen,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        loss_window.clear()

    elapsed = time.perf_counter() - started
    return {
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "history": history,
    }


def generate_predictions(
    model,
    tokenizer,
    controller,
    rows: Iterable[dict],
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict], dict]:
    rows = list(rows)
    model.eval()
    model.config.use_cache = True
    predictions: list[dict] = []
    started = time.perf_counter()
    max_length = int(config["data"]["max_length"])
    max_new_tokens = int(config["data"]["max_new_tokens"])
    with torch.inference_mode():
        for row in rows:
            prompt = build_prompt(row["text"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            controller.set_batch([int(row["depth_label"])])
            with _device_context(device, dtype):
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffix = generated[0, encoded["input_ids"].shape[1] :]
            raw_text = tokenizer.decode(suffix, skip_special_tokens=True)
            prediction = normalize_entity(raw_text)
            predictions.append(
                {
                    "task_id": row["task_id"],
                    "depth_label": int(row["depth_label"]),
                    "answer": row["answer"],
                    "prediction_text": prediction,
                    "raw_generation": raw_text,
                    "correct": prediction == normalize_entity(str(row["answer"])),
                }
            )
    model.config.use_cache = False
    return predictions, {
        **score_predictions(predictions),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _balanced_subset(rows: list[dict], examples_per_depth: int) -> list[dict]:
    grouped = records_by_depth(rows)
    selected: list[dict] = []
    for depth in range(1, 5):
        selected.extend(sorted(grouped[depth], key=lambda row: row["task_id"])[:examples_per_depth])
    return selected


def run_group_knockout(model, tokenizer, controller, rows, config, device, dtype) -> dict:
    subset = _balanced_subset(rows, int(config["analysis"]["knockout_examples_per_depth"]))
    controller.set_knockout(None)
    _, baseline = generate_predictions(model, tokenizer, controller, subset, config, device, dtype)
    groups = int(config["lora"]["groups"])
    knockouts: dict[str, dict] = {}
    for group in range(groups):
        controller.set_knockout(group)
        _, metrics = generate_predictions(model, tokenizer, controller, subset, config, device, dtype)
        metrics["accuracy_delta_from_no_knockout"] = metrics["accuracy"] - baseline["accuracy"]
        knockouts[str(group + 1)] = metrics
    controller.set_knockout(None)
    return {
        "subset_size": len(subset),
        "baseline": baseline,
        "knockout_by_group": knockouts,
    }


def _flatten_gradients(model) -> torch.Tensor:
    chunks = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(parameter.grad.detach().float().cpu().reshape(-1))
    return torch.cat(chunks)


def run_gradient_probe(model, tokenizer, controller, rows, config, device, dtype) -> dict:
    grouped = records_by_depth(rows)
    batch_size = int(config["analysis"]["gradient_probe_batch_size"])
    collator = CausalCollator(tokenizer.pad_token_id)
    vectors: dict[int, torch.Tensor] = {}
    model.train()
    model.config.use_cache = False
    controller.set_knockout(None)
    for depth in range(1, 5):
        selected = sorted(grouped[depth], key=lambda row: row["task_id"])[:batch_size]
        dataset = ExactHopDataset(selected, tokenizer, max_length=int(config["data"]["max_length"]))
        batch = collator([dataset[index] for index in range(len(dataset))])
        batch = _move_batch(batch, device)
        depths = batch.pop("depths")
        batch.pop("task_ids")
        batch.pop("answers")
        controller.set_batch(depths.detach().cpu().tolist())
        model.zero_grad(set_to_none=True)
        with _device_context(device, dtype):
            loss = model(**batch).loss
        loss.backward()
        vectors[depth] = _flatten_gradients(model)
    model.zero_grad(set_to_none=True)
    cosine_matrix: dict[str, float] = {}
    for left in range(1, 5):
        for right in range(left, 5):
            value = F.cosine_similarity(vectors[left], vectors[right], dim=0, eps=1e-12).item()
            cosine_matrix[f"d{left}_d{right}"] = float(value)
    return {
        "batch_size_per_depth": batch_size,
        "gradient_norm_by_depth": {
            str(depth): float(torch.linalg.vector_norm(vector).item()) for depth, vector in vectors.items()
        },
        "cosine": cosine_matrix,
    }


def run_one(
    *,
    repo_root: Path,
    experiment_root: Path,
    config_path: Path,
    config: dict,
    method: str,
    seed: int,
    output_dir: Path,
    model_override: str | None,
) -> dict:
    if method not in config["methods"]:
        raise ValueError(f"method {method!r} is not enabled in config")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    splits, data_audit = load_splits(repo_root, config)
    write_json(output_dir / "environment.json", environment_snapshot(repo_root))
    write_json(output_dir / "data_audit.json", data_audit)
    started = time.perf_counter()
    model, tokenizer, controller, injection, device, dtype, resolved_model = load_runtime(
        config, method, model_override
    )
    training = train_model(model, tokenizer, controller, splits["train"], config, device, dtype)
    adapter_path = output_dir / "adapter_model.pt"
    torch.save(adapter_state_dict(model), adapter_path)
    adapter_artifact = {
        "path": adapter_path.name,
        "size_bytes": adapter_path.stat().st_size,
        "sha256": sha256_file(adapter_path),
        "git_policy": "ignored_weight_artifact; commit metadata, metrics, and predictions",
    }
    write_json(
        output_dir / "adapter_config.json",
        {
            "method": method,
            "seed": seed,
            "base_model": resolved_model,
            "lora": config["lora"],
            "routing_controls": config["routing_controls"],
            "replaced_modules": list(injection.replaced_modules),
        },
    )

    split_metrics: dict[str, dict] = {}
    for split_name in ("validation", "test"):
        predictions, metrics = generate_predictions(
            model, tokenizer, controller, splits[split_name], config, device, dtype
        )
        append_jsonl(output_dir / f"predictions_{split_name}.jsonl", predictions)
        split_metrics[split_name] = metrics

    gradient_probe = run_gradient_probe(
        model, tokenizer, controller, splits["train"], config, device, dtype
    )
    knockout = {}
    if method in config["analysis"]["knockout_methods"]:
        knockout = run_group_knockout(
            model, tokenizer, controller, splits["test"], config, device, dtype
        )

    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "method": method,
        "seed": seed,
        "model": resolved_model,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "data": data_audit,
        "injection": {
            "target_modules": list(injection.target_modules),
            "replaced_module_count": len(injection.replaced_modules),
            "replaced_modules": list(injection.replaced_modules),
            "trainable_parameters": injection.trainable_parameters,
            "total_parameters": injection.total_parameters,
        },
        "training": training,
        "adapter_artifact": adapter_artifact,
        "metrics": split_metrics,
        "gradient_probe": gradient_probe,
        "group_knockout": knockout,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
