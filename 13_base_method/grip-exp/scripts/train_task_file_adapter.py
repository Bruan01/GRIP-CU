"""Train a reusable Qwen LoRA adapter from an official GRIP task file.

This is a dataset-level alternative to the original per-test-graph launcher.
It consumes the already generated, provenance-tracked stage-1/stage-2 task
file and saves loadable adapter directories after each stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainingArguments


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grip.tasks import TaskDataset  # noqa: E402
from grip.paper_protocol import get_paper_protocol  # noqa: E402
from grip.training import train  # noqa: E402
from models import get_ft_model, get_hf_ft_tokenizer  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_source", default="local")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--paper_protocol",
        choices=["none", "wn18rr", "clegr_reasoning"],
        default="none",
        help="Apply the canonical paper settings. CLEGR paper runs must use the per-graph runner.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenize_max_length", type=int, default=4096)
    parser.add_argument("--max_context_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max_qa_samples", type=int, default=0)
    parser.add_argument("--stage1_epochs", type=float, default=1.0)
    parser.add_argument("--stage2_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", nargs="+", default=["down_proj", "up_proj", "gate_proj"])
    parser.add_argument("--s1_stop_loss_threshold", type=float, default=0.15)
    parser.add_argument("--s2_stop_loss_threshold", type=float, default=0.25)
    parser.add_argument("--s1_min_epoch", type=float, default=1.0)
    parser.add_argument("--s2_min_epoch", type=float, default=1.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gather_batches",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stage2_checkpoint_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_context_samples < 0 or args.max_qa_samples < 0:
        raise ValueError("sample caps must be non-negative; 0 means all samples")
    if args.paper_protocol == "clegr_reasoning":
        raise ValueError(
            "The paper CLEGR protocol trains one dedicated adapter per test graph. "
            "Use shells/qwen_grip_inf/clegr_reasoning_3090.sh (scripts/run_grip.py), "
            "not the dataset-level task-file adapter."
        )
    if args.paper_protocol == "wn18rr":
        protocol = get_paper_protocol("wn18rr")
        args.max_context_samples = 0
        args.max_qa_samples = 0
        args.stage1_epochs = protocol.stage1_max_epochs
        args.stage2_epochs = protocol.stage2_max_epochs
        args.lora_r = protocol.lora_r
        args.lora_alpha = protocol.lora_alpha
        if protocol.gradient_accumulation_steps % args.per_device_train_batch_size != 0:
            raise ValueError(
                "Paper effective batch size must be divisible by per-device batch size: "
                f"{protocol.gradient_accumulation_steps} % {args.per_device_train_batch_size}"
            )
        args.gradient_accumulation_steps = (
            protocol.gradient_accumulation_steps // args.per_device_train_batch_size
        )
        args.s1_stop_loss_threshold = protocol.early_stop_threshold
        args.s2_stop_loss_threshold = protocol.early_stop_threshold
        args.s1_min_epoch = protocol.stage1_min_epochs
        args.s2_min_epoch = protocol.stage2_min_epochs
        args.adam_epsilon = protocol.adam_epsilon
        args.gather_batches = protocol.gather_batches
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    heartbeat = args.output_dir / "heartbeat.json"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "status": "starting",
        "started_at": now_iso(),
        "task_file": str(args.task_file.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "config": config,
        "python": sys.executable,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(heartbeat, {"status": "starting", "updated_at": now_iso()})

    payload = json.loads(args.task_file.read_text(encoding="utf-8"))
    generation_config = payload.get("generation_config") or {}
    if args.paper_protocol != "none":
        protocol = get_paper_protocol(args.paper_protocol)
        expected = {
            "num_context_qa": protocol.context_qa,
            "num_reason_qa": protocol.reasoning_qa,
            "num_summarization": protocol.summarization,
            "format_as_instruction": protocol.format_as_instruction,
        }
        mismatches = {
            key: (generation_config.get(key), value)
            for key, value in expected.items()
            if generation_config.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Task file does not match the requested paper protocol. "
                f"Mismatches (actual, expected): {mismatches}. Regenerate the task file first."
            )
    all_context_samples = list(payload.get("context_samples") or [])
    context_samples = (
        all_context_samples[:args.max_context_samples]
        if args.max_context_samples > 0
        else all_context_samples
    )
    all_qa_samples = list(payload.get("qa_samples") or [])
    qa_samples = all_qa_samples[:args.max_qa_samples] if args.max_qa_samples > 0 else all_qa_samples
    if not context_samples or not qa_samples:
        raise ValueError(f"task file must contain non-empty context and QA samples: {args.task_file}")

    write_json(heartbeat, {
        "status": "loading_model",
        "updated_at": now_iso(),
        "context_samples": len(context_samples),
        "qa_samples": len(qa_samples),
    })
    tokenizer = get_hf_ft_tokenizer(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=args.local_files_only,
        padding_side="right",
        truncation_side="right",
        tokenize_max_length=args.tokenize_max_length,
    )
    dataset = TaskDataset(context_samples=context_samples, qa_samples=qa_samples, tokenizer=tokenizer)
    model, tokenizer = get_ft_model(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=args.local_files_only,
        padding_side="right",
        truncation_side="right",
        tokenize_max_length=args.tokenize_max_length,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.stage1_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        tf32=False,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
    )

    write_json(heartbeat, {
        "status": "training_stage1",
        "updated_at": now_iso(),
        "context_samples": len(context_samples),
        "qa_samples": len(qa_samples),
    })
    stage1_dir = args.output_dir / "stage1_adapter"
    stage2_dir = args.output_dir / "stage2_adapter"
    checkpoint_dir = args.output_dir / "stage2_checkpoints"
    model, tokenizer = train(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        training_dataset=dataset,
        involve_qa_epochs=int(args.stage2_epochs),
        gather_batches=args.gather_batches,
        s1_stop_loss_threshold=args.s1_stop_loss_threshold,
        s1_min_epoch=args.s1_min_epoch,
        s2_stop_loss_threshold=args.s2_stop_loss_threshold,
        s2_min_epoch=args.s2_min_epoch,
        stage1_output_dir=stage1_dir,
        stage2_output_dir=stage2_dir,
        stage2_checkpoint_dir=checkpoint_dir,
        stage2_checkpoint_interval=args.stage2_checkpoint_interval,
    )
    summary = {
        "status": "complete",
        "finished_at": now_iso(),
        "elapsed_seconds": time.time() - started,
        "context_samples": len(context_samples),
        "qa_samples": len(qa_samples),
        "stage1_adapter": str(stage1_dir.resolve()),
        "stage2_adapter": str(stage2_dir.resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(heartbeat, {"status": "complete", "updated_at": now_iso(), **summary})
    metadata.update(summary)
    write_json(args.output_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
