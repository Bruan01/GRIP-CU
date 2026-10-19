"""Train original GRIP with candidate-level InfoNCE.

The ``train_graph`` arm is the matched Random-9 control. The ``error_memory``
arm reads the frozen train-only ErrorMemory-9 bank produced by
``scripts/mine_error_memory_negatives.py``. Evaluation uses a standard GRIP
processed graph record supplied by ``--eval_file``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import TrainingArguments

HNG = Path(__file__).resolve().parents[1]
REPO_ROOT = HNG.parents[1]
GRIP_EXP = Path(
    os.environ.get(
        "GRIP_BASELINE_DIR", str(REPO_ROOT / "13_base_method" / "grip-exp")
    )
)
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, SYSTEM_PROMPT, TORCH_DTYPE  # noqa: E402
from evaluation.utils import normalize_answer  # noqa: E402
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset  # noqa: E402
from grip.tasks.train_tasks.gen_context import GenGraphContextTask  # noqa: E402
from grip.tasks.train_tasks.task_dataset import TaskDataset  # noqa: E402
from grip.training.train import load_trainer  # noqa: E402
from hard_negative_grip.decode_io import (  # noqa: E402
    append_jsonl, eval_samples, load_jsonl, merge_resumed_rows, write_jsonl,
)
from hard_negative_grip.listed_training import (  # noqa: E402
    ListedContrastiveTrainer,
    ListedDataCollator,
    ListedQADataset,
    format_answer_prefix,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    listed_relations_from_sample,
    load_train_relation_order,
)
from hard_negative_grip.run_protection import (  # noqa: E402
    adapter_is_complete,
    checkpoint_training_kwargs,
    resolve_resume_checkpoint,
)
from hard_negative_grip.swanlab_tracking import SwanLabTracker  # noqa: E402
from hard_negative_grip.training_budget import (  # noqa: E402
    assert_error_memory_training_budget,
)
from hard_negative_grip.task_file import (  # noqa: E402
    LISTED_NEGATIVE_K,
    LISTED_NEGATIVE_SOURCES,
    assistant_gold,
    build_qa_assets_from_task_texts,
    is_grip_task_file,
    is_relation_gold,
    load_graph_record,
    load_json_payload,
    load_error_memory_manifest,
    load_counterfactual_manifest,
    original_question_ids_from_payload,
    parse_model_answer,
    train_relation_alias_index,
    validate_structure_counterfactual_manifest,
    validate_error_memory_manifest,
)
from models.utils import get_hf_llm_tokenizer, get_lora_model  # noqa: E402
from utils import set_random_seed  # noqa: E402

QUESTION_TEMPLATE = f"{GRIPEvalDataset.template} {GRIPEvalDataset.answer_format}"


def exact_match(prediction: str, targets: list[str]) -> bool:
    normalized = normalize_answer(str(prediction))
    return any(normalized == normalize_answer(str(target)) for target in targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Original GRIP task JSON or a standard processed graph record.",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=None,
        help="Standard processed GRIP graph record for val/test EM. "
        "Required when --input_file is a task JSON.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["s1", "b1", "listed", "all", "compare"],
        default="all",
        help="s1 = shared graph storage; b1/listed = Stage-2 forks; compare = merge EM summaries",
    )
    parser.add_argument("--s1_adapter", type=Path, default=None)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument(
        "--target_modules", nargs="+", default=["down_proj", "up_proj", "gate_proj"]
    )
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--involve_qa_epochs", type=float, default=10)
    parser.add_argument("--s1_stop_loss_threshold", type=float, default=0.15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_candidate", type=float, default=1.0)
    parser.add_argument(
        "--memory_size",
        type=int,
        default=0,
        help="A2 cross-batch relation memory size. 0 disables memory.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--swanlab_project", default="",
                        help="Opt-in SwanLab project. Empty preserves the old no-tracking behavior.")
    parser.add_argument("--swanlab_experiment_name", default="")
    parser.add_argument("--swanlab_group", default="")
    parser.add_argument("--swanlab_mode", choices=("online", "offline"), default="online")
    parser.add_argument(
        "--s1_gradient_checkpointing",
        action="store_true",
        help="Enable checkpointing for Stage 1. Needed for Qwen2.5-7B on 24GB.",
    )
    parser.add_argument(
        "--listed_negative_k",
        type=int,
        default=LISTED_NEGATIVE_K,
        help="Negatives sampled per relation-like paper QA item (task-file path only).",
    )
    parser.add_argument(
        "--listed_negative_source",
        choices=list(LISTED_NEGATIVE_SOURCES),
        default="train_graph",
        help=(
            "train_graph = matched Random-9 control; error_memory = frozen "
            "train-only GRIP-conditioned bank; structure_counterfactual = "
            "frozen train-only graph-profile bank."
        ),
    )
    parser.add_argument(
        "--error_memory_manifest",
        type=Path,
        default=None,
        help="Immutable ErrorMemory-9 JSONL manifest; required for error_memory.",
    )
    parser.add_argument(
        "--counterfactual_manifest",
        type=Path,
        default=None,
        help=(
            "Immutable structure-counterfactual JSONL manifest; required for "
            "structure_counterfactual."
        ),
    )
    parser.add_argument(
        "--counterfactual_metadata",
        type=Path,
        default=None,
        help="Optional sampler metadata JSON recorded with the run configuration.",
    )
    parser.add_argument("--counterfactual_uniform_mix", type=float, default=1.0 / 3.0)
    parser.add_argument("--counterfactual_structure_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_violation_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_temperature", type=float, default=0.25)
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset label for provenance and SwanLab tags; inferred when omitted.",
    )
    parser.add_argument(
        "--negative_sampling_mode",
        choices=("random", "error_memory", "structure_counterfactual"),
        default="random",
        help=(
            "For paired listed-negative runs. error_memory and "
            "structure_counterfactual require their frozen train-only manifest; "
            "default preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--error_memory_uniform_mix",
        type=float,
        default=1.0 / 3.0,
        help="Recorded sampler policy: uniform probability-floor mixture weight.",
    )
    parser.add_argument(
        "--error_memory_weight",
        type=float,
        default=1.0,
        help="Recorded sampler policy: gold-conditioned error-memory weight.",
    )
    parser.add_argument(
        "--error_memory_violation_weight",
        type=float,
        default=1.0,
        help="Recorded sampler policy: B1 score-violation weight.",
    )
    parser.add_argument(
        "--error_memory_temperature",
        type=float,
        default=0.25,
        help="Recorded sampler policy: temperature used by the frozen sampler.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=DEFAULT_RAW_NELL23K,
        help="Raw split directory (train.txt) for train_graph negatives.",
    )
    parser.add_argument(
        "--relation_text_file",
        type=Path,
        default=None,
        help="Optional raw-relation to task-label mapping, required for FB15K237-2.",
    )
    parser.add_argument(
        "--listed_negative_seed",
        type=int,
        default=None,
        help="Private RandomState seed for train_graph negatives. Default: --seed. "
        "Isolated from eval's global numpy stream; does not read val/test triples.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip Stage-2 training when the variant adapter already exists and only evaluate.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help="Write a HuggingFace checkpoint every N optimizer steps. 0 disables mid-run saves.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Keep this many in-progress checkpoints under trainer_*/.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Explicit trainer checkpoint directory. Default: latest checkpoint-* if present.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing trainer checkpoints and start the stage from scratch.",
    )
    parser.add_argument(
        "--allow_underfit_listed",
        action="store_true",
        help="Bypass the full-task budget gate. Do not use this to "
        "revive the retired 64-QA / 10-step shared-pool smoke.",
    )
    return parser.parse_args()


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def flatten_adapter(adapter_dir: Path) -> None:
    nested = adapter_dir / "mylora"
    if nested.is_dir():
        for item in nested.iterdir():
            item.rename(adapter_dir / item.name)
        nested.rmdir()


def _record_trainer_budget(output_dir: Path, trainer: ListedContrastiveTrainer) -> None:
    """Store the Trainer's actual ``max_steps`` next to the planned budget."""
    path = output_dir / "budget.json"
    if not path.is_file():
        return
    budget = json.loads(path.read_text(encoding="utf-8"))
    actual = int(trainer.state.max_steps)
    budget["trainer_max_steps"] = actual
    budget["trainer_global_step"] = int(trainer.state.global_step)
    if budget.get("total_steps") != actual:
        print(
            f"[budget] planned total_steps={budget.get('total_steps')} "
            f"trainer_max_steps={actual}",
            flush=True,
        )
    path.write_text(json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_adapter(model, tokenizer, adapter_dir: Path, metadata: dict) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    flatten_adapter(adapter_dir)
    (adapter_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def training_arguments(
    output_dir: Path,
    args: argparse.Namespace,
    epochs: float,
    dataset_len: int,
    *,
    gradient_checkpointing: bool = False,
) -> TrainingArguments:
    batch = args.per_device_train_batch_size
    accum = args.gradient_accumulation_steps
    if batch * accum > dataset_len:
        accum = max(1, dataset_len // batch)
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=getattr(args, "adam_epsilon", 1e-8),
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        include_num_input_tokens_seen="all",
        bf16=True,
        tf32=False,
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        lr_scheduler_type="linear",
        gradient_checkpointing=gradient_checkpointing,
    )
    kwargs.update(
        checkpoint_training_kwargs(
            int(getattr(args, "save_steps", 10)),
            int(getattr(args, "save_total_limit", 2)),
        )
    )
    if gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


def prepare_lora_checkpointing(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def load_base_lora(args: argparse.Namespace):
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=args.model_cache_dir,
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
    )
    model = get_lora_model(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    return model, tokenizer


def load_adapter(args: argparse.Namespace, adapter_dir: Path, *, trainable: bool):
    # Load on CPU first. A second 7B `device_map=auto` load after training can
    # offload layers to meta/CPU while PEFT still copies the resized 2GB
    # embedding adapter onto CUDA, which trips NVML in CUDACachingAllocator.
    release_cuda()
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=args.model_cache_dir,
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=trainable)
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def build_context_dataset_from_graph(record: dict, tokenizer) -> TaskDataset:
    context_samples = GenGraphContextTask(
        graph_list=[record["graph"]],
        title_list=[record.get("title", "nell23k")],
        tokenizer=tokenizer,
        context_upsampling=False,
        format_as_instruction=False,
    )()[0]
    return TaskDataset(context_samples=context_samples, qa_samples=[], tokenizer=tokenizer)


def build_context_dataset_from_texts(context_samples: list[str], tokenizer) -> TaskDataset:
    if not context_samples:
        raise ValueError("task file contains no context samples")
    return TaskDataset(context_samples=list(context_samples), qa_samples=[], tokenizer=tokenizer)


def build_qa_assets(record: dict, tokenizer) -> tuple[list[str], list[dict]]:
    title = record.get("title", "nell23k")
    texts: list[str] = []
    metas: list[dict] = []
    eos = tokenizer.eos_token or ""
    questions = record.get("questions") or []
    answers = record.get("answers") or []
    if len(questions) != len(answers):
        raise ValueError("standard graph record questions and answers must have equal length")
    for index, (question, answer) in enumerate(zip(questions, answers)):
        question_text = question[0] if isinstance(question, (list, tuple)) else str(question)
        answer_text = str(answer)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUESTION_TEMPLATE.format(title=title, question=question_text),
            },
            {"role": "assistant", "content": f"<answer>{answer_text}</answer>"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if eos and not text.endswith(eos):
            text += eos
        sample = {
            "question_id": f"{record.get('id', 'graph')}:train:{index}",
            "question": question_text,
            "answer": answer_text,
        }
        listed = [
            relation
            for relation in listed_relations_from_sample(sample)
            if relation != answer_text
        ]
        texts.append(text)
        metas.append(
            {
                "question_id": sample["question_id"],
                "positive_relation": answer_text,
                "listed_relations": listed,
                "prefix_text": format_answer_prefix(
                    tokenizer,
                    title=title,
                    question=question_text,
                    system_prompt=SYSTEM_PROMPT,
                    question_template=QUESTION_TEMPLATE,
                ),
            }
        )
    if not texts:
        raise ValueError("training split is empty")
    return texts, metas


def train_stage1(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    context_samples: list[str] | None = None,
    record: dict | None = None,
) -> Path:
    model, tokenizer = load_base_lora(args)
    if context_samples is not None:
        dataset = build_context_dataset_from_texts(context_samples, tokenizer)
    elif record is not None:
        dataset = build_context_dataset_from_graph(record, tokenizer)
    else:
        raise ValueError("Stage 1 needs context_samples or a graph record")
    use_checkpointing = bool(args.s1_gradient_checkpointing)
    if use_checkpointing:
        prepare_lora_checkpointing(model)
    print(
        f"[s1] context samples: {len(dataset)} model={args.model_name} "
        f"batch={args.per_device_train_batch_size} accum={args.gradient_accumulation_steps} "
        f"checkpointing={use_checkpointing}",
        flush=True,
    )
    training_args = training_arguments(
        output_dir / "trainer_s1",
        args,
        args.num_train_epochs,
        len(dataset),
        gradient_checkpointing=use_checkpointing,
    )
    trainer, model = load_trainer(
        model=model,
        tokenizer=tokenizer,
        training_dataset=dataset,
        training_args=training_args,
        gather_batches=False,
        stop_loss_threshold=args.s1_stop_loss_threshold,
        min_epoch=1,
    )
    tracker = getattr(args, "_swanlab_tracker", None)
    if tracker is not None:
        trainer.add_callback(tracker.callback("s1"))
    started = time.time()
    resume_from = resolve_resume_checkpoint(
        output_dir / "trainer_s1",
        resume_from_checkpoint=args.resume_from_checkpoint,
        no_resume=bool(args.no_resume),
    )
    if resume_from:
        print(f"[s1] resume from {resume_from}", flush=True)
    else:
        print("[s1] start fresh (no checkpoint)", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)
    adapter_dir = output_dir / "s1_adapter"
    save_adapter(
        model,
        tokenizer,
        adapter_dir,
        {
            "stage": "s1",
            "model_name": args.model_name,
            "seconds": round(time.time() - started, 1),
            "context_samples": len(dataset),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_checkpointing": use_checkpointing,
            "resumed_from": resume_from if resume_from else None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del trainer
    del model
    release_cuda()
    print(f"[s1] saved {adapter_dir}", flush=True)
    return adapter_dir


def train_stage2(
    args: argparse.Namespace,
    s1_adapter: Path,
    output_dir: Path,
    *,
    lambda_candidate: float,
    variant: str,
    record: dict | None = None,
    texts: list[str] | None = None,
    metas: list[dict] | None = None,
) -> tuple[Path, object, object]:
    """Train a Stage-2 fork and keep the model in memory for evaluation."""
    if texts is not None and lambda_candidate > 0:
        assert_error_memory_training_budget(
            len(texts),
            listed_negative_source=args.listed_negative_source,
            skip_train=False,
            allow_underfit=bool(getattr(args, "allow_underfit_listed", False)),
            batch=args.per_device_train_batch_size,
            requested_accum=args.gradient_accumulation_steps,
            epochs=args.involve_qa_epochs,
        )
    model, tokenizer = load_adapter(args, s1_adapter, trainable=True)
    if texts is None or metas is None:
        if record is None:
            raise ValueError("Stage 2 needs QA texts or a graph record")
        texts, metas = build_qa_assets(record, tokenizer)
    dataset = ListedQADataset(texts, metas, tokenizer)
    stage_args = copy(args)
    use_checkpointing = lambda_candidate > 0
    if use_checkpointing:
        # Candidate scoring is one padded 10-way forward. Keep the original
        # effective batch; microbatch stays 1 on 24GB Qwen2.5-7B.
        effective = args.per_device_train_batch_size * args.gradient_accumulation_steps
        stage_args.per_device_train_batch_size = 1
        stage_args.gradient_accumulation_steps = max(int(effective), 1)
        prepare_lora_checkpointing(model)
    print(
        f"[{variant}] qa samples: {len(dataset)} listed_mean="
        f"{sum(len(m['listed_relations']) for m in metas) / len(metas):.1f} "
        f"lambda={lambda_candidate} "
        f"per_qa_lambda={sum(1 for meta in metas if meta.get('lambda_candidate') is not None)} "
        f"batch={stage_args.per_device_train_batch_size} "
        f"accum={stage_args.gradient_accumulation_steps}",
        flush=True,
    )
    training_args = training_arguments(
        output_dir / f"trainer_{variant}",
        stage_args,
        args.involve_qa_epochs,
        len(dataset),
        gradient_checkpointing=use_checkpointing,
    )
    # Full QA epoch budget; do not early-stop on the mixed contrastive loss.
    trainer = ListedContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=ListedDataCollator(tokenizer=tokenizer),
        lambda_candidate=lambda_candidate,
        temperature=args.temperature,
        memory_size=args.memory_size,
    )
    tracker = getattr(args, "_swanlab_tracker", None)
    if tracker is not None:
        trainer.add_callback(tracker.callback(variant))
    started = time.time()
    trainer_dir = output_dir / f"trainer_{variant}"
    resume_from = resolve_resume_checkpoint(
        trainer_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        no_resume=bool(args.no_resume),
    )
    if resume_from:
        print(f"[{variant}] resume from {resume_from}", flush=True)
    else:
        print(f"[{variant}] start fresh (no checkpoint)", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)
    _record_trainer_budget(output_dir, trainer)
    adapter_dir = output_dir / variant / "adapter"
    save_adapter(
        model,
        tokenizer,
        adapter_dir,
        {
            "stage": "s2",
            "variant": variant,
            "model_name": args.model_name,
            "lambda_candidate": lambda_candidate,
            "memory_size": args.memory_size,
            "memory_forwards": trainer.memory_forwards,
            "qa_samples": len(dataset),
            "candidate_forwards": trainer.candidate_forwards,
            **{
                f"logged_{key}" if key == "lambda_candidate" else key: trainer.logged_loss_fields[key]
                for key in (
                    "generation_loss",
                    "candidate_loss",
                    "contrastive_candidate_loss",
                    "lambda_candidate",
                )
                if key in trainer.logged_loss_fields
            },
            "last_microbatch_generation_loss": trainer.last_generation_loss,
            "last_microbatch_candidate_loss": trainer.last_candidate_loss,
            "last_microbatch_lambda_candidate": trainer.last_lambda_candidate,
            "trainer_max_steps": int(trainer.state.max_steps),
            "trainer_global_step": int(trainer.state.global_step),
            "per_device_train_batch_size": stage_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_checkpointing": use_checkpointing,
            "seconds": round(time.time() - started, 1),
            "s1_adapter": str(s1_adapter),
            "resumed_from": resume_from if resume_from else None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del trainer
    release_cuda()
    print(f"[{variant}] saved {adapter_dir}", flush=True)
    return adapter_dir, model, tokenizer


def _parse_target(answer: object) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    return [str(value) for value in answer]


def evaluate_adapter(
    model,
    tokenizer,
    record: dict,
    max_new_tokens: int,
    progress_every: int = 0,
    skip_question_ids: set[str] | None = None,
    pred_path: Path | None = None,
) -> list[dict]:
    samples = eval_samples(record)
    skip = skip_question_ids or set()
    dataset = GRIPEvalDataset(
        questions=[item["question"] for item in samples],
        answers=[item["answer"] for item in samples],
        tokenizer=tokenizer,
        graph=record["graph"],
        title=record.get("title", "nell23k"),
        no_graph_context=True,
    )
    device = next(model.parameters()).device
    rows: list[dict] = []
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            qid = str(samples[index].get("question_id") or "")
            if qid and qid in skip:
                continue
            input_ids, question, answer = dataset[index]
            input_ids = input_ids.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            raw = tokenizer.decode(generated[0][input_ids.shape[-1] :], skip_special_tokens=True)
            parsed = parse_model_answer(raw)
            target = _parse_target(answer)
            listed = list(dict.fromkeys(listed_relations_from_sample(samples[index]) + [str(answer)]))
            rows.append(
                {
                    "question_id": samples[index].get("question_id"),
                    "split": samples[index].get("split"),
                    "question": question,
                    "target": target,
                    "raw_response": raw.strip(),
                    "response": parsed,
                    "correct": exact_match(parsed, target),
                    "in_list": parsed in listed,
                    "candidate_valid": parsed in listed,
                }
            )
            if pred_path is not None:
                append_jsonl(pred_path, rows[-1])
            done = index + 1
            if progress_every > 0 and (
                done == 1 or done % progress_every == 0 or done == len(dataset)
            ):
                hits = sum(bool(row["correct"]) for row in rows)
                print(
                    f"[generate] {done}/{len(dataset)} EM={hits / len(rows):.4f}",
                    flush=True,
                )
    return rows


def em_summary(rows: list[dict]) -> dict:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("split", "unknown")), []).append(bool(row["correct"]))
    all_values = [bool(row["correct"]) for row in rows]
    in_list_wrong = [row for row in rows if not row["correct"] and row.get("in_list")]
    oov_wrong = [row for row in rows if not row["correct"] and not row.get("in_list")]
    candidate_valid = [bool(row.get("candidate_valid", row.get("in_list"))) for row in rows]
    summary = {
        "all": {"count": len(all_values), "em": (sum(all_values) / len(all_values)) if all_values else 0.0},
        "candidate_valid_rate": (sum(candidate_valid) / len(candidate_valid)) if candidate_valid else 0.0,
        "candidate_invalid": len(candidate_valid) - sum(candidate_valid),
        "wrong_in_list": len(in_list_wrong),
        "wrong_out_of_list": len(oov_wrong),
    }
    for split, values in sorted(buckets.items()):
        summary[split] = {"count": len(values), "em": sum(values) / len(values) if values else 0.0}
    return summary


def write_comparison(
    output_dir: Path,
    *,
    input_file: Path,
    s1_adapter: Path,
    lambda_candidate: float,
    model_name: str,
) -> dict:
    b1_path = output_dir / "b1" / "summary.json"
    listed_path = output_dir / "listed" / "summary.json"
    if not b1_path.is_file() or not listed_path.is_file():
        raise FileNotFoundError(
            f"need {b1_path} and {listed_path} before comparing Stage-2 forks"
        )
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    listed = json.loads(listed_path.read_text(encoding="utf-8"))
    frozen_path = output_dir / "b1" / "FROZEN_FROM.json"
    b1_source = None
    if frozen_path.is_file():
        b1_source = json.loads(frozen_path.read_text(encoding="utf-8"))
    note = (
        "A method result requires listed EM > B1 on generation. "
        "A listed win here is still a thin GRIP extension, not a new negative-sampling theory."
    )
    if b1_source:
        note = (
            "Listed is this run. B1 is the frozen generation-only adapter "
            f"from {b1_source.get('source', 'another run')} "
            "(lambda=0 never used InfoNCE negatives). "
            "A method result requires listed EM > B1 on generation."
        )
    comparison = {
        "input_file": str(input_file),
        "s1_adapter": str(s1_adapter),
        "model_name": model_name,
        "lambda_candidate": lambda_candidate,
        "b1": b1,
        "listed": listed,
        "listed_minus_b1_em": listed["all"]["em"] - b1["all"]["em"],
        "b1_source": b1_source,
        "note": note,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== listed vs original GRIP ===", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return comparison


def evaluate_variant(
    args: argparse.Namespace,
    record: dict,
    adapter_dir: Path,
    output_dir: Path,
    model=None,
    tokenizer=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions_correct.jsonl"
    samples = eval_samples(record)
    kept, missing = merge_resumed_rows(
        samples, [] if args.no_resume else load_jsonl(pred_path)
    )
    # Canonicalize the file first, removing a possibly truncated last line.
    write_jsonl(pred_path, kept)
    owns_model = model is None
    if owns_model and missing:
        model, tokenizer = load_adapter(args, adapter_dir, trainable=False)
    try:
        new_rows = []
        if missing:
            new_rows = evaluate_adapter(
                model, tokenizer, record, args.gen_max_length,
                skip_question_ids={str(row["question_id"]) for row in kept},
                pred_path=pred_path,
            )
        rows, missing = merge_resumed_rows(samples, kept + new_rows)
        if missing:
            raise RuntimeError(f"evaluation incomplete: {len(missing)} questions missing")
        write_jsonl(pred_path, rows)
        summary = em_summary(rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[eval] {adapter_dir} EM={summary['all']['em']:.4f} n={summary['all']['count']}", flush=True)
        return summary
    finally:
        if owns_model:
            del model
            del tokenizer
            release_cuda()


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    dataset_name = str(getattr(args, "dataset", "") or "").strip()
    if not dataset_name:
        for part in args.input_file.parts:
            if part in {"nell23k", "fb15k237_2", "wn18rr", "clegr_reasoning"}:
                dataset_name = part
                break
    if not dataset_name:
        dataset_name = "unknown"
    payload = {
        "stage": args.stage,
        "dataset": dataset_name,
        "method_name": (
            "structure_counterfactual_contrastive_grip"
            if args.listed_negative_source == "structure_counterfactual"
            else "listed_contrastive_grip"
        ),
        "input_file": str(args.input_file),
        "eval_file": str(args.eval_file) if args.eval_file else None,
        "output_dir": str(output_dir),
        "s1_adapter": str(args.s1_adapter) if args.s1_adapter else None,
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "num_train_epochs": args.num_train_epochs,
        "involve_qa_epochs": args.involve_qa_epochs,
        "s1_stop_loss_threshold": args.s1_stop_loss_threshold,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_epsilon": getattr(args, "adam_epsilon", 1e-8),
        "max_grad_norm": args.max_grad_norm,
        "lambda_candidate": args.lambda_candidate,
        "memory_size": args.memory_size,
        "temperature": args.temperature,
        "gen_max_length": args.gen_max_length,
        "seed": args.seed,
        "swanlab_project": getattr(args, "swanlab_project", ""),
        "swanlab_group": getattr(args, "swanlab_group", ""),
        "swanlab_mode": getattr(args, "swanlab_mode", "online"),
        "s1_gradient_checkpointing": bool(args.s1_gradient_checkpointing),
        "training_arguments": {
            "bf16": True,
            "tf32": False,
            "lr_scheduler_type": "linear",
            "logging_strategy": "steps",
            "logging_steps": 1,
            "remove_unused_columns": False,
            "ddp_find_unused_parameters": False,
        },
        "listed_negative_k": args.listed_negative_k,
        "listed_negative_source": args.listed_negative_source,
        "negative_sampling_mode": getattr(args, "negative_sampling_mode", "random"),
        "error_memory_uniform_mix": float(getattr(args, "error_memory_uniform_mix", 1.0 / 3.0)),
        "error_memory_weight": float(getattr(args, "error_memory_weight", 1.0)),
        "error_memory_violation_weight": float(
            getattr(args, "error_memory_violation_weight", 1.0)
        ),
        "error_memory_temperature": float(getattr(args, "error_memory_temperature", 0.25)),
        "listed_negative_seed": (
            args.listed_negative_seed if args.listed_negative_seed is not None else args.seed
        ),
        "error_memory_manifest": (
            str(args.error_memory_manifest) if args.error_memory_manifest else None
        ),
        "counterfactual_manifest": (
            str(args.counterfactual_manifest) if args.counterfactual_manifest else None
        ),
        "counterfactual_metadata": (
            str(args.counterfactual_metadata) if args.counterfactual_metadata else None
        ),
        "counterfactual_sampling": {
            "uniform_mix": float(
                getattr(args, "counterfactual_uniform_mix", 1.0 / 3.0)
            ),
            "structure_weight": float(
                getattr(args, "counterfactual_structure_weight", 1.0)
            ),
            "violation_weight": float(
                getattr(args, "counterfactual_violation_weight", 1.0)
            ),
            "temperature": float(
                getattr(args, "counterfactual_temperature", 0.25)
            ),
        },
        "raw_dir": str(args.raw_dir),
        "skip_train": bool(args.skip_train),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "no_resume": bool(args.no_resume),
        "allow_underfit_listed": bool(args.allow_underfit_listed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Stage 1 is shared graph-context storage. Stage 2 then forks: "
            "b1 = original GRIP generation loss; listed = generation + InfoNCE. "
            "A paper task JSON trains S1 on context+summarization and S2 on generated QA; "
            "relation-like QA items sample 9 negatives from the official train-graph "
            "relation vocabulary with the process.py permutation rule "
            "(--listed_negative_source train_graph), error_memory accepts "
            "a frozen train-only GRIP-conditioned error-memory bank, and "
            "structure_counterfactual accepts a frozen train-only bank scored "
            "by relation-profile similarity plus B1 score violation. These "
            "modes only change negative identities. "
            "Prompts stay unchanged. Val/test EM uses the standard processed GRIP "
            "graph record. Both Stage-2 arms use the full QA "
            "epoch budget (no S2 early stop) so the generation objective is compute-matched."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_training_assets(args: argparse.Namespace) -> tuple[dict | None, list[str] | None, list[str] | None, list[dict] | None, dict]:
    payload = load_json_payload(args.input_file)
    if is_grip_task_file(payload):
        if args.eval_file is None:
            raise ValueError(
                "--eval_file is required when --input_file is a GRIP task JSON; "
                "pass a standard processed graph record"
            )
        eval_path = args.eval_file
        eval_record = load_graph_record(eval_path)
        args.eval_file = eval_path
        context_samples = list(payload["context_samples"])
        relation_order = None
        error_memory_manifest = None
        counterfactual_manifest = None
        if args.listed_negative_source in {
            "train_graph",
            "error_memory",
            "structure_counterfactual",
        }:
            raw_dir = Path(args.raw_dir)
            if not (raw_dir / "train.txt").is_file():
                raise FileNotFoundError(f"missing NELL23K train.txt under {raw_dir}")
            relation_order = load_train_relation_order(
                raw_dir, relation_text_file=args.relation_text_file
            )
        if args.listed_negative_source == "error_memory":
            if args.error_memory_manifest is None:
                raise ValueError(
                    f"{args.listed_negative_source} requires --error_memory_manifest"
                )
            error_memory_manifest = load_error_memory_manifest(args.error_memory_manifest)
        if args.listed_negative_source == "structure_counterfactual":
            if args.counterfactual_manifest is None:
                raise ValueError(
                    "structure_counterfactual requires --counterfactual_manifest"
                )
            counterfactual_manifest = load_counterfactual_manifest(
                args.counterfactual_manifest
            )
        if args.listed_negative_source == "error_memory":
            if args.negative_sampling_mode != "error_memory":
                raise ValueError(
                    "listed_negative_source=error_memory requires "
                    "--negative_sampling_mode error_memory"
                )
            if args.listed_negative_k != LISTED_NEGATIVE_K:
                raise ValueError("error-memory sampling requires listed_negative_k=9")
            validate_error_memory_manifest(
                error_memory_manifest or {},
                k=args.listed_negative_k,
            )
        if args.listed_negative_source == "structure_counterfactual":
            if args.negative_sampling_mode != "structure_counterfactual":
                raise ValueError(
                    "listed_negative_source=structure_counterfactual requires "
                    "--negative_sampling_mode structure_counterfactual"
                )
            if args.listed_negative_k != LISTED_NEGATIVE_K:
                raise ValueError(
                    "structure-counterfactual sampling requires listed_negative_k=9"
                )
            validate_structure_counterfactual_manifest(
                counterfactual_manifest or {},
                k=args.listed_negative_k,
            )
        elif args.negative_sampling_mode != "random":
            raise ValueError(
                "non-random negative_sampling_mode requires a matching "
                "listed_negative_source"
            )
        listed_seed = args.listed_negative_seed if args.listed_negative_seed is not None else args.seed
        qa_samples = list(payload["qa_samples"])
        question_ids = original_question_ids_from_payload(payload, len(qa_samples))
        qa_texts, qa_metas = build_qa_assets_from_task_texts(
            qa_samples,
            seed=listed_seed,
            listed_negative_k=args.listed_negative_k,
            listed_negative_source=args.listed_negative_source,
            relation_order=relation_order,
            error_memory_manifest=error_memory_manifest,
            counterfactual_manifest=counterfactual_manifest,
            question_ids=question_ids,
        )
        listed_n = sum(1 for meta in qa_metas if meta["listed_relations"])
        relation_aliases = train_relation_alias_index(relation_order or [])
        relation_n = sum(
            1 for text in qa_texts
            if is_relation_gold(assistant_gold(text), text, relation_alias_index=relation_aliases)
        )
        skipped_n = relation_n - listed_n
        exact_n = sum(
            1
            for meta in qa_metas
            if meta["listed_relations"]
            and meta.get("matched_train_relation") == meta.get("positive_relation")
        )
        alias_n = listed_n - exact_n
        vocab_n = len(relation_order) if relation_order is not None else len(
            {meta["positive_relation"] for meta in qa_metas if meta["listed_relations"]}
        )
        print(
            f"[data] paper task file context={len(context_samples)} "
            f"qa={len(qa_texts)} relation_qa={relation_n} "
            f"listed_relation_qa={listed_n} skipped_unmatched={skipped_n} "
            f"exact_match={exact_n} alias_match={alias_n} "
            f"listed_negative_source={args.listed_negative_source} "
            f"vocab={vocab_n} listed_negative_seed={listed_seed} "
            f"question_ids={'original' if question_ids is not None else 'positional'} "
            f"eval={eval_path}",
            flush=True,
        )
        return None, context_samples, qa_texts, qa_metas, eval_record
    eval_path = args.eval_file or args.input_file
    eval_record = load_graph_record(eval_path)
    args.eval_file = eval_path
    record = load_graph_record(args.input_file)
    return record, None, None, None, eval_record


def _run_main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    s1_adapter = Path(args.s1_adapter) if args.s1_adapter else (output_dir / "s1_adapter")

    if args.stage == "compare":
        if not (output_dir / "run_config.json").is_file():
            write_run_config(output_dir, args)
        write_comparison(
            output_dir,
            input_file=args.input_file,
            s1_adapter=s1_adapter,
            lambda_candidate=args.lambda_candidate,
            model_name=args.model_name,
        )
        return

    set_random_seed(args.seed)
    record, context_samples, qa_texts, qa_metas, eval_record = resolve_training_assets(args)
    write_run_config(output_dir, args)
    tracker = getattr(args, "_swanlab_tracker", None)
    if tracker is not None:
        tracker.start(json.loads((output_dir / "run_config.json").read_text(encoding="utf-8")))

    if qa_texts is not None:
        listed_adapter = output_dir / "listed" / "adapter"
        will_train_listed = (
            args.stage in {"listed", "all"}
            and not args.skip_train
            and (args.no_resume or not adapter_is_complete(listed_adapter))
        )
        budget = assert_error_memory_training_budget(
            len(qa_texts),
            listed_negative_source=args.listed_negative_source,
            skip_train=not will_train_listed,
            allow_underfit=bool(args.allow_underfit_listed),
            batch=args.per_device_train_batch_size,
            requested_accum=args.gradient_accumulation_steps,
            epochs=args.involve_qa_epochs,
        )
        if budget is not None:
            # Planned step count only. train_listed overwrites trainer_max_steps
            # with the Trainer value after the optimizer schedule exists.
            budget["trainer_max_steps"] = None
            (output_dir / "budget.json").write_text(
                json.dumps(budget, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[budget] listed n={budget['n_samples']} "
                f"accum={budget['effective_accum']} "
                f"steps={budget['total_steps']}",
                flush=True,
            )

    if args.stage in {"s1", "all"}:
        if adapter_is_complete(Path(s1_adapter)) and not args.no_resume:
            print(f"[s1] reuse existing {s1_adapter}", flush=True)
        else:
            s1_adapter = train_stage1(
                args,
                output_dir,
                context_samples=context_samples,
                record=record,
            )

    summaries = {}
    if args.stage in {"b1", "all"}:
        if not Path(s1_adapter).is_dir():
            raise FileNotFoundError(f"missing stage-1 adapter: {s1_adapter}")
        b1_adapter = output_dir / "b1" / "adapter"
        model = tokenizer = None
        if args.skip_train or (adapter_is_complete(b1_adapter) and not args.no_resume):
            if not adapter_is_complete(b1_adapter):
                raise FileNotFoundError(f"missing b1 adapter: {b1_adapter}")
            print(f"[b1] skip train; evaluate {b1_adapter}", flush=True)
        else:
            b1_adapter, model, tokenizer = train_stage2(
                args,
                Path(s1_adapter),
                output_dir,
                lambda_candidate=0.0,
                variant="b1",
                record=record,
                texts=qa_texts,
                metas=qa_metas,
            )
        summaries["b1"] = evaluate_variant(
            args, eval_record, b1_adapter, output_dir / "b1", model=model, tokenizer=tokenizer
        )
        if tracker is not None:
            tracker.log_evaluation("b1", summaries["b1"])
        del model, tokenizer
        release_cuda()

    if args.stage in {"listed", "all"}:
        if not Path(s1_adapter).is_dir():
            raise FileNotFoundError(f"missing stage-1 adapter: {s1_adapter}")
        listed_adapter = output_dir / "listed" / "adapter"
        model = tokenizer = None
        if args.skip_train or (adapter_is_complete(listed_adapter) and not args.no_resume):
            if not adapter_is_complete(listed_adapter):
                raise FileNotFoundError(f"missing listed adapter: {listed_adapter}")
            print(f"[listed] skip train; evaluate {listed_adapter}", flush=True)
        else:
            listed_adapter, model, tokenizer = train_stage2(
                args,
                Path(s1_adapter),
                output_dir,
                lambda_candidate=args.lambda_candidate,
                variant="listed",
                record=record,
                texts=qa_texts,
                metas=qa_metas,
            )
        summaries["listed"] = evaluate_variant(
            args,
            eval_record,
            listed_adapter,
            output_dir / "listed",
            model=model,
            tokenizer=tokenizer,
        )
        if tracker is not None:
            tracker.log_evaluation("listed", summaries["listed"])
        del model, tokenizer
        release_cuda()

    b1_ready = (output_dir / "b1" / "summary.json").is_file()
    listed_ready = (output_dir / "listed" / "summary.json").is_file()
    if b1_ready and listed_ready:
        write_comparison(
            output_dir,
            input_file=args.input_file,
            s1_adapter=s1_adapter,
            lambda_candidate=args.lambda_candidate,
            model_name=args.model_name,
        )
    if tracker is not None:
        tracker.complete(stage=args.stage, summaries=summaries)


def main() -> None:
    args = parse_args()
    tracker = None
    if args.swanlab_project and args.stage != "compare":
        tracker = SwanLabTracker(
            args.output_dir,
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name or args.output_dir.name,
            group=args.swanlab_group or args.output_dir.name,
            mode=args.swanlab_mode,
        )
    args._swanlab_tracker = tracker
    try:
        _run_main(args)
    except BaseException as error:
        if tracker is not None:
            tracker.fail(error)
        raise
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
