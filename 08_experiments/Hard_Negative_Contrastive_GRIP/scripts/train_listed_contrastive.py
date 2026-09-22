"""Train original GRIP vs GRIP + listed contrastive.

Default input is the aligned NELL23K relation-prediction split. Passing
``grip_nell23k_tasks.json`` trains Stage 1 on paper context+summarization and
Stage 2 on generated QA; relation-like items sample 9 train-graph negatives
with the official ``process.py`` rule. Pass ``--listed_negative_source embed_sim``
to keep that 198-relation pool but prefer Stage-1 cosine-similar relations.
Pass ``rollout_hard`` to use one actual frozen-B1 free-generation error plus
uniform train-graph fallbacks. ``--listed_negative_source qa_vocab`` restores
 the older 370-relation QA-gold pool. Val/test EM always uses an aligned graph record (``--eval_file``).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import TrainingArguments

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, SYSTEM_PROMPT, TORCH_DTYPE  # noqa: E402
from evaluation.recurrent_metrics import exact_match, parse_recurrent_answer  # noqa: E402
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset  # noqa: E402
from grip.tasks.recurrent_tasks.task_dataset import (  # noqa: E402
    QUESTION_TEMPLATE,
    format_recurrent_qa,
)
from grip.tasks.train_tasks.gen_context import GenGraphContextTask  # noqa: E402
from grip.tasks.train_tasks.task_dataset import TaskDataset  # noqa: E402
from grip.training.train import load_trainer  # noqa: E402
from hard_negative_grip.decode_io import append_jsonl  # noqa: E402
from hard_negative_grip.listed_training import (  # noqa: E402
    ListedContrastiveTrainer,
    ListedDataCollator,
    ListedQADataset,
    format_answer_prefix,
    listed_negatives,
)
from hard_negative_grip.embed_negatives import (  # noqa: E402
    DEFAULT_EMBED_POOL_SIZE,
    DEFAULT_EMBED_TEMPERATURE,
    align_embeddings,
    load_relation_embeddings,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    load_train_relation_order,
)
from hard_negative_grip.run_protection import (  # noqa: E402
    adapter_is_complete,
    checkpoint_training_kwargs,
    resolve_resume_checkpoint,
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
    load_score_hard_manifest,
)
from models.utils import get_hf_llm_tokenizer, get_lora_model  # noqa: E402
from utils import set_random_seed  # noqa: E402

ALIGNED_SMOKE = HNG / "data/nell23k/recurrent_relation_prediction.aligned.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=ALIGNED_SMOKE)
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=None,
        help="Aligned relation-prediction record for val/test EM. "
        "Required in practice when --input_file is a paper task JSON.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["s1", "b1", "listed", "all", "compare"],
        default="all",
        help="s1 = shared graph storage; b1/listed = Stage-2 forks; compare = merge EM summaries",
    )
    parser.add_argument("--s1_adapter", type=Path, default=None)
    parser.add_argument("--model_name", default="qwen-0.5b")
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
        help="train_graph = official 198 NELL23K train relations via process.py; "
        "embed_sim = same 198-relation pool, sampled by Stage-1 cosine similarity; "
        "qa_vocab = the older 370-relation QA-gold pool.",
    )
    parser.add_argument(
        "--relation_embedding_file",
        type=Path,
        default=None,
        help="NPZ from scripts/precompute_relation_embeddings.py. Required for embed_sim.",
    )
    parser.add_argument(
        "--score_hard_manifest",
        type=Path,
        default=None,
        help="Immutable JSONL manifest from mine_score_hard_negatives.py; required for score_hard.",
    )
    parser.add_argument(
        "--embed_pool_size",
        type=int,
        default=DEFAULT_EMBED_POOL_SIZE,
        help="embed_sim: only sample from the top-N most similar train relations.",
    )
    parser.add_argument(
        "--embed_sample_temperature",
        type=float,
        default=DEFAULT_EMBED_TEMPERATURE,
        help="embed_sim: softmax temperature over cosine scores inside the pool.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=DEFAULT_RAW_NELL23K,
        help="NELL23K raw split directory (train.txt) for train_graph negatives.",
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
        adam_epsilon=1e-8,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
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
    for sample in record["recurrent_questions"]:
        if sample.get("split") != "train":
            continue
        text = format_recurrent_qa(tokenizer, title, sample["question"], sample["answer"])
        if eos and not text.endswith(eos):
            text += eos
        texts.append(text)
        metas.append(
            {
                "question_id": sample.get("question_id"),
                "positive_relation": sample["answer"],
                "listed_relations": listed_negatives(sample),
                "prefix_text": format_answer_prefix(
                    tokenizer,
                    title=title,
                    question=sample["question"],
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
        f"lambda={lambda_candidate} batch={stage_args.per_device_train_batch_size} "
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
            "last_generation_loss": trainer.last_generation_loss,
            "last_candidate_loss": trainer.last_candidate_loss,
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
    samples = [
        item for item in record["recurrent_questions"] if item["split"] in {"validation", "test"}
    ]
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
            parsed = parse_recurrent_answer(raw)
            target = _parse_target(answer)
            listed = listed_negatives(samples[index]) + [samples[index]["answer"]]
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
    summary = {
        "all": {"count": len(all_values), "em": (sum(all_values) / len(all_values)) if all_values else 0.0},
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
    owns_model = model is None
    if owns_model:
        model, tokenizer = load_adapter(args, adapter_dir, trainable=False)
    try:
        rows = evaluate_adapter(model, tokenizer, record, args.gen_max_length)
        summary = em_summary(rows)
        pred_path = output_dir / "predictions_correct.jsonl"
        with pred_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    payload = {
        "stage": args.stage,
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
        "lambda_candidate": args.lambda_candidate,
        "memory_size": args.memory_size,
        "temperature": args.temperature,
        "seed": args.seed,
        "s1_gradient_checkpointing": bool(args.s1_gradient_checkpointing),
        "listed_negative_k": args.listed_negative_k,
        "listed_negative_source": args.listed_negative_source,
        "listed_negative_seed": (
            args.listed_negative_seed if args.listed_negative_seed is not None else args.seed
        ),
        "relation_embedding_file": (
            str(args.relation_embedding_file) if args.relation_embedding_file else None
        ),
        "score_hard_manifest": (
            str(args.score_hard_manifest) if args.score_hard_manifest else None
        ),
        "embed_pool_size": int(args.embed_pool_size),
        "embed_sample_temperature": float(args.embed_sample_temperature),
        "raw_dir": str(args.raw_dir),
        "skip_train": bool(args.skip_train),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "no_resume": bool(args.no_resume),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Stage 1 is shared graph-context storage. Stage 2 then forks: "
            "b1 = original GRIP generation loss; listed = generation + InfoNCE. "
            "A paper task JSON trains S1 on context+summarization and S2 on generated QA; "
            "relation-like QA items sample 9 negatives from the official train-graph "
            "relation vocabulary with the process.py permutation rule "
            "(--listed_negative_source train_graph). embed_sim keeps that vocabulary "
            "but prefers Stage-1 cosine-similar relations. qa_vocab restores the older "
            "370-relation QA-gold pool. Prompts stay unchanged. Val/test EM still uses "
            "the aligned relation-prediction split. Both Stage-2 arms use the full QA "
            "epoch budget (no S2 early stop) so the generation objective is compute-matched."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_training_assets(args: argparse.Namespace) -> tuple[dict | None, list[str] | None, list[str] | None, list[dict] | None, dict]:
    payload = load_json_payload(args.input_file)
    if is_grip_task_file(payload):
        eval_path = args.eval_file or ALIGNED_SMOKE
        eval_record = load_graph_record(eval_path)
        args.eval_file = eval_path
        context_samples = list(payload["context_samples"])
        relation_order = None
        relation_embeddings = None
        score_hard_manifest = None
        if args.listed_negative_source in {"train_graph", "embed_sim", "score_hard", "rollout_hard"}:
            raw_dir = Path(args.raw_dir)
            if not (raw_dir / "train.txt").is_file():
                raise FileNotFoundError(f"missing NELL23K train.txt under {raw_dir}")
            relation_order = load_train_relation_order(raw_dir)
        if args.listed_negative_source in {"score_hard", "rollout_hard"}:
            if args.score_hard_manifest is None:
                raise ValueError(
                    f"{args.listed_negative_source} requires --score_hard_manifest"
                )
            score_hard_manifest = load_score_hard_manifest(args.score_hard_manifest)
        if args.listed_negative_source == "embed_sim":
            if args.relation_embedding_file is None:
                raise ValueError("embed_sim requires --relation_embedding_file")
            stored_relations, stored_embeddings = load_relation_embeddings(
                Path(args.relation_embedding_file)
            )
            relation_embeddings = align_embeddings(
                relation_order or [], stored_relations, stored_embeddings
            )
        listed_seed = args.listed_negative_seed if args.listed_negative_seed is not None else args.seed
        qa_texts, qa_metas = build_qa_assets_from_task_texts(
            list(payload["qa_samples"]),
            seed=listed_seed,
            listed_negative_k=args.listed_negative_k,
            listed_negative_source=args.listed_negative_source,
            relation_order=relation_order,
            relation_embeddings=relation_embeddings,
            embed_pool_size=args.embed_pool_size,
            embed_sample_temperature=args.embed_sample_temperature,
            score_hard_manifest=score_hard_manifest,
        )
        listed_n = sum(1 for meta in qa_metas if meta["listed_relations"])
        relation_n = sum(
            1 for text in qa_texts if is_relation_gold(assistant_gold(text), text)
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
            f"embed_file={args.relation_embedding_file} "
            f"embed_pool={args.embed_pool_size} embed_tau={args.embed_sample_temperature} "
            f"eval={eval_path}",
            flush=True,
        )
        return None, context_samples, qa_texts, qa_metas, eval_record
    eval_path = args.eval_file or args.input_file
    eval_record = load_graph_record(eval_path)
    args.eval_file = eval_path
    record = load_graph_record(args.input_file)
    return record, None, None, None, eval_record


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
