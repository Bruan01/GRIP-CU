"""Train original GRIP vs GRIP + listed 10-way contrastive on aligned NELL23K.

Stage 1 (graph context) is shared. Stage 2 then forks:

- ``b1``: generation loss only (Original GRIP)
- ``listed``: generation + InfoNCE over the prompt's 9 distractors

Evaluation is greedy generation exact match on validation/test. This is the
method gate, not a paper result by itself.
"""

from __future__ import annotations

import argparse
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
from hard_negative_grip.listed_training import (  # noqa: E402
    ListedContrastiveTrainer,
    ListedDataCollator,
    ListedQADataset,
    format_answer_prefix,
    listed_negatives,
)
from models.utils import get_hf_llm_tokenizer, get_lora_model  # noqa: E402
from utils import load_list_json, set_random_seed  # noqa: E402

ALIGNED_SMOKE = HNG / "data/nell23k/recurrent_relation_prediction.aligned.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=ALIGNED_SMOKE)
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


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
        overwrite_output_dir=True,
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
        save_strategy="no",
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
    if gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


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
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=args.model_cache_dir,
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=trainable)
    return model, tokenizer


def build_context_dataset(record: dict, tokenizer) -> TaskDataset:
    context_samples = GenGraphContextTask(
        graph_list=[record["graph"]],
        title_list=[record.get("title", "nell23k")],
        tokenizer=tokenizer,
        context_upsampling=False,
        format_as_instruction=False,
    )()[0]
    return TaskDataset(context_samples=context_samples, qa_samples=[], tokenizer=tokenizer)


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


def train_stage1(args: argparse.Namespace, record: dict, output_dir: Path) -> Path:
    model, tokenizer = load_base_lora(args)
    dataset = build_context_dataset(record, tokenizer)
    print(f"[s1] context samples: {len(dataset)}", flush=True)
    training_args = training_arguments(output_dir / "trainer_s1", args, args.num_train_epochs, len(dataset))
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
    trainer.train()
    adapter_dir = output_dir / "s1_adapter"
    save_adapter(
        model,
        tokenizer,
        adapter_dir,
        {
            "stage": "s1",
            "seconds": round(time.time() - started, 1),
            "context_samples": len(dataset),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del model
    torch.cuda.empty_cache()
    print(f"[s1] saved {adapter_dir}", flush=True)
    return adapter_dir


def train_stage2(
    args: argparse.Namespace,
    record: dict,
    s1_adapter: Path,
    output_dir: Path,
    *,
    lambda_candidate: float,
    variant: str,
) -> Path:
    model, tokenizer = load_adapter(args, s1_adapter, trainable=True)
    texts, metas = build_qa_assets(record, tokenizer)
    dataset = ListedQADataset(texts, metas, tokenizer)
    stage_args = copy(args)
    use_checkpointing = lambda_candidate > 0
    if use_checkpointing:
        # 10-way continuation logits are vocab-sized; keep one QA example and
        # one candidate sequence in memory, but match B1's effective batch.
        effective = args.per_device_train_batch_size * args.gradient_accumulation_steps
        stage_args.per_device_train_batch_size = 1
        stage_args.gradient_accumulation_steps = max(int(effective), 1)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
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
    )
    started = time.time()
    trainer.train()
    adapter_dir = output_dir / variant / "adapter"
    save_adapter(
        model,
        tokenizer,
        adapter_dir,
        {
            "stage": "s2",
            "variant": variant,
            "lambda_candidate": lambda_candidate,
            "qa_samples": len(dataset),
            "candidate_forwards": trainer.candidate_forwards,
            "last_generation_loss": trainer.last_generation_loss,
            "last_candidate_loss": trainer.last_candidate_loss,
            "per_device_train_batch_size": stage_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_checkpointing": use_checkpointing,
            "seconds": round(time.time() - started, 1),
            "s1_adapter": str(s1_adapter),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del model
    torch.cuda.empty_cache()
    print(f"[{variant}] saved {adapter_dir}", flush=True)
    return adapter_dir


def _parse_target(answer: object) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    return [str(value) for value in answer]


def evaluate_adapter(model, tokenizer, record: dict, max_new_tokens: int) -> list[dict]:
    samples = [
        item for item in record["recurrent_questions"] if item["split"] in {"validation", "test"}
    ]
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
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
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
) -> dict:
    b1_path = output_dir / "b1" / "summary.json"
    listed_path = output_dir / "listed" / "summary.json"
    if not b1_path.is_file() or not listed_path.is_file():
        raise FileNotFoundError(
            f"need {b1_path} and {listed_path} before comparing Stage-2 forks"
        )
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    listed = json.loads(listed_path.read_text(encoding="utf-8"))
    comparison = {
        "input_file": str(input_file),
        "s1_adapter": str(s1_adapter),
        "lambda_candidate": lambda_candidate,
        "b1": b1,
        "listed": listed,
        "listed_minus_b1_em": listed["all"]["em"] - b1["all"]["em"],
        "note": (
            "A method result requires listed EM > B1 on generation. "
            "A listed win here is still a thin GRIP extension, not a new negative-sampling theory."
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== listed vs original GRIP ===", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return comparison


def evaluate_variant(args: argparse.Namespace, record: dict, adapter_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_adapter(args, adapter_dir, trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = evaluate_adapter(model, tokenizer, record, args.gen_max_length)
    summary = em_summary(rows)
    pred_path = output_dir / "predictions_correct.jsonl"
    with pred_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    print(f"[eval] {adapter_dir} EM={summary['all']['em']:.4f} n={summary['all']['count']}", flush=True)
    return summary


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "stage": args.stage,
        "input_file": str(args.input_file),
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
        "temperature": args.temperature,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Stage 1 is shared graph-context storage. Stage 2 then forks: "
            "b1 = original GRIP generation loss; listed = generation + 10-way InfoNCE. "
            "Both Stage-2 arms use the full QA epoch budget (no S2 early stop) so the "
            "generation objective is compute-matched."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(output_dir, args)
    s1_adapter = Path(args.s1_adapter) if args.s1_adapter else (output_dir / "s1_adapter")

    if args.stage == "compare":
        write_comparison(
            output_dir,
            input_file=args.input_file,
            s1_adapter=s1_adapter,
            lambda_candidate=args.lambda_candidate,
        )
        return

    set_random_seed(args.seed)
    records = load_list_json(str(args.input_file))
    if not records:
        raise ValueError(f"no records in {args.input_file}")
    record = records[0]

    if args.stage in {"s1", "all"}:
        s1_adapter = train_stage1(args, record, output_dir)

    summaries = {}
    if args.stage in {"b1", "all"}:
        if not Path(s1_adapter).is_dir():
            raise FileNotFoundError(f"missing stage-1 adapter: {s1_adapter}")
        b1_adapter = train_stage2(
            args, record, Path(s1_adapter), output_dir, lambda_candidate=0.0, variant="b1"
        )
        summaries["b1"] = evaluate_variant(args, record, b1_adapter, output_dir / "b1")

    if args.stage in {"listed", "all"}:
        if not Path(s1_adapter).is_dir():
            raise FileNotFoundError(f"missing stage-1 adapter: {s1_adapter}")
        listed_adapter = train_stage2(
            args,
            record,
            Path(s1_adapter),
            output_dir,
            lambda_candidate=args.lambda_candidate,
            variant="listed",
        )
        summaries["listed"] = evaluate_variant(args, record, listed_adapter, output_dir / "listed")

    b1_ready = (output_dir / "b1" / "summary.json").is_file()
    listed_ready = (output_dir / "listed" / "summary.json").is_file()
    if b1_ready and listed_ready:
        write_comparison(
            output_dir,
            input_file=args.input_file,
            s1_adapter=s1_adapter,
            lambda_candidate=args.lambda_candidate,
        )


if __name__ == "__main__":
    main()
