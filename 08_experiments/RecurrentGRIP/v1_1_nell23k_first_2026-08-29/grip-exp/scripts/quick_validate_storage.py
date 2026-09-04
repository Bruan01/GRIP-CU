"""Storage-only validation for the 0.5B quick check.

This script answers a single question: with the paper's GRIP recipe (LoRA on the
MLP modules, full layers, two-stage training), does the trained graph adapter
outperform the *same* base model without any adapter on NELL23K relation
prediction?

It deliberately avoids the recurrent-execution machinery (single-layer executor,
``layers_to_transform``), which the failed pilot used and which is incompatible
with the paper's "store the graph in the MLP across all layers" design.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from evaluation.recurrent_metrics import exact_match, parse_recurrent_answer
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset
from grip.tasks.recurrent_tasks import build_recurrent_task_dataset
from grip.tasks.train_tasks.gen_context import GenGraphContextTask
from grip.training import train
from models.utils import get_hf_llm_tokenizer, get_lora_model
from utils import load_list_json, set_random_seed


def _parse_target(answer: object) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    return [str(value) for value in answer]


def subset_graph(graph: dict, max_edges: Optional[int]) -> dict:
    """Optionally shrink the graph to the first ``max_edges`` edges (smoke tests).

    Node indices are remapped to a compact range so that both the node and edge
    counts shrink together.
    """
    if not max_edges or max_edges <= 0:
        return graph
    edge_list = graph["edge_list"][:max_edges]
    edge_index = graph["edge_index"][:max_edges]
    if not edge_index:
        return {**graph, "node_list": [], "edge_list": edge_list, "edge_index": edge_index}
    node_set = sorted({node for src, tgt in edge_index for node in (src, tgt)})
    node_map = {old: new for new, old in enumerate(node_set)}
    new_edge_index = [[node_map[src], node_map[tgt]] for src, tgt in edge_index]
    new_node_list = [graph["node_list"][old] for old in node_set]
    return {
        **graph,
        "node_list": new_node_list,
        "edge_list": edge_list,
        "edge_index": new_edge_index,
    }


def build_context_samples(record: dict, tokenizer, context_upsampling: bool) -> list[str]:
    return GenGraphContextTask(
        graph_list=[record["graph"]],
        title_list=[record.get("title", "nell23k")],
        tokenizer=tokenizer,
        context_upsampling=context_upsampling,
        format_as_instruction=False,
    )()[0]


def evaluate_adapter(
    model,
    tokenizer,
    record: dict,
    max_new_tokens: int,
    max_eval_questions: Optional[int],
) -> list[dict]:
    samples = [
        item for item in record["recurrent_questions"]
        if item["split"] in {"validation", "test"}
    ]
    if max_eval_questions and max_eval_questions > 0:
        samples = samples[:max_eval_questions]

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
            attention_mask = torch.ones_like(input_ids)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            raw = tokenizer.decode(
                generated[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            parsed = parse_recurrent_answer(raw)
            target = _parse_target(answer)
            rows.append(
                {
                    "question_id": samples[index].get("question_id"),
                    "split": samples[index].get("split"),
                    "question": question,
                    "target": target,
                    "raw_response": raw.strip(),
                    "response": parsed,
                    "correct": exact_match(parsed, target),
                }
            )
    return rows


def _em_summary(rows: list[dict]) -> dict:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("split", "unknown")), []).append(bool(row["correct"]))
    all_values = [bool(row["correct"]) for row in rows]
    summary = {
        "all": {"count": len(all_values), "em": (sum(all_values) / len(all_values)) if all_values else 0.0},
    }
    for split, values in sorted(buckets.items()):
        summary[split] = {"count": len(values), "em": sum(values) / len(values) if values else 0.0}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="qwen-0.5b")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--target_modules", nargs="+", default=["down_proj", "up_proj", "gate_proj"])
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--involve_qa_epochs", type=float, default=10)
    parser.add_argument("--s1_stop_loss_threshold", type=float, default=0.15)
    parser.add_argument("--s2_stop_loss_threshold", type=float, default=0.15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--context_upsampling", action="store_true")
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--max_context_edges", type=int, default=0)
    parser.add_argument("--max_eval_questions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--adapter_dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    records = load_list_json(args.input_file)
    if not records:
        raise ValueError("No records loaded from input file")
    record = records[0]
    record["graph"] = subset_graph(record["graph"], args.max_context_edges)

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

    context_samples = build_context_samples(record, tokenizer, args.context_upsampling)
    dataset = build_recurrent_task_dataset(
        tokenizer=tokenizer,
        title=record.get("title", "nell23k"),
        context_samples=context_samples,
        samples=record["recurrent_questions"],
    )

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    print(f"Trainable LoRA modules: {len(trainable)}")
    print(f"Target modules: {args.target_modules}")
    print(f"Context samples: {len(context_samples)}")

    if not args.skip_train:
        training_args = TrainingArguments(
            output_dir=str(output_dir / "trainer"),
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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
            remove_unused_columns=True,
            lr_scheduler_type="linear",
        )
        model, tokenizer = train(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            training_dataset=dataset,
            involve_qa_epochs=args.involve_qa_epochs,
            gather_batches=False,
            s1_stop_loss_threshold=args.s1_stop_loss_threshold,
            s2_stop_loss_threshold=args.s2_stop_loss_threshold,
            s1_min_epoch=1,
            s2_min_epoch=1,
        )

        adapter_dir = Path(args.adapter_dir) if args.adapter_dir else output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        # Flatten the PEFT "mylora" subdirectory, mirroring train_nell23k_lora.py.
        mylora_dir = adapter_dir / "mylora"
        if mylora_dir.is_dir():
            for item in mylora_dir.iterdir():
                item.rename(adapter_dir / item.name)
            mylora_dir.rmdir()
        print(f"Adapter saved to {adapter_dir.resolve()}")

    if model.device.type == "cpu":
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # "correct adapter" vs "no adapter" on the same base model.
    correct_rows = evaluate_adapter(model, tokenizer, record, args.gen_max_length, args.max_eval_questions)
    with model.disable_adapter():
        none_rows = evaluate_adapter(model, tokenizer, record, args.gen_max_length, args.max_eval_questions)

    correct_summary = _em_summary(correct_rows)
    none_summary = _em_summary(none_rows)

    result = {
        "model": args.model_name,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": args.target_modules},
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "involve_qa_epochs": args.involve_qa_epochs,
            "context_samples": len(context_samples),
            "learning_rate": args.learning_rate,
            "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        },
        "correct_adapter": correct_summary,
        "no_adapter": none_summary,
        "correct_beats_none": correct_summary["all"]["em"] > none_summary["all"]["em"],
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_rows(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_rows(output_dir / "predictions_correct.jsonl", correct_rows)
    _write_rows(output_dir / "predictions_none.jsonl", none_rows)

    print("\n=== Storage validation result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = (
        "PASS: correct adapter > no adapter"
        if result["correct_beats_none"]
        else "FAIL: correct adapter <= no adapter"
    )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
