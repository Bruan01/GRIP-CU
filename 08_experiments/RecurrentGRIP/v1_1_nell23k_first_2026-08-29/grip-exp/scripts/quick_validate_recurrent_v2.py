"""Fixed RecurrentGRIP v2: MLP-LoRA storage across all layers + recurrent execution.

This is the corrected version of RecurrentGRIP. The failed pilot stored the graph
in a single-layer attention LoRA (``layers_to_transform=[12]``), which destroyed the
"storage" step. This script keeps the recurrent *execution* machinery (wrap one
decoder layer and repeat it K times) but stores the graph in an MLP LoRA across all
layers, matching the paper's GRIP recipe.

Comparison target: the storage-only GRIP baseline (quick01) reached
correct adapter 47.8% vs no-adapter 22.5% on the same 0.5B + NELL23K data.
Here we check whether recurrent execution (K>1) adds anything on top of that.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from evaluation.recurrent_metrics import exact_match, parse_recurrent_answer
from grip.recurrent.executor import find_decoder_layers, set_recurrent_depth, wrap_decoder_layer
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset
from grip.tasks.recurrent_tasks import build_recurrent_task_dataset
from grip.tasks.train_tasks.gen_context import GenGraphContextTask
from grip.training import train
from models.utils import get_hf_llm_tokenizer
from utils import load_list_json, set_random_seed


def _parse_target(answer: object) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    return [str(value) for value in answer]


def subset_graph(graph: dict, max_edges: Optional[int]) -> dict:
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
    return {**graph, "node_list": new_node_list, "edge_list": edge_list, "edge_index": new_edge_index}


def build_context_samples(record: dict, tokenizer, context_upsampling: bool) -> list[str]:
    return GenGraphContextTask(
        graph_list=[record["graph"]],
        title_list=[record.get("title", "nell23k")],
        tokenizer=tokenizer,
        context_upsampling=context_upsampling,
        format_as_instruction=False,
    )()[0]


def build_storage_recurrent_model(
    model_id: str,
    model_cache_dir: str,
    lora_r: int,
    lora_alpha: int,
    target_modules: list[str],
    recurrent_depth_train: int,
    executor_layer_index: int = -1,
):
    """Load base model, wrap one decoder layer for recurrence, attach MLP LoRA everywhere."""
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=model_cache_dir,
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map=None,  # CPU load so wrapping a ModuleList element is plain PyTorch
    )
    _, layers = find_decoder_layers(base_model)
    resolved_index = wrap_decoder_layer(base_model, executor_layer_index, recurrent_depth_train)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # no layers_to_transform -> LoRA on ALL layers
    )
    model = get_peft_model(base_model, lora_config, adapter_name="mylora")
    model.config.use_cache = False
    return model, tokenizer, resolved_index


def evaluate(model, tokenizer, record, max_new_tokens, depth_sweep, max_eval_questions, adapter_control, predictions_path=None):
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
        for depth in depth_sweep:
            set_recurrent_depth(model, depth)
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
                    use_cache=False,
                )
                raw = tokenizer.decode(
                    generated[0][input_ids.shape[-1]:], skip_special_tokens=True
                )
                parsed = parse_recurrent_answer(raw)
                target = _parse_target(answer)
                row = {
                    "question_id": samples[index].get("question_id"),
                    "split": samples[index].get("split"),
                    "question": question,
                    "target": target,
                    "raw_response": raw.strip(),
                    "response": parsed,
                    "correct": exact_match(parsed, target),
                    "recurrence_k": int(depth),
                    "adapter_control": adapter_control,
                }
                rows.append(row)
                if predictions_path is not None:
                    with predictions_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (index + 1) % 50 == 0 or (index + 1) == len(dataset):
                    print(
                        f"[eval {adapter_control} depth={depth}] "
                        f"{index + 1}/{len(dataset)} done",
                        flush=True,
                    )
    return rows


def summarize(rows: list[dict]) -> dict:
    by_k_adapter: dict[str, list[bool]] = {}
    for row in rows:
        key = f"k={row['recurrence_k']}|adapter={row['adapter_control']}"
        by_k_adapter.setdefault(key, []).append(bool(row["correct"]))
    by_k_adapter_summary = {
        key: {"count": len(values), "em": sum(values) / len(values) if values else 0.0}
        for key, values in sorted(by_k_adapter.items())
    }

    def best_k(control: str) -> tuple[int, float]:
        k_em = {}
        for (key, values) in by_k_adapter.items():
            k = int(key.split("|")[0].split("=")[1])
            adapter = key.split("adapter=")[1]
            if adapter == control:
                k_em[k] = sum(values) / len(values) if values else 0.0
        if not k_em:
            return 0, 0.0
        best = max(k_em.items(), key=lambda item: item[1])
        return best

    correct_best_k, correct_best_em = best_k("correct")
    correct_k1_em = by_k_adapter_summary.get("k=1|adapter=correct", {}).get("em", 0.0)
    none_k1_em = by_k_adapter_summary.get("k=1|adapter=none", {}).get("em", 0.0)

    return {
        "by_k_adapter": by_k_adapter_summary,
        "correct_best_k": correct_best_k,
        "correct_best_em": correct_best_em,
        "correct_k1_em": correct_k1_em,
        "none_k1_em": none_k1_em,
        "recurrent_gain_over_k1": correct_best_em - correct_k1_em,
    }


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
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--context_upsampling", action="store_true")
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--max_context_edges", type=int, default=0)
    parser.add_argument("--max_eval_questions", type=int, default=0)
    parser.add_argument("--recurrent_depth_train", type=int, default=2)
    parser.add_argument("--recurrent_depth_sweep", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--executor_layer_index", type=int, default=-1)
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
    model, tokenizer, executor_index = build_storage_recurrent_model(
        model_id=model_id,
        model_cache_dir=args.model_cache_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        recurrent_depth_train=args.recurrent_depth_train,
        executor_layer_index=args.executor_layer_index,
    )

    context_samples = build_context_samples(record, tokenizer, args.context_upsampling)
    dataset = build_recurrent_task_dataset(
        tokenizer=tokenizer,
        title=record.get("title", "nell23k"),
        context_samples=context_samples,
        samples=record["recurrent_questions"],
    )

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Executor layer index: {executor_index}")
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
        set_recurrent_depth(model, args.recurrent_depth_train)
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
        mylora_dir = adapter_dir / "mylora"
        if mylora_dir.is_dir():
            for item in mylora_dir.iterdir():
                item.rename(adapter_dir / item.name)
            mylora_dir.rmdir()
        print(f"Adapter saved to {adapter_dir.resolve()}")

    if model.device.type == "cpu":
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")  # fresh, append during eval

    correct_rows = evaluate(
        model, tokenizer, record, args.gen_max_length,
        args.recurrent_depth_sweep, args.max_eval_questions, "correct",
        predictions_path=predictions_path,
    )
    with model.disable_adapter():
        none_rows = evaluate(
            model, tokenizer, record, args.gen_max_length,
            args.recurrent_depth_sweep, args.max_eval_questions, "none",
            predictions_path=predictions_path,
        )

    all_rows = correct_rows + none_rows
    result = {
        "model": args.model_name,
        "executor_layer_index": executor_index,
        "recurrent_depth_train": args.recurrent_depth_train,
        "recurrent_depth_sweep": args.recurrent_depth_sweep,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": args.target_modules},
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "involve_qa_epochs": args.involve_qa_epochs,
            "context_samples": len(context_samples),
            "learning_rate": args.learning_rate,
            "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        },
        **summarize(all_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== Fixed RecurrentGRIP v2 result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
