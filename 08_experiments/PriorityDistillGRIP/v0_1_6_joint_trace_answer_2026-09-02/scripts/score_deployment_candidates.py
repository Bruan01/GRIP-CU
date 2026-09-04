#!/usr/bin/env python3
"""Score answer candidates under the exact graph-free deployment prompt.

Unlike the structured trace audit, this scorer supplies no gold trace or
intermediate node.  It compares the four candidate answer strings after the
same answer-only evaluation prompt used by constrained decoding.  The result
is an exact sequence-level candidate selector (not open-vocabulary generation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from priority_distill.candidates import build_candidate_pool_for_query
from priority_distill.config import load_config
from priority_distill.experiment import _device_context, load_runtime
from priority_distill.io_utils import write_json, write_jsonl
from priority_distill.modules import load_adapter_state_dict
from priority_distill.records import build_evaluation_prompt, load_splits


def score_sequences(model, tokenizer, prompts: list[str], answers: list[str], device: torch.device, dtype: torch.dtype, batch_size: int, length_normalization: float) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        sequences: list[list[int]] = []
        offsets: list[int] = []
        for prompt, answer in zip(prompts[start:start + batch_size], answers[start:start + batch_size]):
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
            if not answer_ids:
                raise ValueError(f"empty answer tokenization: {answer!r}")
            sequences.append(prompt_ids + answer_ids)
            offsets.append(len(prompt_ids))
        max_len = max(len(seq) for seq in sequences)
        pad_id = int(tokenizer.pad_token_id)
        input_ids = torch.tensor([seq + [pad_id] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)
        attention = torch.tensor([[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)
        with torch.inference_mode(), _device_context(device, dtype):
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        for index, (seq, offset) in enumerate(zip(sequences, offsets)):
            token_scores = [log_probs[index, position - 1, token].item() for position, token in enumerate(seq[offset:], start=offset)]
            scores.append(sum(token_scores) / (len(token_scores) ** length_normalization))
    return scores


def summarize(rows: list[dict]) -> dict:
    ranks = [int(row["rank"]) for row in rows]
    return {
        "count": len(rows),
        "rank1_accuracy": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else 0.0,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
        "rank_counts": {str(rank): ranks.count(rank) for rank in range(1, 5)},
        "accuracy_by_depth": {
            str(depth): sum(row["rank"] == 1 for row in rows if int(row["depth_label"]) == depth) / max(1, sum(int(row["depth_label"]) == depth for row in rows))
            for depth in range(1, 5)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length-normalization", type=float, default=0.0, help="0=sequence sum, 1=mean token log-probability")
    args = parser.parse_args()
    if args.length_normalization < 0:
        raise ValueError("--length-normalization must be non-negative")
    config = load_config(args.config.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, _, audit = load_splits(REPO, config["data"])
    model, tokenizer, _, device, dtype, resolved = load_runtime(config, args.model_name_or_path)
    load_adapter_state_dict(model, torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=True))
    model.eval()

    rows = splits[args.split]
    predictions: list[dict] = []
    for row in tqdm(rows, desc=f"deployment candidate scoring {args.split}", unit="query"):
        pool = build_candidate_pool_for_query(row, splits["train"], int(config["candidates"]["distractor_count"]), int(config["candidates"]["seed"]))
        answers = [str(candidate["answer"]) for candidate in pool]
        prompts = [build_evaluation_prompt(row["text"])] * len(answers)
        scores = score_sequences(model, tokenizer, prompts, answers, device, dtype, args.batch_size, args.length_normalization)
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        gold_index = next(index for index, candidate in enumerate(pool) if candidate["is_gold"])
        rank = order.index(gold_index) + 1
        predictions.append({
            "task_id": row["task_id"],
            "depth_label": int(row["depth_label"]),
            "answer": row["answer"],
            "condition": "deployment_answer_prompt",
            "rank": rank,
            "gold_score": scores[gold_index],
            "best_distractor_score": max(score for index, score in enumerate(scores) if index != gold_index),
            "margin": scores[gold_index] - max(score for index, score in enumerate(scores) if index != gold_index),
            "candidate_scores": [
                {"answer": answers[index], "is_gold": bool(pool[index]["is_gold"]), "score": scores[index]}
                for index in order
            ],
        })
    report = {
        "format_version": 1,
        "purpose": "deployment_prompt_sequence_level_candidate_ranking",
        "prompt": "priority_distill.records.build_evaluation_prompt; no gold trace, path, or evidence",
        "length_normalization": args.length_normalization,
        "score_definition": "sum(log p(answer_token | answer_only_deployment_prompt, previous_answer_tokens)) / len(answer_tokens)^length_normalization",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "model": resolved,
        "device": str(device),
        "dtype": str(dtype),
        "data": audit,
        "metrics": summarize(predictions),
        "candidate_set": "one held-out gold answer plus three deterministic same-depth train-only distractors with unique answer strings",
    }
    write_json(output_dir / f"deployment_candidate_ranking_{args.split}.json", report)
    write_jsonl(output_dir / f"deployment_candidate_ranking_rows_{args.split}.jsonl", predictions)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
