#!/usr/bin/env python3
"""Teacher-forced candidate ranking audit for an existing adapter checkpoint.

The audit avoids free-form decoding.  For each held-out query it scores the
correct answer against three same-depth train distractors.  This separates
"the model assigns probability to the right entity" from "greedy generation
emits a malformed/truncated entity".
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
from transformers import AutoTokenizer

from priority_distill.candidates import build_candidate_pool_for_query
from priority_distill.config import load_config
from priority_distill.experiment import _device_context, load_runtime
from priority_distill.io_utils import write_json, write_jsonl
from priority_distill.modules import load_adapter_state_dict
from priority_distill.records import load_splits
from priority_distill.supervision import (
    build_gold_trace_prompt,
    build_graph_free_trace_evaluation_prompt,
    build_oracle_evidence_prompt,
)


def _answer_tokens(tokenizer, answer: str) -> list[int]:
    return tokenizer(str(answer), add_special_tokens=False)["input_ids"]


def _score_suffixes(model, tokenizer, prompts: list[str], prefixes: list[str], answers: list[str], device: torch.device, dtype: torch.dtype, batch_size: int) -> list[float]:
    """Mean teacher-forced log-probability of answer after prompt+prefix."""
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_prefixes = prefixes[start:start + batch_size]
            batch_answers = answers[start:start + batch_size]
            sequences = []
            answer_offsets = []
            for prompt, prefix, answer in zip(batch_prompts, batch_prefixes, batch_answers):
                prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                answer_ids = _answer_tokens(tokenizer, answer)
                if not answer_ids:
                    raise ValueError(f"empty answer tokenization: {answer!r}")
                sequences.append(prompt_ids + prefix_ids + answer_ids)
                answer_offsets.append(len(prompt_ids) + len(prefix_ids))
            max_len = max(map(len, sequences))
            pad_id = tokenizer.pad_token_id
            input_ids = torch.tensor([seq + [pad_id] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)
            attention = torch.tensor([[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=device)
            with _device_context(device, dtype):
                logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            for index, (seq, offset) in enumerate(zip(sequences, answer_offsets)):
                answer_ids = seq[offset:]
                token_scores = [log_probs[index, position - 1, token].item() for position, token in enumerate(answer_ids, start=offset)]
                scores.append(sum(token_scores) / len(token_scores))
    return scores


def _prompt_and_prefix(row: dict, condition: str) -> tuple[str, str]:
    if condition == "oracle_evidence":
        return build_oracle_evidence_prompt(row), ""
    if condition == "gold_trace":
        prompt = build_gold_trace_prompt(row)
    elif condition == "graph_free":
        prompt = build_graph_free_trace_evaluation_prompt(row["text"])
    else:
        raise ValueError(condition)
    expected_trace = " -> ".join([str(node) for node in row["path_nodes"][:-1]] + ["<MASKED_TERMINAL>"])
    return prompt, f"Trace: {expected_trace}\nAnswer: "


def _summarize(rows: list[dict]) -> dict:
    rank1 = sum(row["rank"] == 1 for row in rows)
    ranks = [row["rank"] for row in rows]
    return {
        "count": len(rows),
        "rank1_accuracy": rank1 / len(rows) if rows else 0.0,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
        "rank_counts": {str(rank): ranks.count(rank) for rank in range(1, 5)},
        "accuracy_by_depth": {
            str(depth): sum(r["rank"] == 1 for r in rows if int(r["depth_label"]) == depth) / max(1, sum(int(r["depth_label"]) == depth for r in rows))
            for depth in range(1, 5)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    config_path = args.config.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    splits, _, audit = load_splits(REPO, config["data"])
    model, tokenizer, injection, device, dtype, resolved = load_runtime(config, args.model_name_or_path)
    state = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=True)
    load_adapter_state_dict(model, state)

    all_rows: list[dict] = []
    for condition in ("graph_free", "gold_trace", "oracle_evidence"):
        condition_rows: list[dict] = []
        for row in tqdm(splits["test"], desc=f"candidate scoring {condition}", unit="query"):
            pool = build_candidate_pool_for_query(row, splits["train"], 3, int(config["candidates"]["seed"]))
            # Use the deterministic candidate order from the helper: gold first,
            # followed by same-depth train distractors.  Ranking is by score.
            prompt, prefix = _prompt_and_prefix(row, condition)
            prompts = [prompt] * len(pool)
            prefixes = [prefix] * len(pool)
            answers = [str(candidate["answer"]) for candidate in pool]
            scores = _score_suffixes(model, tokenizer, prompts, prefixes, answers, device, dtype, args.batch_size)
            order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
            gold_index = next(i for i, candidate in enumerate(pool) if candidate["is_gold"])
            rank = order.index(gold_index) + 1
            result = {
                "task_id": row["task_id"],
                "depth_label": int(row["depth_label"]),
                "answer": row["answer"],
                "condition": condition,
                "rank": rank,
                "gold_score": scores[gold_index],
                "best_distractor_score": max(score for i, score in enumerate(scores) if i != gold_index),
                "margin": scores[gold_index] - max(score for i, score in enumerate(scores) if i != gold_index),
                "candidate_scores": [
                    {"answer": answers[i], "is_gold": bool(pool[i]["is_gold"]), "score": scores[i]}
                    for i in order
                ],
            }
            condition_rows.append(result)
            all_rows.append(result)
        write_json(out / f"candidate_ranking_{condition}.json", _summarize(condition_rows))
    write_jsonl(out / "candidate_ranking_rows.jsonl", all_rows)
    report = {
        "format_version": 1,
        "purpose": "teacher_forced_answer_candidate_ranking",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "model": resolved,
        "device": str(device),
        "dtype": str(dtype),
        "data": audit,
        "conditions": {condition: _summarize([r for r in all_rows if r["condition"] == condition]) for condition in ("graph_free", "gold_trace", "oracle_evidence")},
        "interpretation": {
            "rank1_high_but_generation_low": "the model assigns probability to the answer but decoding/formatting may be the bottleneck",
            "rank1_low_even_with_oracle_evidence": "the model is not using the supplied path reliably",
            "candidate_set": "one held-out gold answer plus three deterministic same-depth train distractors; this is a relative diagnostic, not full-vocabulary accuracy",
        },
    }
    write_json(out / "candidate_ranking_audit.json", report)
    print(json.dumps(report["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
