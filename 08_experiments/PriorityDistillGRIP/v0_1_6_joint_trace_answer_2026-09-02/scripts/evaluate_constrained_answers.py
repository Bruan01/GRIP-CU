#!/usr/bin/env python3
"""Evaluate graph-free answers with per-query constrained entity decoding.

This is a post-hoc diagnostic: it loads an existing LoRA adapter and never
updates model weights.  Each held-out query gets one gold answer and three
same-depth, train-only distractors.  Generation is restricted to those four
entity strings, so malformed public-prefix outputs cannot be mistaken for an
entity.  This is a candidate-set diagnostic, not full-vocabulary accuracy.
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
from tqdm import tqdm
from transformers import LogitsProcessorList

from priority_distill.candidates import build_candidate_pool_for_query
from priority_distill.config import load_config
from priority_distill.constrained import CandidateTrieLogitsProcessor, deduplicate_candidates
from priority_distill.experiment import _device_context, load_runtime
from priority_distill.io_utils import environment_snapshot, write_json, write_jsonl
from priority_distill.metrics import normalize_entity
from priority_distill.modules import load_adapter_state_dict
from priority_distill.records import build_evaluation_prompt, load_splits


def _score_rows(rows: list[dict], predictions: list[dict]) -> dict:
    correct = [int(normalize_entity(p["prediction_answer"]) == normalize_entity(p["answer"])) for p in predictions]
    by_depth: dict[str, dict[str, int]] = {}
    for depth in range(1, 5):
        subset = [value for value, row in zip(correct, rows) if int(row["depth_label"]) == depth]
        by_depth[str(depth)] = {"correct": sum(subset), "count": len(subset), "accuracy": sum(subset) / len(subset) if subset else 0.0}
    return {
        "count": len(correct),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "random_candidate_baseline": 1.0 / 4.0,
        "by_depth": by_depth,
    }


def _load_state(path: Path) -> dict:
    return torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=True)


def evaluate_split(model, tokenizer, rows, train_rows, config, device, dtype, split_name, checkpoint_name, batch_size):
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    predictions: list[dict] = []
    distractor_count = int(config["candidates"]["distractor_count"])
    seed = int(config["candidates"]["seed"])
    with torch.inference_mode():
        progress = tqdm(
            range(0, len(rows), batch_size),
            total=(len(rows) + batch_size - 1) // batch_size,
            desc=f"constrained {split_name} {checkpoint_name}",
            unit="batch",
            disable=bool(config.get("runtime", {}).get("disable_progress", False)),
        )
        for start in progress:
            batch_rows = rows[start : start + batch_size]
            pools = [build_candidate_pool_for_query(row, train_rows, distractor_count, seed) for row in batch_rows]
            prompts = [build_evaluation_prompt(row["text"]) for row in batch_rows]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config["data"]["max_length"]),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            candidate_strings: list[list[str]] = []
            candidate_ids: list[list[list[int]]] = []
            for pool in pools:
                strings, ids = deduplicate_candidates(tokenizer, [str(item["answer"]) for item in pool])
                candidate_strings.append(strings)
                candidate_ids.append(ids)
            processor = CandidateTrieLogitsProcessor(
                candidate_ids,
                start_length=int(encoded["input_ids"].shape[1]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            )
            with _device_context(device, dtype):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max(len(ids) for row_ids in candidate_ids for ids in row_ids) + 1,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([processor]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_only = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            for row, pool, strings, ids, text in zip(batch_rows, pools, candidate_strings, candidate_ids, texts):
                normalized_text = normalize_entity(text)
                matched = [candidate for candidate in strings if normalize_entity(candidate) == normalized_text]
                prediction_answer = matched[0] if matched else text.strip()
                gold = next(candidate for candidate in pool if candidate["is_gold"])
                predictions.append({
                    "task_id": row["task_id"],
                    "depth_label": int(row["depth_label"]),
                    "answer": str(row["answer"]),
                    "prediction_text": text,
                    "prediction_answer": prediction_answer,
                    "candidate_answers": strings,
                    "gold_in_candidate_set": str(gold["answer"]) in strings,
                    "constrained_exact_match": normalize_entity(prediction_answer) == normalize_entity(row["answer"]),
                })
    tokenizer.padding_side = "right"
    return predictions, _score_rows(rows, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    splits, _, split_audit = load_splits(REPO, config["data"])
    model, tokenizer, injection, device, dtype, resolved_model = load_runtime(config, args.model_name_or_path)
    load_adapter_state_dict(model, _load_state(checkpoint_path))

    results = {}
    for split in args.splits:
        predictions, metrics = evaluate_split(
            model, tokenizer, splits[split], splits["train"], config, device, dtype,
            split, checkpoint_path.parent.name, args.batch_size,
        )
        write_jsonl(output_dir / f"predictions_{split}.jsonl", predictions)
        results[split] = metrics

    report = {
        "format_version": 1,
        "purpose": "graph_free_candidate_set_constrained_entity_decoding",
        "checkpoint": str(checkpoint_path),
        "model": resolved_model,
        "device": str(device),
        "dtype": str(dtype),
        "candidate_set": {
            "gold_plus_train_distractors": int(config["candidates"]["distractor_count"]) + 1,
            "same_depth_only": bool(config["candidates"]["same_depth_only"]),
            "train_split_only": bool(config["candidates"]["train_split_only"]),
        },
        "data": split_audit,
        "metrics": results,
        "environment": environment_snapshot(REPO),
        "interpretation": {
            "scope": "candidate-set diagnostic, not full-vocabulary accuracy",
            "higher_than_free_generation": "suggests malformed/open-vocabulary decoding is a major bottleneck",
            "near_random": "suggests the model does not reliably rank the correct entity even when generation is constrained",
        },
    }
    write_json(output_dir / "constrained_entity_decoding.json", report)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
