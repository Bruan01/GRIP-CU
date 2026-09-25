#!/usr/bin/env python3
"""Score 20 dump QA with the training scorer and the offline scorer.

Loads the frozen B1 adapter once. Does not train. Runs each scorer twice and
compares candidate_score, ranks, and output order against the production dump.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE  # noqa: E402
from hard_negative_grip.confusion_analysis import iter_qa_groups_from_scores  # noqa: E402
from hard_negative_grip.confusion_audit import (  # noqa: E402
    DEFAULT_SEED,
    SCORE_COMPARE_ATOL,
    SCORER_QA_SAMPLE,
    compare_score_groups,
    reservoir_add,
    summarize_comparisons,
)
from hard_negative_grip.listed_training import ListedContrastiveTrainer  # noqa: E402
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.offline_scoring import dense_ranks, parse_matchable_relation_qa  # noqa: E402
from hard_negative_grip.scoring import (  # noqa: E402
    pack_decision_set_rows,
    score_candidates,
)
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    load_json_payload,
    train_relation_alias_index,
)
from models.utils import get_hf_llm_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--b1_adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--qa_ids", type=Path, default=None)
    parser.add_argument("--n_qa", type=int, default=SCORER_QA_SAMPLE)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--atol", type=float, default=SCORE_COMPARE_ATOL)
    return parser.parse_args()


def load_b1(args: argparse.Namespace):
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, str(args.b1_adapter), is_trainable=False)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def score_maps_to_rows(score_map: dict[str, float], relation_order: list[str]) -> list[dict]:
    ranks = dense_ranks([(rel, score_map[rel]) for rel in relation_order])
    return [
        {
            "candidate_relation": rel,
            "candidate_score": score_map[rel],
            "candidate_rank_all": ranks[rel],
        }
        for rel in relation_order
    ]


def dump_group_to_rows(group: list[dict], relation_order: list[str]) -> list[dict]:
    by_rel = {str(row["candidate_relation"]): row for row in group}
    return [
        {
            "candidate_relation": rel,
            "candidate_score": float(by_rel[rel]["candidate_score"]),
            "candidate_rank_all": int(by_rel[rel]["candidate_rank_all"]),
        }
        for rel in relation_order
        if rel in by_rel
    ]


def sample_dump_groups(scores_path: Path, wanted: set[str] | None, n_qa: int, seed: int) -> dict[str, list[dict]]:
    if wanted:
        found: dict[str, list[dict]] = {}
        for group in iter_qa_groups_from_scores(scores_path):
            qa_id = str(group[0].get("qa_id") or "")
            if qa_id in wanted:
                found[qa_id] = group
                if len(found) == len(wanted):
                    break
        return found
    rng = random.Random(seed + 4)
    pool: list[list[dict]] = []
    seen = 0
    for group in iter_qa_groups_from_scores(scores_path):
        seen += 1
        reservoir_add(pool, group, seen, rng, n_qa)
    return {str(group[0]["qa_id"]): group for group in pool}


def score_offline(model, tokenizer, prefix: str, relations: list[str], batch_size: int) -> dict[str, float]:
    device = next(model.parameters()).device
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(relations), batch_size):
            chunk = relations[start : start + batch_size]
            scored = score_candidates(model, tokenizer, prefix, chunk, device=device)
            scores.extend(float(value) for value in scored["candidate_score"][0])
    return dict(zip(relations, scores))


def score_training(model, tokenizer, prefix: str, relations: list[str], batch_size: int) -> dict[str, float]:
    device = next(model.parameters()).device
    trainer = ListedContrastiveTrainer.__new__(ListedContrastiveTrainer)
    trainer.candidate_forwards = 0
    trainer.accelerator = None
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(relations), batch_size):
            chunk = relations[start : start + batch_size]
            rows, prefix_lens = pack_decision_set_rows(tokenizer, prefix, chunk)
            tensor = trainer._score_candidate_rows(model, tokenizer, rows, prefix_lens, device)
            scores.extend(float(value) for value in tensor.detach().cpu())
    return dict(zip(relations, scores))


def compare_maps(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    relation_order: list[str],
    label: str,
) -> dict:
    rows = []
    for qa_id in sorted(set(left) & set(right)):
        comparison = compare_score_groups(
            score_maps_to_rows(left[qa_id], relation_order),
            score_maps_to_rows(right[qa_id], relation_order),
        )
        rows.append({"qa_id": qa_id, **comparison})
    summary = summarize_comparisons(rows, label=label)
    summary["output_order_mismatch"] = any(row.get("output_order_mismatch") for row in rows)
    return summary


def main() -> None:
    args = parse_args()
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    relation_order = load_train_relation_order(args.raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    payload = load_json_payload(args.task_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{args.task_file} is not a GRIP task file")
    qa_by_id = {}
    for index, text in enumerate(payload["qa_samples"]):
        item = parse_matchable_relation_qa(index, str(text), alias_index)
        if item is not None:
            qa_by_id[item["qa_id"]] = item

    wanted = None
    if args.qa_ids is not None and args.qa_ids.is_file():
        loaded = json.loads(args.qa_ids.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded = loaded.get("qa_ids") or loaded.get("sampled_scorer_qa_ids") or []
        wanted = {str(item) for item in loaded}
    dump_groups = sample_dump_groups(args.scores, wanted, args.n_qa, args.seed)
    qa_ids = [qa_id for qa_id in dump_groups if qa_id in qa_by_id]
    if not qa_ids:
        raise ValueError("no overlapping QA ids between dump sample and task file")

    model, tokenizer = load_b1(args)
    dump_maps = {
        qa_id: {
            str(row["candidate_relation"]): float(row["candidate_score"])
            for row in dump_groups[qa_id]
        }
        for qa_id in qa_ids
    }
    dump_rows = {qa_id: dump_group_to_rows(dump_groups[qa_id], relation_order) for qa_id in qa_ids}

    def run_pass(scorer_name: str, fn) -> dict[str, dict[str, float]]:
        produced = {}
        started = time.time()
        for index, qa_id in enumerate(qa_ids, start=1):
            prefix = assistant_answer_prefix(qa_by_id[qa_id]["text"])
            produced[qa_id] = fn(model, tokenizer, prefix, relation_order, args.candidate_batch_size)
            print(
                f"[{scorer_name}] {index}/{len(qa_ids)} {qa_id}",
                flush=True,
            )
        print(
            json.dumps(
                {
                    "scorer": scorer_name,
                    "n_qa": len(qa_ids),
                    "elapsed_sec": time.time() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return produced

    training_1 = run_pass("training_1", score_training)
    training_2 = run_pass("training_2", score_training)
    offline_1 = run_pass("offline_1", score_offline)
    offline_2 = run_pass("offline_2", score_offline)

    dump_vs_training_rows = [
        {"qa_id": qa_id, **compare_score_groups(dump_rows[qa_id], score_maps_to_rows(training_1[qa_id], relation_order))}
        for qa_id in qa_ids
    ]
    dump_vs_offline_rows = [
        {"qa_id": qa_id, **compare_score_groups(dump_rows[qa_id], score_maps_to_rows(offline_1[qa_id], relation_order))}
        for qa_id in qa_ids
    ]
    result = {
        "n_qa": len(qa_ids),
        "qa_ids": qa_ids,
        "checkpoint": str(args.b1_adapter),
        "candidate_batch_size": args.candidate_batch_size,
        "atol": args.atol,
        "dump_vs_training": summarize_comparisons(dump_vs_training_rows, label="dump_vs_training"),
        "dump_vs_offline": summarize_comparisons(dump_vs_offline_rows, label="dump_vs_offline"),
        "training_vs_offline": compare_maps(training_1, offline_1, relation_order, "training_vs_offline"),
        "training_run1_vs_run2": compare_maps(training_1, training_2, relation_order, "training_run1_vs_run2"),
        "offline_run1_vs_run2": compare_maps(offline_1, offline_2, relation_order, "offline_run1_vs_run2"),
    }
    result["dump_vs_training"]["output_order_mismatch"] = any(
        row.get("output_order_mismatch") for row in dump_vs_training_rows
    )
    result["dump_vs_offline"]["output_order_mismatch"] = any(
        row.get("output_order_mismatch") for row in dump_vs_offline_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_qa", "qa_ids") if k in result}, ensure_ascii=False), flush=True)
    print(json.dumps({k: result[k] for k in result if k not in {"qa_ids"}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
