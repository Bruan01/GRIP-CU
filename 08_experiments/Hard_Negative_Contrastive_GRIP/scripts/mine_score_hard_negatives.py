"""Mine fixed negatives with a frozen B1 continuation scorer.

Only relation-like QA items from the supplied task file are mined. The script
scores every train-graph relation against each QA prefix and writes a JSONL
manifest. It never reads validation/test predictions; their triples are used
only to remove known true relations from the negative-selection pool.
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
from hard_negative_grip.listed_training import score_candidate_rows  # noqa: E402
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.score_hard import (  # noqa: E402
    merge_negative_sources,
    rank_scores,
    select_score_hard_negatives,
)
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    assistant_gold,
    is_relation_gold,
    known_pair_relations,
    load_json_payload,
    match_train_relation,
    question_entity_pair,
    train_relation_alias_index,
)
from models.utils import get_hf_llm_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--b1_adapter", type=Path, required=True)
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="NELL23K raw split directory containing train.txt, valid.txt, and test.txt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--total_k", type=int, default=9)
    parser.add_argument("--hard_k", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and skip question IDs already present.",
    )
    return parser.parse_args()


def encode_without_specials(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    if ids:
        return ids
    unk = getattr(tokenizer, "unk_token_id", None)
    return [0 if unk is None else unk]


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


def score_relations(
    model,
    tokenizer,
    prefix: str,
    relations: list[str],
    batch_size: int,
) -> dict[str, float]:
    prefix_ids = encode_without_specials(tokenizer, prefix)
    device = next(model.parameters()).device
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(relations), batch_size):
            chunk = relations[start : start + batch_size]
            rows = [prefix_ids + encode_without_specials(tokenizer, rel) for rel in chunk]
            values = score_candidate_rows(
                model,
                tokenizer,
                rows,
                [len(prefix_ids)] * len(rows),
                device,
            )
            scores.extend(float(value) for value in values.detach().cpu())
    return dict(zip(relations, scores))


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: manifest row must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in completed:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id!r}")
            completed.add(question_id)
    return completed


def main() -> None:
    args = parse_args()
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if args.total_k < 0 or args.hard_k < 0 or args.hard_k > args.total_k:
        raise ValueError("require 0 <= hard_k <= total_k")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    payload = load_json_payload(args.input_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{args.input_file} is not a GRIP task file")
    relation_order = load_train_relation_order(args.raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_relations = known_pair_relations(args.raw_dir)
    model, tokenizer = load_b1(args)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids(args.output) if args.resume and args.output.is_file() else set()
    mode = "a" if args.resume else "w"
    started = time.time()
    processed = 0
    skipped = 0
    with args.output.open(mode, encoding="utf-8") as stream:
        for index, text in enumerate(payload["qa_samples"]):
            question_id = f"task_qa:{index}"
            if question_id in completed_ids:
                continue
            gold = assistant_gold(text)
            if not is_relation_gold(gold, text):
                skipped += 1
                continue
            matched = match_train_relation(gold, alias_index)
            if matched is None:
                skipped += 1
                continue
            if args.limit and processed >= args.limit:
                break
            scores = score_relations(
                model,
                tokenizer,
                assistant_answer_prefix(text),
                relation_order,
                args.candidate_batch_size,
            )
            gold_rank, gold_score = rank_scores(matched, scores)
            pair = question_entity_pair(text)
            excluded_relations = known_relations.get(pair, set()) if pair else set()
            hard, uniform = select_score_hard_negatives(
                matched,
                relation_order,
                scores,
                total_k=args.total_k,
                hard_k=args.hard_k,
                rng=rng,
                excluded_relations=excluded_relations,
            )
            negatives = merge_negative_sources(hard, uniform)
            ranked = sorted(scores, key=lambda rel: (-scores[rel], rel))
            row = {
                "question_id": question_id,
                "split": "train",
                "positive_relation": gold,
                "matched_train_relation": matched,
                "negative_relations": negatives,
                "hard_negative_relations": hard,
                "uniform_negative_relations": uniform,
                "positive_score": gold_score,
                "positive_rank": gold_rank,
                "hard_negative_scores": [
                    {"relation": rel, "score": float(scores[rel])} for rel in hard
                ],
                "all_candidate_scores": [
                    {"relation": rel, "score": float(scores[rel]), "rank": rank + 1}
                    for rank, rel in enumerate(ranked)
                ],
                "known_pair_relations": sorted(excluded_relations),
                "entity_pair": list(pair) if pair else None,
                "teacher": "frozen_b1",
                "negative_source": "b1_score_plus_uniform",
                "seed": args.seed,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            completed_ids.add(question_id)
            processed += 1
            if args.progress_every > 0 and (
                processed == 1 or processed % args.progress_every == 0
            ):
                elapsed = time.time() - started
                print(
                    f"[mine] {processed} items elapsed={elapsed / 3600:.2f}h "
                    f"rate={processed / max(elapsed, 1):.3f}/s gold_rank={gold_rank}",
                    flush=True,
                )
    print(
        f"[mine] complete processed={processed} skipped={skipped} "
        f"output={args.output} seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
