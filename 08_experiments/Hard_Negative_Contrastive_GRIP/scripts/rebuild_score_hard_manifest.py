#!/usr/bin/env python3
"""Rebuild a fixed score-hard manifest from an existing full score table.

The miner records every train-vocabulary candidate score for each question. This
script reuses those immutable scores to compare hard/uniform mixtures without
running the frozen teacher again. Hard candidates are selected by B1 score and
uniform candidates are sampled from the remaining valid candidates.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing score-hard JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="New immutable JSONL manifest.")
    parser.add_argument("--hard_k", type=int, default=4)
    parser.add_argument("--total_k", type=int, default=9, help="Total negatives per row.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=0,
        help="Require exactly this many input rows; 0 disables the check.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id!r}")
            seen.add(question_id)
            rows.append(row)
    return rows


def select_row(
    row: dict,
    *,
    hard_k: int,
    total_k: int,
    rng: random.Random,
) -> dict:
    scores = row.get("all_candidate_scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"{row['question_id']!r}: missing all_candidate_scores")

    score_map: dict[str, float] = {}
    for item in scores:
        if not isinstance(item, dict):
            raise ValueError(f"{row['question_id']!r}: invalid candidate score item")
        relation = str(item.get("relation") or "")
        if not relation or relation in score_map:
            raise ValueError(f"{row['question_id']!r}: duplicate/empty candidate relation")
        score_map[relation] = float(item["score"])

    gold = str(row.get("matched_train_relation") or "")
    if not gold or gold not in score_map:
        raise ValueError(f"{row['question_id']!r}: matched gold is absent from scores")
    known = {str(rel) for rel in (row.get("known_pair_relations") or [])}
    candidates = [rel for rel in score_map if rel != gold and rel not in known]
    ranked = sorted(candidates, key=lambda rel: (-score_map[rel], rel))
    hard = ranked[: min(hard_k, len(ranked))]
    hard_set = set(hard)
    remaining = [rel for rel in candidates if rel not in hard_set]
    uniform_k = min(total_k - len(hard), len(remaining))
    uniform = rng.sample(remaining, uniform_k) if uniform_k else []
    negatives = list(dict.fromkeys([*hard, *uniform]))
    if len(negatives) != min(total_k, len(candidates)):
        raise ValueError(f"{row['question_id']!r}: failed to build requested negative set")

    rebuilt = dict(row)
    rebuilt.update(
        {
            "negative_relations": negatives,
            "hard_negative_relations": hard,
            "uniform_negative_relations": uniform,
            "hard_negative_scores": [
                {"relation": rel, "score": score_map[rel]} for rel in hard
            ],
            "negative_source": "b1_score_hard_plus_uniform",
            "hard_k": hard_k,
            "total_k": total_k,
            "selection_rule": "top_b1_score_excluding_known_then_uniform_remaining",
            "source_manifest": "all_candidate_scores",
            "selection_seed": None,
        }
    )
    return rebuilt


def main() -> None:
    args = parse_args()
    if args.hard_k < 0 or args.total_k < 0 or args.hard_k > args.total_k:
        raise ValueError("require 0 <= hard_k <= total_k")
    if args.expected_rows < 0:
        raise ValueError("expected_rows must be non-negative")
    rows = load_rows(args.input)
    if args.expected_rows and len(rows) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} rows in {args.input}, found {len(rows)}; "
            "mining is not complete"
        )
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            rebuilt = select_row(row, hard_k=args.hard_k, total_k=args.total_k, rng=rng)
            rebuilt["selection_seed"] = args.seed
            stream.write(json.dumps(rebuilt, ensure_ascii=False) + "\n")
    temp.replace(args.output)
    print(
        f"[rebuild] rows={len(rows)} hard_k={args.hard_k} "
        f"uniform_k={args.total_k - args.hard_k} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
