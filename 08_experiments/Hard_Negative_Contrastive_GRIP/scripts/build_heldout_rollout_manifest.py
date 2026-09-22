"""Transfer held-out rollout errors onto a disjoint Stage-2 train split.

The source manifest was mined on held-out QA with a frozen teacher. Its observed
errors become a global hard pool. Target QA items receive at most one hard
label from that pool plus uniform train-graph negatives. The source QA text is
never reused for generation loss, so the holdout remains a mining-only split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.score_hard import merge_negative_sources  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_gold,
    is_relation_gold,
    known_pair_relations,
    load_json_payload,
    match_train_relation,
    question_entity_pair,
    train_relation_alias_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-task", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-k", type=int, default=9)
    parser.add_argument("--hard-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def main() -> None:
    args = parse_args()
    if args.total_k < 1 or args.hard_k < 0 or args.hard_k > args.total_k:
        raise ValueError("require total_k >= 1 and 0 <= hard_k <= total_k")
    source_rows = load_manifest(args.source_manifest)
    payload = load_json_payload(args.target_task)
    if not isinstance(payload, dict) or not isinstance(payload.get("qa_samples"), list):
        raise ValueError(f"{args.target_task} is not a GRIP task file")

    relation_order = load_train_relation_order(args.raw_dir)
    relation_set = set(relation_order)
    alias_index = train_relation_alias_index(relation_order)
    known = known_pair_relations(args.raw_dir)
    hard_counts = Counter(
        str(row["rollout_hard_negative"])
        for row in source_rows
        if row.get("rollout_hard_negative")
    )
    hard_pool = [relation for relation, _ in hard_counts.most_common()]
    if not hard_pool:
        raise ValueError("source manifest contains no rollout hard negatives")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    skipped = 0
    hard_used = 0
    for index, text in enumerate(payload["qa_samples"]):
        question_id = f"task_qa:{index}"
        gold = assistant_gold(text)
        if not is_relation_gold(gold, text):
            continue
        matched = match_train_relation(gold, alias_index)
        if matched is None:
            skipped += 1
            continue
        pair = question_entity_pair(text)
        excluded = known.get(pair, set()) if pair else set()
        excluded_canonical = {
            rel if rel.startswith("concept:") else f"concept:{rel}" for rel in excluded
        }
        hard: list[str] = []
        if args.hard_k:
            eligible_hard = [
                relation
                for relation in hard_pool
                if relation != gold
                and relation != matched
                and (
                    relation if relation.startswith("concept:") else f"concept:{relation}"
                ) not in excluded_canonical
            ]
            if eligible_hard:
                hard = rng.sample(eligible_hard, min(args.hard_k, len(eligible_hard)))
                hard_used += len(hard)

        pool = [
            relation
            for relation in relation_order
            if relation != matched and relation not in excluded and relation not in hard
        ]
        uniform = rng.sample(pool, min(args.total_k - len(hard), len(pool)))
        negatives = merge_negative_sources(hard, uniform)
        if len(negatives) != min(args.total_k, len(hard) + len(pool)):
            raise ValueError(f"could not build negatives for {question_id}")
        rows.append(
            {
                "question_id": question_id,
                "split": "train",
                "positive_relation": gold,
                "matched_train_relation": matched,
                "negative_relations": negatives,
                "hard_negative_relations": hard,
                "uniform_negative_relations": uniform,
                "known_pair_relations": sorted(excluded),
                "entity_pair": list(pair) if pair else None,
                "negative_source": "heldout_rollout_error_pool_plus_uniform",
                "source_manifest": str(args.source_manifest),
                "source_hard_pool": hard_pool,
                "source_hard_counts": dict(hard_counts),
                "hard_k": args.hard_k,
                "total_k": args.total_k,
                "seed": args.seed,
            }
        )

    if not rows:
        raise ValueError("target task contains no matched relation QA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "source_rows": len(source_rows),
                "target_rows": len(rows),
                "skipped_target_relation_rows": skipped,
                "source_hard_pool": hard_pool,
                "source_hard_counts": dict(hard_counts),
                "hard_used": hard_used,
                "mean_hard_per_target": hard_used / len(rows),
                "total_k": args.total_k,
                "hard_k": args.hard_k,
                "output": str(args.output),
                "train_vocab_size": len(relation_set),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
