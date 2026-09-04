"""Deterministic train-only candidate-path construction."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable


def _stable_key(seed: int, *parts: object) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relation_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Combination of multiset Jaccard and same-position overlap."""
    left = list(left)
    right = list(right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    jaccard = shared / union if union else 0.0
    positions = sum(a == b for a, b in zip(left, right)) / max(len(left), len(right), 1)
    return 0.7 * jaccard + 0.3 * positions


def candidate_from_row(row: dict, *, is_gold: bool) -> dict:
    return {
        "source_task_id": row["task_id"],
        "source_split": row["split"],
        "answer": row["answer"],
        "depth_label": int(row["depth_label"]),
        "path_nodes": list(row["path_nodes"]),
        "path_relations": list(row["path_relations"]),
        "path_edges": [dict(edge) for edge in row["path_edges"]],
        "is_gold": bool(is_gold),
    }


def build_candidate_pools(train_rows: list[dict], distractor_count: int, seed: int) -> dict[str, list[dict]]:
    if distractor_count < 1:
        raise ValueError("distractor_count must be positive")
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("candidate pools must be built from train rows only")
    canonical_rows = sorted(train_rows, key=lambda row: row["task_id"])
    pools: dict[str, list[dict]] = {}
    for row in canonical_rows:
        eligible = [
            candidate
            for candidate in canonical_rows
            if candidate["task_id"] != row["task_id"]
            and candidate["answer"] != row["answer"]
            and int(candidate["depth_label"]) == int(row["depth_label"])
        ]
        eligible.sort(
            key=lambda candidate: (
                -relation_overlap(row["path_relations"], candidate["path_relations"]),
                _stable_key(seed, row["task_id"], candidate["task_id"]),
                candidate["task_id"],
            )
        )
        if len(eligible) < distractor_count:
            raise ValueError(
                f"not enough same-depth, different-answer distractors for {row['task_id']}: "
                f"need {distractor_count}, found {len(eligible)}"
            )
        selected = eligible[:distractor_count]
        pool = [candidate_from_row(row, is_gold=True)] + [candidate_from_row(item, is_gold=False) for item in selected]
        pools[row["task_id"]] = pool
    return pools
