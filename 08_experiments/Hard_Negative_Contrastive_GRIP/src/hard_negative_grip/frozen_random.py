"""Frozen uniform negatives for a conditional, compute-matched SC-CL control.

Scores are inspected only to verify the parent's legal pool. Neither scores
nor structural features enter the random selection probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

FROZEN_RANDOM_SOURCE = "train_only_frozen_uniform_without_replacement"
SELECTION_RULE = "uniform_without_replacement_from_parent_legal_pool"


def pool_digest(relations: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(relations), ensure_ascii=False).encode()).hexdigest()


def uniform_bank_row(
    rows: list[dict], *, relation_order: list[str], known_relations: set[str],
    seed: int = 2026, k: int = 9,
) -> tuple[dict, dict]:
    """Verify the entire parent score group and draw K uniform candidates."""
    if not rows or k < 1 or any(row.get("split") != "train" for row in rows):
        raise ValueError("uniform control requires nonempty train-only score groups and K>=1")
    ids = {row.get("qa_id") for row in rows}
    gold_rows = [row for row in rows if row.get("is_gold") is True]
    if len(ids) != 1 or None in ids or "" in ids or len(gold_rows) != 1:
        raise ValueError("score group requires one question ID and exactly one gold")
    gold_row = gold_rows[0]
    qid, gold = str(gold_row["qa_id"]), str(gold_row["matched_train_relation"])
    vocabulary = set(relation_order)
    candidates = [row["candidate_relation"] for row in rows]
    if len(candidates) != len(set(candidates)) or set(candidates) != vocabulary:
        raise ValueError("parent score group does not cover the unique train vocabulary")
    if gold not in vocabulary or gold_row["candidate_relation"] != gold:
        raise ValueError("gold relation is inconsistent with train vocabulary")
    for row in rows:
        name = row["candidate_relation"]
        if row.get("matched_train_relation") != gold or row.get("gold_relation") != gold_row["gold_relation"]:
            raise ValueError("inconsistent gold across parent score rows")
        expected_valid = name != gold and name not in known_relations
        if row.get("is_valid_negative") is not expected_valid or row.get("is_gold") is not (name == gold):
            raise ValueError("parent legal pool differs from train-known filtering")
    pool = sorted(vocabulary - known_relations - {gold})
    if len(pool) < k:
        raise ValueError(f"{qid}: legal pool has {len(pool)} candidates, fewer than K={k}")
    digest = hashlib.sha256(f"{int(seed)}::{qid}".encode()).digest()
    row_seed = int.from_bytes(digest[:8], "big")
    negatives = random.Random(row_seed).sample(pool, k)
    bank = {
        "question_id": qid, "split": "train", "positive_relation": gold_row["gold_relation"],
        "matched_train_relation": gold, "gold_relation": gold,
        "negative_source": FROZEN_RANDOM_SOURCE, "selection_rule": SELECTION_RULE,
        "filter_splits": ["train"], "negative_relations": negatives,
        "hard_negative_relations": [], "uniform_negative_relations": negatives,
        "k": k, "actual_k": k, "requested_k": k, "seed": int(seed),
        "pool_size": len(pool), "pool_sha256": pool_digest(pool),
        "excluded_train_relations": sorted(known_relations | {gold}),
        "selected_negatives": [
            {"relation": name, "sampling_probability": 1/len(pool),
             "conditional_probability": 1/(len(pool)-i), "train_known_absent": True}
            for i, name in enumerate(negatives)
        ],
    }
    audit = {
        "question_id": qid, "seed": int(seed), "row_seed": row_seed,
        "pool_size": len(pool), "pool_sha256": pool_digest(pool),
        "legal_relations": pool, "excluded_train_relations": sorted(known_relations | {gold}),
        "negative_relations": negatives, "parent_pool_verified": True,
    }
    return bank, audit


def validate_frozen_random_manifest(rows: dict[str, dict], *, k: int = 9) -> None:
    if not rows or k < 1:
        raise ValueError("frozen_random requires a nonempty bank and K>=1")
    for qid, row in rows.items():
        negatives, selected = row.get("negative_relations", []), row.get("selected_negatives", [])
        pool_size = row.get("pool_size", 0)
        if (
            row.get("question_id") != qid or row.get("split") != "train"
            or row.get("negative_source") != FROZEN_RANDOM_SOURCE
            or row.get("selection_rule") != SELECTION_RULE or row.get("filter_splits") != ["train"]
            or any(row.get(field) != k for field in ("k", "actual_k", "requested_k"))
            or len(negatives) != k or len(set(negatives)) != k or pool_size < k
            or row.get("hard_negative_relations") != []
            or row.get("uniform_negative_relations") != negatives
            or len(selected) != k or [item.get("relation") for item in selected] != negatives
            or set(negatives) & set(row.get("excluded_train_relations", []))
            or row.get("positive_relation") in negatives or row.get("matched_train_relation") in negatives
            or row.get("lambda_candidate") is not None
            or not isinstance(row.get("seed"), int) or not row.get("pool_sha256")
        ):
            raise ValueError(f"{qid}: frozen_random provenance/K/pool mismatch")
        for i, item in enumerate(selected):
            if (item.get("train_known_absent") is not True
                    or not math.isclose(float(item.get("sampling_probability", -1)), 1/pool_size)
                    or not math.isclose(float(item.get("conditional_probability", -1)), 1/(pool_size-i))):
                raise ValueError(f"{qid}: non-uniform selection probability")


def score_groups(path: Path):
    """Stream large score dumps, retaining only one question in memory."""
    current, group, seen = None, [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qa_id")
            if not qid:
                raise ValueError("score row missing qa_id")
            if qid != current:
                if group:
                    yield group
                if qid in seen:
                    raise ValueError("noncontiguous or duplicate score group")
                seen.add(qid)
                current, group = qid, []
            group.append(row)
    if group:
        yield group
