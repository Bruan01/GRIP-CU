"""Train-only gold-plus-distractor candidate sets for D3 diagnostics."""
from __future__ import annotations

import hashlib


def _stable_key(seed: int, task_id: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{task_id}|{value}".encode("utf-8")).hexdigest()


def build_diagnostic_candidates(query: dict, train_rows: list[dict], *, distractor_count: int = 3, seed: int = 20260904) -> list[str]:
    answer = str(query["answer"])
    depth = int(query["depth_label"])
    pool = sorted({str(row["answer"]) for row in train_rows if int(row["depth_label"]) == depth and str(row["answer"]) != answer})
    if len(pool) < distractor_count:
        raise ValueError(f"not enough train-only same-depth distractors for {query['task_id']}")
    ordered = sorted(pool, key=lambda value: _stable_key(seed, str(query["task_id"]), value))
    candidates = [answer, *ordered[:distractor_count]]
    return sorted(candidates, key=lambda value: _stable_key(seed, str(query["task_id"]), "order|" + value))
