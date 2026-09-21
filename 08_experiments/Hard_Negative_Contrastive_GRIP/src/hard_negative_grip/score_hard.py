"""Utilities for B1-score-driven negative selection."""

from __future__ import annotations

import random
from typing import Iterable


def select_score_hard_negatives(
    gold: str,
    relation_order: Iterable[str],
    scores: dict[str, float],
    *,
    total_k: int = 9,
    hard_k: int = 1,
    rng: random.Random,
    excluded_relations: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Return score-hard and uniform negatives from one relation vocabulary.

    ``excluded_relations`` contains relations known to be true for this
    question's entity pair. They are scored for auditability but never used as
    negatives.
    """
    if total_k < 0:
        raise ValueError(f"total_k must be non-negative, got {total_k}")
    if hard_k < 0 or hard_k > total_k:
        raise ValueError(f"hard_k must be in [0, total_k], got {hard_k}")
    relations = list(dict.fromkeys(str(rel) for rel in relation_order))
    excluded = {str(rel) for rel in excluded_relations}
    candidates = [
        rel for rel in relations if rel != gold and rel in scores and rel not in excluded
    ]
    ranked = sorted(candidates, key=lambda rel: (-float(scores[rel]), rel))
    hard = ranked[: min(hard_k, len(ranked))]
    hard_set = set(hard)
    remaining = [rel for rel in candidates if rel not in hard_set]
    uniform_k = min(total_k - len(hard), len(remaining))
    uniform = rng.sample(remaining, uniform_k) if uniform_k else []
    return hard, uniform


def merge_negative_sources(hard: list[str], uniform: list[str]) -> list[str]:
    """Preserve hard-first provenance while removing accidental duplicates."""
    return list(dict.fromkeys([*hard, *uniform]))


def rank_scores(gold: str, scores: dict[str, float]) -> tuple[int, float]:
    """Return the one-indexed gold rank and its score."""
    if gold not in scores:
        raise ValueError(f"gold relation {gold!r} is absent from score table")
    ranked = sorted(scores, key=lambda rel: (-float(scores[rel]), rel))
    return ranked.index(gold) + 1, float(scores[gold])
