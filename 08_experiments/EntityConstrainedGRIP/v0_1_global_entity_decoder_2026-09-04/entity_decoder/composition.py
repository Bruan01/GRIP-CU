"""Train-only composition labels and analysis buckets."""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def build_train_compositions(rows: Iterable[dict]) -> set[tuple[str, ...]]:
    return {tuple(map(str, row["path_relations"])) for row in rows}


def annotate_composition(row: dict, train_compositions: set[tuple[str, ...]]) -> str:
    key = tuple(map(str, row["path_relations"]))
    return "seen-composition" if key in train_compositions else "novel-composition"


def relation_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(map(str, row["path_relations"]))
    return dict(counts)


def relation_frequency_bucket(relations: Iterable[str], counts: dict[str, int], rare_max: int = 5, frequent_min: int = 21) -> str:
    values = [int(counts.get(str(relation), 0)) for relation in relations]
    minimum = min(values) if values else 0
    if minimum <= rare_max:
        return "rare"
    if minimum >= frequent_min:
        return "frequent"
    return "medium"


def prefix_ambiguity(entity_tokens: list[int], all_token_sequences: Iterable[Iterable[int]], prefix_length: int = 2) -> int:
    """Count entities sharing the first k answer tokens (k=min(prefix_length, answer length))."""
    width = min(max(1, int(prefix_length)), len(entity_tokens))
    prefix = tuple(entity_tokens[:width])
    return sum(tuple(sequence[:width]) == prefix for sequence in all_token_sequences)
