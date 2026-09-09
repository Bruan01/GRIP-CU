"""Replay the official GRIP NELL23K 10-way candidate protocol.

The paper processor is ``data/raw_datasets/nell23k/process.py``, invoked by
``scripts/process_raw_data.py --datasets nell23k --seed 2026``. It seeds global
``numpy.random`` once, then for every ``valid.txt`` triple and then every
``test.txt`` triple samples 9 negatives from the train-relation insertion order
and shuffles them with the gold relation. Train questions have no paper 10-way
list.

RecurrentGRIP's ``prepare_recurrent_nell23k.py`` uses the same question template
but a per-subset ``random.Random`` stream, so the same ``question_id`` can carry
a different 10-way list. This module rebuilds the frozen paper lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .candidates import parse_listed_relations

QUESTION_TEMPLATE = (
    "What is the relation between word node {src} and word node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)
WAY = 10
SPLIT_FILE = {"validation": "valid.txt", "test": "test.txt"}


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, found {len(fields)}")
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def train_relation_insertion_order(train_triples: Iterable[tuple[str, str, str]]) -> list[str]:
    """Match ``process.py``: ``unique_rel = list(set())`` over train file order."""
    relations: list[str] = []
    seen: set[str] = set()
    for _, relation, _ in train_triples:
        if relation not in seen:
            seen.add(relation)
            relations.append(relation)
    return relations


def sample_official_candidates(
    gold: str,
    relation_order: list[str],
    *,
    way: int = WAY,
) -> list[str]:
    """Consume the global ``numpy.random`` stream exactly as ``process.py`` does."""
    if gold not in relation_order:
        raise ValueError(f"gold relation absent from train vocabulary: {gold}")
    unique_rel_copy = relation_order.copy()
    unique_rel_copy.remove(gold)
    negative = np.random.permutation(unique_rel_copy)[: way - 1]
    shuffled = np.random.permutation(negative.tolist() + [gold])
    return [str(item) for item in shuffled]


def build_official_nell23k_lists(
    raw_dir: Path,
    *,
    seed: int = 2026,
    way: int = WAY,
) -> dict[str, list[list[str]]]:
    """Return ``{"validation": [...], "test": [...]}`` in source-file order."""
    train_triples = load_triples(raw_dir / "train.txt")
    relation_order = train_relation_insertion_order(train_triples)
    np.random.seed(seed)
    lists: dict[str, list[list[str]]] = {}
    for split, filename in (("validation", "valid.txt"), ("test", "test.txt")):
        triples = load_triples(raw_dir / filename)
        lists[split] = [
            sample_official_candidates(relation, relation_order, way=way)
            for _, relation, _ in triples
        ]
    return lists


def official_candidates_for_question(
    question: dict,
    official_lists: dict[str, list[list[str]]],
) -> list[str] | None:
    """Look up the paper 10-way for a RecurrentGRIP val/test question."""
    split = str(question.get("split", ""))
    if split == "train":
        return None
    if split not in official_lists:
        return None
    index = question.get("source_triple_index")
    if index is None:
        question_id = str(question.get("question_id", ""))
        try:
            index = int(question_id.rsplit(":", 1)[-1])
        except ValueError:
            return None
    rows = official_lists[split]
    if not isinstance(index, int) or index < 0 or index >= len(rows):
        return None
    return list(rows[index])


def rewrite_question_with_official_list(question: dict, candidates: list[str]) -> dict:
    """Replace the 10-way list in question text and provenance fields."""
    updated = dict(question)
    updated["candidate_relations"] = list(candidates)
    updated["question"] = QUESTION_TEMPLATE.format(
        src=question["source_node"],
        tgt=question["target_node"],
        candidates="; ".join(candidates),
    )
    updated["candidate_source"] = "official_grip_nell23k_seed2026"
    return updated


def listed_relations_from_sample(sample: dict) -> list[str]:
    """Prefer the stored 10-way, then parse the question string."""
    stored = sample.get("candidate_relations")
    if isinstance(stored, list) and stored:
        return [str(item) for item in stored]
    return parse_listed_relations(str(sample.get("question", "")))
