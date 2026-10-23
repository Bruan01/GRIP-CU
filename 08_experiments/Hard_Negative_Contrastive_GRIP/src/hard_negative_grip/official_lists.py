"""Replay the official GRIP NELL23K 10-way candidate protocol.

The paper processor is ``data/raw_datasets/nell23k/process.py``, invoked by
``scripts/process_raw_data.py --datasets nell23k --seed 2026``. It seeds global
``numpy.random`` once, then for every ``valid.txt`` triple and then every
``test.txt`` triple samples 9 negatives from the train-relation insertion order
and shuffles them with the gold relation. Train questions have no paper 10-way
list.

The candidate list is part of the standard GRIP relation-prediction prompt.
This module rebuilds the frozen paper lists when a task file does not already
store them.
"""

from __future__ import annotations

import json
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
DEFAULT_RAW_NELL23K = (
    Path(__file__).resolve().parents[4]
    / "13_base_method"
    / "grip-exp"
    / "data"
    / "raw_datasets"
    / "nell23k"
)


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


def load_relation_text_mapping(path: Path | None) -> dict[str, str]:
    """Load an optional raw-relation to task-label mapping.

    FB15K237-2 stores relation IDs in triples but emits human-readable
    relation descriptions in task answers. Keeping this conversion explicit
    prevents raw IDs and answer labels from being mixed during filtering.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"relation text mapping must be an object: {path}")
    mapping = {str(key): str(value).strip() for key, value in payload.items()}
    if any(not value for value in mapping.values()):
        raise ValueError("relation text mapping must contain non-empty strings")
    return mapping


def load_train_relation_order(
    raw_dir: Path, *, relation_text_file: Path | None = None
) -> list[str]:
    """Train-only vocabulary; an explicit mapping enables FB text relations.

    NELL keeps its original insertion order. Text relations use sorted order,
    matching the deterministic FB15K-237 preparation protocol.
    """
    relations = train_relation_insertion_order(load_triples(raw_dir / "train.txt"))
    if relation_text_file is None:
        # WN18RR stores relation IDs as ``_relation_name`` while the released
        # GRIP task files expose the human-readable label ``relation name``.
        # Keep NELL/other raw IDs unchanged, but canonicalize this explicit
        # WN-style encoding so task answers and candidate continuations share
        # one vocabulary.
        return [
            " ".join(value.split("_")[1:]).strip()
            if value.startswith("_") and "_" in value
            else value
            for value in relations
        ]
    # The mapping must cover every train relation before we build the frozen
    # candidate vocabulary; silently dropping an ID would change the protocol.
    mapping = load_relation_text_mapping(relation_text_file)
    missing = [relation for relation in relations if relation not in mapping]
    if missing:
        raise ValueError(
            f"relation text mapping is missing {len(missing)} train relations; "
            f"examples={missing[:3]}"
        )
    mapped = [mapping[relation] for relation in relations]
    if any(not isinstance(value, str) or not value.strip() for value in mapped):
        raise ValueError("relation text mapping must contain non-empty strings")
    # The official FB processor samples a set of relation TEXTS. Multiple raw
    # relation IDs can share one answer string (237 IDs -> 233 texts).
    return sorted(set(mapped))


def sample_official_candidates(
    gold: str,
    relation_order: list[str],
    *,
    way: int = WAY,
    rng: np.random.RandomState | None = None,
) -> list[str]:
    """Sample a 10-way list the way ``process.py`` does.

    Eval uses the global ``numpy.random`` stream after ``seed=2026``. Training
    listed negatives pass a private ``RandomState`` so they reuse the same
    permutation rule without consuming the eval stream or val/test triples.
    """
    if gold not in relation_order:
        raise ValueError(f"gold relation absent from train vocabulary: {gold}")
    stream = np.random if rng is None else rng
    unique_rel_copy = relation_order.copy()
    unique_rel_copy.remove(gold)
    negative = stream.permutation(unique_rel_copy)[: way - 1]
    shuffled = stream.permutation(negative.tolist() + [gold])
    return [str(item) for item in shuffled]


def official_negatives(
    gold: str,
    relation_order: list[str],
    *,
    way: int = WAY,
    rng: np.random.RandomState | None = None,
) -> list[str]:
    """Return the 9 non-gold relations from ``sample_official_candidates``."""
    return [rel for rel in sample_official_candidates(gold, relation_order, way=way, rng=rng) if rel != gold]


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
    """Look up the paper 10-way for a standard GRIP val/test question."""
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
    question = sample.get("question", "")
    if isinstance(question, (list, tuple)) and question:
        question = question[0]
    return parse_listed_relations(str(question))
