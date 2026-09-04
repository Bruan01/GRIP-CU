"""Exact-hop loading, graph-free prompts, and analysis annotations."""
from __future__ import annotations

from pathlib import Path

from .composition import annotate_composition, build_train_compositions, relation_counts, relation_frequency_bucket
from .io_utils import read_jsonl

REQUIRED_FIELDS = {"task_id", "text", "answer", "depth_label", "path_nodes", "path_relations", "split"}


def validate_row(row: dict, source: str) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"{source} missing fields: {sorted(missing)}")
    depth = int(row["depth_label"])
    if depth not in {1, 2, 3, 4}:
        raise ValueError(f"{source} invalid depth: {depth}")
    if len(row["path_relations"]) != depth or len(row["path_nodes"]) != depth + 1:
        raise ValueError(f"{source} path length is inconsistent with depth")
    if str(row["path_nodes"][-1]) != str(row["answer"]):
        raise ValueError(f"{source} answer differs from terminal path node")


def load_split(path: Path, expected_split: str | None = None) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty split: {path}")
    seen = set()
    for index, row in enumerate(rows):
        validate_row(row, f"{path}:{index + 1}")
        if expected_split is not None and row["split"] != expected_split:
            raise ValueError(f"split mismatch at {path}:{index + 1}")
        if row["task_id"] in seen:
            raise ValueError(f"duplicate task id: {row['task_id']}")
        seen.add(row["task_id"])
    return rows


def build_prompt(row: dict, protocol: str) -> str:
    question = str(row["text"])
    if protocol == "answer_only":
        return (
            "You are answering a graph path query.\n"
            "Follow the ordered relation chain exactly.\n"
            "Return only the final entity identifier and no explanation.\n\n"
            f"Question: {question}\nAnswer:"
        )
    if protocol == "joint_answer_slot":
        return (
            "Solve this graph-path question by composing the ordered relations. "
            "The internal trace slot is fixed and contains no graph evidence. "
            "Return only the final entity identifier after Answer.\n\n"
            f"Question:\n{question}\n\n"
            "Trace: <MASKED_INTERNAL_TRACE>\nAnswer:"
        )
    raise ValueError(f"unknown prompt protocol: {protocol}")


def _length_bucket(value: int, short_max: int, medium_max: int) -> str:
    if value <= short_max:
        return "short"
    if value <= medium_max:
        return "medium"
    return "long"


def _ambiguity_bucket(value: int) -> str:
    if value <= 1:
        return "unique"
    if value <= 4:
        return "low-ambiguity"
    if value <= 16:
        return "medium-ambiguity"
    return "high-ambiguity"


def annotate_rows(train_rows: list[dict], rows: list[dict], *, answer_token_lengths: dict[str, int], prefix_ambiguities: dict[str, int]) -> list[dict]:
    compositions = build_train_compositions(train_rows)
    frequencies = relation_counts(train_rows)
    annotated = []
    for source in rows:
        row = dict(source)
        answer = str(row["answer"])
        char_length = len(answer)
        token_length = int(answer_token_lengths[answer])
        ambiguity = int(prefix_ambiguities[answer])
        row.update({
            "composition": annotate_composition(row, compositions),
            "relation_frequency_bucket": relation_frequency_bucket(row["path_relations"], frequencies),
            "answer_character_length": char_length,
            "answer_character_length_bucket": _length_bucket(char_length, 30, 55),
            "answer_token_length": token_length,
            "answer_token_length_bucket": _length_bucket(token_length, 8, 16),
            "entity_prefix_ambiguity": ambiguity,
            "entity_prefix_ambiguity_bucket": _ambiguity_bucket(ambiguity),
        })
        annotated.append(row)
    return annotated
