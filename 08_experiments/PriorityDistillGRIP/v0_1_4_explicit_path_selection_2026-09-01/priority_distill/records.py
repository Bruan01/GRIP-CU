"""Strict exact-hop records and graph-free prompt boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REQUIRED_FIELDS = {
    "task_id",
    "text",
    "answer",
    "depth_label",
    "depth_source",
    "path_nodes",
    "path_relations",
    "path_edges",
    "split",
    "shortest_path_verified",
    "unique_relation_chain_answer",
}


def validate_exact_hop_row(row: dict, source: str = "record") -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"{source} missing fields: {sorted(missing)}")
    depth = int(row["depth_label"])
    if depth not in {1, 2, 3, 4}:
        raise ValueError(f"{source} invalid depth {depth}")
    if not row["shortest_path_verified"] or not row["unique_relation_chain_answer"]:
        raise ValueError(f"{source} is not a strict exact-hop task")
    if len(row["path_relations"]) != depth:
        raise ValueError(f"{source} path_relations length does not equal depth")
    if len(row["path_nodes"]) != depth + 1:
        raise ValueError(f"{source} path_nodes length does not equal depth + 1")
    if len(row["path_edges"]) != depth:
        raise ValueError(f"{source} path_edges length does not equal depth")
    if str(row["path_nodes"][-1]) != str(row["answer"]):
        raise ValueError(f"{source} final path node does not equal answer")


def load_exact_hop_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_exact_hop_row(row, f"{path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    ids_by_split: dict[str, set[str]] = {}
    for split, rows in splits.items():
        ids: set[str] = set()
        for index, row in enumerate(rows):
            validate_exact_hop_row(row, f"{split}[{index}]")
            if row["split"] != split:
                raise ValueError(f"split label mismatch: {row['task_id']} says {row['split']} but loaded as {split}")
            if row["task_id"] in ids:
                raise ValueError(f"duplicate task id in {split}: {row['task_id']}")
            ids.add(row["task_id"])
        ids_by_split[split] = ids
    overlaps: dict[str, list[str]] = {}
    names = sorted(ids_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(ids_by_split[left] & ids_by_split[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps:
        raise ValueError(f"task-id leakage across splits: {overlaps}")
    return {
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "depth_counts": {
            split: {str(depth): sum(int(row["depth_label"]) == depth for row in rows) for depth in range(1, 5)}
            for split, rows in splits.items()
        },
        "task_id_overlaps": overlaps,
    }


def load_splits(repo_root: Path, data_config: dict) -> tuple[dict[str, list[dict]], dict[str, Path], dict]:
    paths = {split: repo_root / data_config[split] for split in ("train", "validation", "test")}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} split: {path}")
    splits = {split: load_exact_hop_jsonl(path) for split, path in paths.items()}
    audit = validate_splits(splits)
    for split, expected in data_config.get("expected_sizes", {}).items():
        if len(splits[split]) != int(expected):
            raise ValueError(f"{split} size changed: expected {expected}, found {len(splits[split])}")
    return splits, paths, audit


def build_training_answer_prompt(question: str) -> str:
    return (
        "Internalize the answer to this graph-path question into model parameters.\n"
        "Return only the final entity identifier.\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def build_evaluation_prompt(question: str) -> str:
    return (
        "You are answering a graph path query.\n"
        "Follow the ordered relation chain exactly.\n"
        "Return only the final entity identifier and no explanation.\n\n"
        f"Question: {question}\nAnswer:"
    )


def records_by_depth(rows: Iterable[dict]) -> dict[int, list[dict]]:
    grouped = {depth: [] for depth in range(1, 5)}
    for row in rows:
        grouped[int(row["depth_label"])].append(row)
    return grouped
