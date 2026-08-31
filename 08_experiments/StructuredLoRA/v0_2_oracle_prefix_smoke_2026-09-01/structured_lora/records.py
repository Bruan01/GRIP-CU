"""Exact-hop record loading and prompt construction."""

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
    "split",
    "shortest_path_verified",
    "unique_relation_chain_answer",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            if row["depth_label"] not in {1, 2, 3, 4}:
                raise ValueError(f"{path}:{line_number} invalid depth {row['depth_label']}")
            if not row["shortest_path_verified"] or not row["unique_relation_chain_answer"]:
                raise ValueError(f"{path}:{line_number} is not a strict exact-hop task")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    ids_by_split = {name: {row["task_id"] for row in rows} for name, rows in splits.items()}
    overlaps: dict[str, list[str]] = {}
    names = sorted(ids_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(ids_by_split[left] & ids_by_split[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps:
        raise ValueError(f"task-id leakage across splits: {overlaps}")
    depth_counts = {
        split: {str(depth): sum(row["depth_label"] == depth for row in rows) for depth in range(1, 5)}
        for split, rows in splits.items()
    }
    return {
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "depth_counts": depth_counts,
        "overlaps": overlaps,
    }


def build_prompt(question: str) -> str:
    return (
        "You are answering a graph path query. Follow the ordered relation chain exactly.\n"
        "Return only the final entity identifier and no explanation.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def records_by_depth(rows: Iterable[dict]) -> dict[int, list[dict]]:
    grouped = {depth: [] for depth in range(1, 5)}
    for row in rows:
        grouped[int(row["depth_label"])].append(row)
    return grouped
