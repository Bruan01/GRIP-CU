"""Registered training-only evidence variants for the oracle gate."""

from __future__ import annotations

import hashlib
from typing import Iterable

from .records import build_training_answer_prompt

METHODS = (
    "answer_only",
    "more_qa_equal_token",
    "random_path_equal_token",
    "all_paths_equal_token",
    "oracle_priority_equal_token",
    "all_paths_terminal_masked_equal_token",
)
STAGE1_METHODS = METHODS[1:]


def _stable_int(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def serialize_path(candidate: dict) -> str:
    nodes = candidate["path_nodes"]
    relations = candidate["path_relations"]
    pieces = [str(nodes[0])]
    for relation, node in zip(relations, nodes[1:]):
        pieces.append(f"--{relation}--> {node}")
    return " ".join(pieces)


def _evidence_prompt(question: str, serialized_paths: Iterable[str]) -> str:
    paths = list(serialized_paths)
    lines = [
        "You are internalizing graph evidence for later graph-free reasoning.",
        "Use the selected path evidence to learn the answer, then return only the final entity identifier.",
        "",
        "Question:",
        question,
        "",
        "Candidate evidence:",
    ]
    for index, path in enumerate(paths, start=1):
        lines.append(f"Path {index}: " + path)
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def _ordered_candidates(pool: list[dict], seed: int, task_id: str) -> list[dict]:
    return sorted(pool, key=lambda candidate: _stable_int(seed, task_id, "order", candidate["source_task_id"]))


def _masked_path(candidate: dict) -> str:
    """Serialize a path without exposing its terminal entity."""
    nodes = candidate["path_nodes"]
    relations = candidate["path_relations"]
    pieces = [str(nodes[0])]
    for index, relation in enumerate(relations):
        tail = "<MASKED_TERMINAL>" if index == len(relations) - 1 else str(nodes[index + 1])
        pieces.append(f"--{relation}--> {tail}")
    return " ".join(pieces)


def _anti_copy_evidence_prompt(question: str, candidates: list[dict], seed: int, task_id: str) -> str:
    """Present paths with hidden terminals and an unassociated endpoint pool.

    The endpoint list makes the diagnostic task well-defined, while hiding the
    terminal from each path prevents a one-token terminal-to-answer copy.
    The model must associate one endpoint with the relation chain.
    """
    ordered = _ordered_candidates(candidates, seed, task_id)
    endpoints = sorted(
        (str(candidate["answer"]) for candidate in ordered),
        key=lambda answer: _stable_int(seed, task_id, "endpoint", answer),
    )
    lines = [
        "You are internalizing graph evidence for later graph-free reasoning.",
        "Match the question to the correct candidate path, infer its hidden terminal from the graph evidence, then return only the final entity identifier.",
        "",
        "Question:",
        question,
        "",
        "Candidate paths (terminal entity hidden):",
    ]
    for index, candidate in enumerate(ordered, start=1):
        lines.append(f"Path {index}: " + _masked_path(candidate))
    lines.extend(["", "Unassociated candidate terminal entities:"])
    lines.extend(f"- {answer}" for answer in endpoints)
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def build_method_supervision(rows: list[dict], pools: dict[str, list[dict]], method: str, seed: int) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    supervision: list[dict] = []
    for row in sorted(rows, key=lambda item: item["task_id"]):
        pool = pools[row["task_id"]]
        gold = next(candidate for candidate in pool if candidate["is_gold"])
        distractors = [candidate for candidate in pool if not candidate["is_gold"]]
        selected: list[dict]
        evidence_kind: str
        gold_position = -1
        if method in {"answer_only", "more_qa_equal_token"}:
            selected = []
            evidence_kind = "answer_rehearsal" if method == "more_qa_equal_token" else "answer_only"
            prompt = build_training_answer_prompt(row["text"])
        elif method == "oracle_priority_equal_token":
            selected = [gold]
            evidence_kind = "oracle_gold_path"
            gold_position = 0
            prompt = _evidence_prompt(row["text"], [serialize_path(gold)])
        elif method == "random_path_equal_token":
            index = _stable_int(seed, row["task_id"], "random_distractor") % len(distractors)
            selected = [distractors[index]]
            evidence_kind = "random_same_depth_path"
            prompt = _evidence_prompt(row["text"], [serialize_path(selected[0])])
        elif method == "all_paths_terminal_masked_equal_token":
            selected = _ordered_candidates(pool, seed, row["task_id"])
            evidence_kind = "gold_plus_distractors_terminal_masked"
            gold_position = next(index for index, candidate in enumerate(selected) if candidate["is_gold"])
            prompt = _anti_copy_evidence_prompt(row["text"], selected, seed, row["task_id"])
        else:
            selected = _ordered_candidates(pool, seed, row["task_id"])
            evidence_kind = "gold_plus_distractors_unprioritized"
            gold_position = next(index for index, candidate in enumerate(selected) if candidate["is_gold"])
            prompt = _evidence_prompt(row["text"], [serialize_path(candidate) for candidate in selected])
        supervision.append(
            {
                "task_id": row["task_id"],
                "source_split": row["split"],
                "method": method,
                "prompt": prompt,
                "answer": row["answer"],
                "depth_label": int(row["depth_label"]),
                "evidence_kind": evidence_kind,
                "candidate_count": len(selected),
                "gold_position": gold_position,
                "selected_source_task_ids": [candidate["source_task_id"] for candidate in selected],
                "selected_source_splits": [candidate["source_split"] for candidate in selected],
            }
        )
    return supervision


def supervision_audit(method_rows: dict[str, list[dict]]) -> dict:
    return {
        method: {
            "count": len(rows),
            "depth_counts": {str(depth): sum(row["depth_label"] == depth for row in rows) for depth in range(1, 5)},
            "candidate_count_mean": sum(row["candidate_count"] for row in rows) / len(rows),
            "gold_inclusion_rate": sum(row["gold_position"] >= 0 for row in rows) / len(rows),
            "train_only_selected_sources": all(
                split == "train" for row in rows for split in row["selected_source_splits"]
            ),
        }
        for method, rows in method_rows.items()
    }


def build_oracle_evidence_prompt(row: dict) -> str:
    """Build the diagnostic-only prompt that exposes the row's gold path."""
    candidate = {
        "path_nodes": row["path_nodes"],
        "path_relations": row["path_relations"],
    }
    return _evidence_prompt(row["text"], [serialize_path(candidate)])
