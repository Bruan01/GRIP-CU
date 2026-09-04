#!/usr/bin/env python3
"""Break held-out predictions down by relation-chain composition novelty.

The split is deliberately post-hoc: it labels each evaluation task according to
whether its complete ordered relation tuple occurred in the training split.
The script performs strict task-id/condition validation so a partial or mixed
prediction file cannot silently produce an optimistic number.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relation_composition(row: dict) -> tuple[str, ...]:
    relations = row.get("path_relations")
    if not isinstance(relations, list) or not relations:
        raise ValueError(f"row has invalid path_relations: {row.get('task_id')}")
    return tuple(str(relation) for relation in relations)


def _unique_task_rows(rows: Iterable[dict], *, label: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            raise ValueError(f"{label} row is missing task_id")
        if task_id in by_id:
            raise ValueError(f"duplicate task_id in {label}: {task_id}")
        by_id[task_id] = row
    return by_id


def summarize(items: list[dict]) -> dict:
    by_depth = {}
    for depth in range(1, 5):
        values = [item["correct"] for item in items if item["depth"] == depth]
        by_depth[str(depth)] = {
            "correct": sum(values),
            "count": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
    values = [item["correct"] for item in items]
    return {
        "correct": sum(values),
        "count": len(values),
        "accuracy": sum(values) / len(values) if values else None,
        "by_depth": by_depth,
    }


def analyze(
    train_rows: list[dict],
    eval_rows: list[dict],
    prediction_rows: list[dict],
    *,
    prediction_kind: str,
    condition: str = "graph_free",
) -> dict:
    if prediction_kind not in {"constrained", "ranking"}:
        raise ValueError(f"unsupported prediction_kind: {prediction_kind}")
    eval_by_id = _unique_task_rows(eval_rows, label="eval")
    train_by_id = _unique_task_rows(train_rows, label="train")
    if set(eval_by_id) & set(train_by_id):
        raise ValueError("train/eval task_id overlap")
    seen_compositions = {relation_composition(row) for row in train_rows}

    selected: list[dict] = []
    for prediction in prediction_rows:
        task_id = prediction.get("task_id")
        if not task_id:
            raise ValueError("prediction row is missing task_id")
        if prediction_kind == "ranking":
            row_condition = prediction.get("condition")
            if row_condition != condition:
                continue
        elif prediction.get("condition") not in (None, ""):
            raise ValueError(f"constrained prediction unexpectedly has condition: {task_id}")
        if task_id not in eval_by_id:
            raise ValueError(f"prediction task_id not found in eval split: {task_id}")
        selected.append(prediction)

    selected_by_id = _unique_task_rows(selected, label=f"{prediction_kind} predictions ({condition})")
    missing = sorted(set(eval_by_id) - set(selected_by_id))
    extra = sorted(set(selected_by_id) - set(eval_by_id))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} (first: {missing[:3]})")
        if extra:
            details.append(f"extra={len(extra)} (first: {extra[:3]})")
        raise ValueError(f"prediction/eval task coverage mismatch: {'; '.join(details)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for task_id, row in eval_by_id.items():
        prediction = selected_by_id[task_id]
        if prediction_kind == "constrained":
            candidate_answers = prediction.get("candidate_answers")
            if not isinstance(candidate_answers, list) or len(candidate_answers) != 4:
                raise ValueError(f"constrained row must contain exactly 4 candidates: {task_id}")
            if len({str(answer) for answer in candidate_answers}) != 4:
                raise ValueError(f"constrained candidate answers are not unique: {task_id}")
            if not prediction.get("gold_in_candidate_set", False):
                raise ValueError(f"gold answer is not in candidate set: {task_id}")
            correct = int(bool(prediction.get("constrained_exact_match", False)))
        else:
            rank = prediction.get("rank")
            if not isinstance(rank, int) or rank < 1:
                raise ValueError(f"ranking row has invalid rank: {task_id}")
            correct = int(rank == 1)
        group = "seen_composition" if relation_composition(row) in seen_compositions else "novel_composition"
        grouped[group].append({"correct": correct, "depth": int(row["depth_label"])})

    return {
        "format_version": 2,
        "purpose": "seen_vs_novel_relation_chain_composition_audit",
        "prediction_kind": prediction_kind,
        "condition": condition if prediction_kind == "ranking" else None,
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "train_composition_count": len(seen_compositions),
        "eval_composition_count": len({relation_composition(row) for row in eval_rows}),
        "prediction_count": len(selected),
        "groups": {group: summarize(items) for group, items in sorted(grouped.items())},
        "definition": "seen_composition means the exact ordered relation tuple occurred in train; novel_composition means it did not",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True, help="exact-hop split JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-kind", choices=("constrained", "ranking"), required=True)
    parser.add_argument("--condition", default="graph_free", help="filter a multi-condition prediction JSONL")
    args = parser.parse_args()

    report = analyze(
        load_jsonl(args.train),
        load_jsonl(args.eval),
        load_jsonl(args.predictions),
        prediction_kind=args.prediction_kind,
        condition=args.condition,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
