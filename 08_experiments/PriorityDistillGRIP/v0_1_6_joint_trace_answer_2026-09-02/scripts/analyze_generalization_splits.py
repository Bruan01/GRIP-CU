#!/usr/bin/env python3
"""Audit candidate predictions on stricter, orthogonal generalization splits.

This is a post-hoc diagnostic.  It does not change model outputs.  Every
prediction must cover exactly one evaluation task, and every split is defined
only from the training/evaluation records (never from predictions).
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


def _relations(row: dict) -> tuple[str, ...]:
    values = row.get("path_relations")
    if not isinstance(values, list) or not values:
        raise ValueError(f"row has invalid path_relations: {row.get('task_id')}")
    return tuple(str(value) for value in values)


def _nodes(row: dict) -> list[str]:
    values = row.get("path_nodes")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"row has invalid path_nodes: {row.get('task_id')}")
    return [str(value) for value in values]


def _correct(prediction: dict, prediction_kind: str) -> int:
    if prediction_kind == "constrained":
        candidates = prediction.get("candidate_answers")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"constrained row must contain exactly 4 candidates: {prediction.get('task_id')}")
        if len({str(value) for value in candidates}) != 4:
            raise ValueError(f"constrained candidate answers are not unique: {prediction.get('task_id')}")
        if not prediction.get("gold_in_candidate_set", False):
            raise ValueError(f"gold answer is not in candidate set: {prediction.get('task_id')}")
        return int(bool(prediction.get("constrained_exact_match", False)))
    rank = prediction.get("rank")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"ranking row has invalid rank: {prediction.get('task_id')}")
    return int(rank == 1)


def _summarize(items: list[dict]) -> dict:
    values = [int(item["correct"]) for item in items]
    by_depth = {}
    for depth in range(1, 5):
        subset = [item["correct"] for item in items if item["depth"] == depth]
        by_depth[str(depth)] = {
            "correct": sum(subset),
            "count": len(subset),
            "accuracy": sum(subset) / len(subset) if subset else None,
        }
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
    train_by_id = _unique_task_rows(train_rows, label="train")
    eval_by_id = _unique_task_rows(eval_rows, label="eval")
    if set(train_by_id) & set(eval_by_id):
        raise ValueError("train/eval task_id overlap")

    selected = []
    for prediction in prediction_rows:
        task_id = prediction.get("task_id")
        if not task_id:
            raise ValueError("prediction row is missing task_id")
        if prediction_kind == "ranking":
            if prediction.get("condition") != condition:
                continue
        elif prediction.get("condition") not in (None, ""):
            raise ValueError(f"constrained prediction unexpectedly has condition: {task_id}")
        if task_id not in eval_by_id:
            raise ValueError(f"prediction task_id not found in eval split: {task_id}")
        selected.append(prediction)
    prediction_by_id = _unique_task_rows(selected, label=f"{prediction_kind} predictions ({condition})")
    missing = sorted(set(eval_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(eval_by_id))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if extra:
            details.append(f"extra={len(extra)}")
        raise ValueError(f"prediction/eval task coverage mismatch: {'; '.join(details)}")

    train_answers = {str(row["answer"]) for row in train_rows}
    train_heads = { _nodes(row)[0] for row in train_rows }
    train_relations = {relation for row in train_rows for relation in _relations(row)}
    train_compositions = {_relations(row) for row in train_rows}

    axes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for task_id, row in eval_by_id.items():
        prediction = prediction_by_id[task_id]
        relations = _relations(row)
        nodes = _nodes(row)
        answer = str(row["answer"])
        item = {"correct": _correct(prediction, prediction_kind), "depth": int(row["depth_label"])}
        composition_seen = relations in train_compositions
        all_relations_seen = all(relation in train_relations for relation in relations)
        axes["answer_seen_in_train"]["seen" if answer in train_answers else "unseen"].append(item)
        axes["head_seen_in_train"]["seen" if nodes[0] in train_heads else "unseen"].append(item)
        axes["composition"]["seen" if composition_seen else "novel"].append(item)
        axes["all_relations_seen_individually"]["yes" if all_relations_seen else "no"].append(item)
        if not composition_seen:
            axes["novel_composition_relation_coverage"]["all_relations_seen" if all_relations_seen else "contains_unseen_relation"].append(item)

    return {
        "format_version": 1,
        "purpose": "strict_generalization_split_audit",
        "prediction_kind": prediction_kind,
        "condition": condition if prediction_kind == "ranking" else None,
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "prediction_count": len(selected),
        "train_answer_count": len(train_answers),
        "train_head_count": len(train_heads),
        "train_relation_count": len(train_relations),
        "train_composition_count": len(train_compositions),
        "axes": {
            axis: {group: _summarize(items) for group, items in sorted(groups.items())}
            for axis, groups in sorted(axes.items())
        },
        "definitions": {
            "answer_seen_in_train": "gold answer string occurred in a training row",
            "head_seen_in_train": "starting entity occurred as a training path head",
            "composition": "complete ordered relation tuple occurred in train",
            "all_relations_seen_individually": "every relation in the ordered tuple occurred somewhere in train, regardless of order",
            "novel_composition_relation_coverage": "only novel-composition rows, split by whether all component relations were individually seen",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-kind", choices=("constrained", "ranking"), required=True)
    parser.add_argument("--condition", default="graph_free")
    args = parser.parse_args()
    report = analyze(
        load_jsonl(args.train), load_jsonl(args.eval), load_jsonl(args.predictions),
        prediction_kind=args.prediction_kind, condition=args.condition,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
