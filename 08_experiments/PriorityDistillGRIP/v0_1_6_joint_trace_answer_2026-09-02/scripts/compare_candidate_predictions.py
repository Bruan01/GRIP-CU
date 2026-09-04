#!/usr/bin/env python3
"""Compare two candidate-set prediction files task by task."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        task_id = row.get("task_id")
        if not task_id or task_id in result:
            raise ValueError(f"duplicate or missing task_id: {task_id}")
        result[task_id] = row
    return result


def load_eval(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row["task_id"]
        if task_id in result:
            raise ValueError(f"duplicate eval task_id: {task_id}")
        result[task_id] = row
    return result


def summarize(items: list[dict]) -> dict:
    direct = [item["direct"] for item in items]
    other = [item["other"] for item in items]
    return {
        "count": len(items),
        "direct_correct": sum(direct),
        "other_correct": sum(other),
        "direct_accuracy": sum(direct) / len(items) if items else None,
        "other_accuracy": sum(other) / len(items) if items else None,
        "other_wins": sum((not d) and o for d, o in zip(direct, other)),
        "direct_wins": sum(d and (not o) for d, o in zip(direct, other)),
        "both_correct": sum(d and o for d, o in zip(direct, other)),
        "both_wrong": sum((not d) and (not o) for d, o in zip(direct, other)),
    }


def compare(direct_rows: dict[str, dict], other_rows: dict[str, dict], eval_rows: dict[str, dict], train_compositions: set[tuple[str, ...]], *, direct_name: str, other_name: str, prediction_kind: str) -> dict:
    if prediction_kind not in {"ranking", "constrained"}:
        raise ValueError(f"unsupported prediction_kind: {prediction_kind}")
    if set(direct_rows) != set(other_rows) or set(direct_rows) != set(eval_rows):
        raise ValueError("prediction files and eval split must have identical task coverage")
    groups: dict[str, list[dict]] = defaultdict(list)
    for task_id, direct in direct_rows.items():
        other = other_rows[task_id]
        if direct.get("candidate_answers") and other.get("candidate_answers"):
            if set(direct["candidate_answers"]) != set(other["candidate_answers"]):
                raise ValueError(f"candidate pool mismatch: {task_id}")
        row = eval_rows[task_id]
        composition = tuple(str(value) for value in row["path_relations"])
        group = "seen_composition" if composition in train_compositions else "novel_composition"
        if prediction_kind == "ranking":
            direct_correct = int(direct.get("rank", 0) == 1)
            other_correct = int(other.get("rank", 0) == 1)
        else:
            direct_correct = int(bool(direct.get("constrained_exact_match", False)))
            other_correct = int(bool(other.get("constrained_exact_match", False)))
        groups[group].append({"direct": direct_correct, "other": other_correct})
    all_items = [item for values in groups.values() for item in values]
    return {
        "format_version": 1,
        "direct_name": direct_name,
        "other_name": other_name,
        "overall": summarize(all_items),
        "by_composition": {group: summarize(items) for group, items in sorted(groups.items())},
        "interpretation": "paired counts are descriptive; this is a 4-way candidate-set diagnostic, not full-vocabulary accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-name", default="direct")
    parser.add_argument("--prediction-kind", choices=("ranking", "constrained"), default="ranking")
    parser.add_argument("--other-name", default="other")
    args = parser.parse_args()
    train_rows = [json.loads(line) for line in args.train.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    train_compositions = {tuple(str(value) for value in row["path_relations"]) for row in train_rows}
    eval_rows = load_eval(args.eval)
    report = compare(load(args.direct), load(args.other), eval_rows, train_compositions, direct_name=args.direct_name, other_name=args.other_name, prediction_kind=args.prediction_kind)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
