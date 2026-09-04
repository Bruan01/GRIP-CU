#!/usr/bin/env python3
"""Turn teacher-forced candidate scores into sequence-level predictions.

This is exact selection over the four supplied candidate strings, not
open-vocabulary generation.  It makes the existing ranking diagnostic usable
as a reproducible candidate-set decoder and writes task-level predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def select(rows: list[dict], condition: str = "graph_free") -> list[dict]:
    selected = [row for row in rows if row.get("condition") == condition]
    if not selected:
        raise ValueError(f"no rows found for condition={condition}")
    predictions = []
    seen: set[str] = set()
    for row in tqdm(selected, desc=f"select sequence candidates {condition}", unit="query"):
        task_id = row.get("task_id")
        if not task_id or task_id in seen:
            raise ValueError(f"duplicate or missing task_id: {task_id}")
        seen.add(task_id)
        candidates = row.get("candidate_scores")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"expected four candidate scores: {task_id}")
        if any(not isinstance(item.get("answer"), str) for item in candidates):
            raise ValueError(f"candidate answer missing: {task_id}")
        if len({item["answer"] for item in candidates}) != 4:
            raise ValueError(f"candidate answer strings are not unique: {task_id}")
        gold = [item for item in candidates if item.get("is_gold")]
        if len(gold) != 1:
            raise ValueError(f"expected exactly one gold candidate: {task_id}")
        ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        prediction = ranked[0]
        gold_answer = gold[0]["answer"]
        predictions.append({
            "task_id": task_id,
            "depth_label": int(row["depth_label"]),
            "condition": condition,
            "answer": row["answer"],
            "prediction_answer": prediction["answer"],
            "prediction_score": float(prediction["score"]),
            "gold_score": float(gold[0]["score"]),
            "rank": int(row.get("rank", next(i for i, item in enumerate(ranked, start=1) if item.get("is_gold")))),
            "sequence_exact_match": prediction["answer"] == gold_answer,
            "candidate_answers": [item["answer"] for item in ranked],
            "candidate_scores": ranked,
        })
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="candidate_ranking_rows.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", default="graph_free")
    args = parser.parse_args()
    predictions = select(load_jsonl(args.input), args.condition)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")
    correct = sum(bool(row["sequence_exact_match"]) for row in predictions)
    print(json.dumps({"condition": args.condition, "count": len(predictions), "correct": correct, "accuracy": correct / len(predictions)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
