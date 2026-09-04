"""Graph-free exact-entity evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def normalize_entity(text: str) -> str:
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = text.strip("`'\".,;:()[]{}")
    if " " in text:
        text = text.split()[0]
    return text.lower()


def score_predictions(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    by_depth: dict[int, list[int]] = defaultdict(list)
    correct_values: list[int] = []
    for row in rows:
        correct = int(normalize_entity(str(row["prediction_text"])) == normalize_entity(str(row["answer"])))
        correct_values.append(correct)
        by_depth[int(row["depth_label"])].append(correct)
    depth_accuracy = {
        str(depth): (sum(by_depth[depth]) / len(by_depth[depth]) if by_depth[depth] else None)
        for depth in range(1, 5)
    }
    observed = [value for value in depth_accuracy.values() if value is not None]
    deep_values = by_depth[3] + by_depth[4]
    return {
        "count": len(rows),
        "correct": sum(correct_values),
        "accuracy": sum(correct_values) / len(rows),
        "accuracy_by_depth": depth_accuracy,
        "macro_depth_accuracy": sum(observed) / len(observed),
        "worst_depth_accuracy": min(observed),
        "deep_3_4_accuracy": sum(deep_values) / len(deep_values) if deep_values else None,
    }
