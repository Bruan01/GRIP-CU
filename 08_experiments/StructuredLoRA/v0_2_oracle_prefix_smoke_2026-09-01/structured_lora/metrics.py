"""Pure evaluation and gate metrics."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Mapping

ENTITY_PATTERN = re.compile(r"concept_[A-Za-z0-9_:-]+")


def normalize_entity(text: str) -> str:
    match = ENTITY_PATTERN.search(text.strip())
    if match:
        return match.group(0).rstrip(".,;:!?)]}\"")
    return text.strip().splitlines()[0].strip().strip("`'\".,;:!?()[]{}") if text.strip() else ""


def score_predictions(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    by_depth: dict[int, list[int]] = defaultdict(list)
    correct_values: list[int] = []
    for row in rows:
        gold = normalize_entity(str(row["answer"]))
        prediction = normalize_entity(str(row["prediction_text"]))
        correct = int(prediction == gold)
        correct_values.append(correct)
        by_depth[int(row["depth_label"])].append(correct)
    depth_accuracy = {
        str(depth): (sum(by_depth[depth]) / len(by_depth[depth]) if by_depth[depth] else None)
        for depth in range(1, 5)
    }
    observed = [value for value in depth_accuracy.values() if value is not None]
    return {
        "count": len(rows),
        "correct": sum(correct_values),
        "accuracy": sum(correct_values) / len(rows),
        "accuracy_by_depth": depth_accuracy,
        "macro_depth_accuracy": sum(observed) / len(observed),
        "worst_depth_accuracy": min(observed),
    }


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left = list(left)
    right = list(right)
    if len(left) != len(right):
        raise ValueError("cosine vectors must have equal length")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)
