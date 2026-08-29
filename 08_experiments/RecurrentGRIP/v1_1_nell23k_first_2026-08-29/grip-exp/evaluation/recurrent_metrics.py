from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable

from evaluation.utils import normalize_answer


def exact_match(prediction: str, targets: Iterable[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(str(target)) for target in targets)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denominator = sqrt(
        sum((x - mean_x) ** 2 for x in rx) * sum((y - mean_y) ** 2 for y in ry)
    )
    return numerator / denominator if denominator else 0.0


def _known_hop(row: dict) -> int | None:
    value = row.get("true_hop")
    if value is None:
        return None
    hop = int(value)
    return hop if hop >= 1 else None


def summarize_recurrent_predictions(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "correct_adapter_accuracy": 0.0,
            "known_hop_count": 0,
            "unknown_hop_count": 0,
            "best_k_true_hop_spearman": 0.0,
            "solved_question_count": 0,
            "unsolved_question_count": 0,
            "by_hop_and_k": {},
            "by_k_and_adapter": {},
            "by_split_k_and_adapter": {},
        }

    hop_buckets: dict[tuple[int, int, str], list[bool]] = defaultdict(list)
    depth_buckets: dict[tuple[int, str], list[bool]] = defaultdict(list)
    split_depth_buckets: dict[tuple[str, int, str], list[bool]] = defaultdict(list)
    known_hop_count = 0
    for row in rows:
        depth = int(row["recurrence_k"])
        control = row.get("adapter_control", "correct")
        correct = bool(row["correct"])
        depth_buckets[(depth, control)].append(correct)
        split = str(row.get("metadata", {}).get("split", "unknown"))
        split_depth_buckets[(split, depth, control)].append(correct)
        hop = _known_hop(row)
        if hop is not None:
            known_hop_count += 1
            hop_buckets[(hop, depth, control)].append(correct)

    by_hop_and_k = {
        f"hop={hop}|k={depth}|adapter={control}": {
            "count": len(values),
            "accuracy": sum(values) / len(values),
        }
        for (hop, depth, control), values in sorted(hop_buckets.items())
    }
    by_k_and_adapter = {
        f"k={depth}|adapter={control}": {
            "count": len(values),
            "accuracy": sum(values) / len(values),
        }
        for (depth, control), values in sorted(depth_buckets.items())
    }

    by_split_k_and_adapter = {
        f"split={split}|k={depth}|adapter={control}": {
            "count": len(values),
            "accuracy": sum(values) / len(values),
        }
        for (split, depth, control), values in sorted(split_depth_buckets.items())
    }

    correct_rows = [row for row in rows if row.get("adapter_control", "correct") == "correct"]
    question_ids = {row["question_id"] for row in correct_rows}
    successful_depths: dict[str, list[int]] = defaultdict(list)
    hop_lookup: dict[str, int] = {}
    for row in correct_rows:
        question_id = row["question_id"]
        hop = _known_hop(row)
        if hop is not None:
            hop_lookup[question_id] = hop
        if bool(row["correct"]):
            successful_depths[question_id].append(int(row["recurrence_k"]))

    best_depth_by_question = {
        question_id: min(depths)
        for question_id, depths in successful_depths.items()
        if depths
    }
    solved_ids = sorted(best_depth_by_question)
    hop_solved_ids = [question_id for question_id in solved_ids if question_id in hop_lookup]
    rho = spearman_correlation(
        [hop_lookup[question_id] for question_id in hop_solved_ids],
        [best_depth_by_question[question_id] for question_id in hop_solved_ids],
    )
    return {
        "count": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "correct_adapter_accuracy": (
            sum(bool(row["correct"]) for row in correct_rows) / len(correct_rows)
            if correct_rows
            else 0.0
        ),
        "known_hop_count": known_hop_count,
        "unknown_hop_count": len(rows) - known_hop_count,
        "best_k_true_hop_spearman": rho,
        "solved_question_count": len(solved_ids),
        "unsolved_question_count": len(question_ids) - len(solved_ids),
        "by_hop_and_k": by_hop_and_k,
        "by_k_and_adapter": by_k_and_adapter,
        "by_split_k_and_adapter": by_split_k_and_adapter,
    }
