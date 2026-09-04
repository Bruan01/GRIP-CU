"""Exact-entity and explicit candidate-selection metrics."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping


_SELECTION_RE = re.compile(r"selected\s*path\s*:\s*(\d+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*:\s*([^\n\r]+)", re.IGNORECASE)


def normalize_entity(text: str) -> str:
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = text.strip("`'\".,;:()[]{}")
    if " " in text:
        text = text.split()[0]
    return text.lower()



def parse_answer_output(text: str) -> str:
    """Extract an answer from either answer-only or structured-trace output."""
    value = str(text)
    match = _ANSWER_RE.search(value)
    if match:
        return match.group(1).strip()
    return value.strip()

def parse_explicit_output(text: str) -> dict:
    """Parse the two-line explicit target, tolerating casing and extra text."""
    selection_match = _SELECTION_RE.search(str(text))
    answer_match = _ANSWER_RE.search(str(text))
    answer = answer_match.group(1).strip() if answer_match else ""
    return {
        "selected_path": int(selection_match.group(1)) if selection_match else None,
        "answer": answer,
    }


def _accuracy_summary(correct_values: list[int], by_depth: dict[int, list[int]]) -> dict:
    depth_accuracy = {
        str(depth): (sum(by_depth[depth]) / len(by_depth[depth]) if by_depth[depth] else None)
        for depth in range(1, 5)
    }
    observed = [value for value in depth_accuracy.values() if value is not None]
    deep_values = by_depth[3] + by_depth[4]
    return {
        "count": len(correct_values),
        "correct": sum(correct_values),
        "accuracy": sum(correct_values) / len(correct_values) if correct_values else 0.0,
        "accuracy_by_depth": depth_accuracy,
        "macro_depth_accuracy": sum(observed) / len(observed) if observed else 0.0,
        "worst_depth_accuracy": min(observed) if observed else 0.0,
        "deep_3_4_accuracy": sum(deep_values) / len(deep_values) if deep_values else None,
    }


def score_predictions(rows: Iterable[Mapping]) -> dict:
    """Score graph-free predictions, using parsed answer when available."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    by_depth: dict[int, list[int]] = defaultdict(list)
    correct_values: list[int] = []
    for row in rows:
        prediction = row.get("prediction_answer", row["prediction_text"])
        correct = int(normalize_entity(str(prediction)) == normalize_entity(str(row["answer"])))
        correct_values.append(correct)
        by_depth[int(row["depth_label"])].append(correct)
    return _accuracy_summary(correct_values, by_depth)


def score_candidate_selection_predictions(rows: Iterable[Mapping]) -> dict:
    """Score path selection, answer, and their conjunction."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty candidate-selection set")
    selection_by_depth: dict[int, list[int]] = defaultdict(list)
    answer_by_depth: dict[int, list[int]] = defaultdict(list)
    joint_by_depth: dict[int, list[int]] = defaultdict(list)
    selection_correct: list[int] = []
    answer_correct: list[int] = []
    joint_correct: list[int] = []
    missing_selection = 0
    missing_answer = 0
    for row in rows:
        parsed = parse_explicit_output(str(row.get("prediction_text", "")))
        predicted_selection = row.get("predicted_selection", parsed["selected_path"])
        predicted_answer = row.get("prediction_answer", parsed["answer"])
        expected_selection = row.get("selection_target", row.get("gold_position", -1) + 1)
        s_ok = int(predicted_selection is not None and int(predicted_selection) == int(expected_selection))
        a_ok = int(normalize_entity(str(predicted_answer)) == normalize_entity(str(row["answer"])))
        j_ok = s_ok * a_ok
        if predicted_selection is None:
            missing_selection += 1
        if not str(predicted_answer).strip():
            missing_answer += 1
        depth = int(row["depth_label"])
        selection_correct.append(s_ok); answer_correct.append(a_ok); joint_correct.append(j_ok)
        selection_by_depth[depth].append(s_ok); answer_by_depth[depth].append(a_ok); joint_by_depth[depth].append(j_ok)
    return {
        "selection": _accuracy_summary(selection_correct, selection_by_depth),
        "answer": _accuracy_summary(answer_correct, answer_by_depth),
        "joint": _accuracy_summary(joint_correct, joint_by_depth),
        "missing_selection": missing_selection,
        "missing_answer": missing_answer,
        "random_selection_baseline": 1.0 / max(1, int(rows[0].get("candidate_count", 4))),
    }
