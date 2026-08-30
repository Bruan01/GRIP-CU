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


def _train_k(row: dict) -> int:
    value = row.get("recurrent_train_k")
    if value is None:
        value = row.get("metadata", {}).get("recurrent_train_k", 0)
    return int(value or 0)


def _accuracy_bucket(values: list[bool]) -> dict:
    return {"count": len(values), "accuracy": sum(values) / len(values)}


def _quality_stats(rows: list[dict]) -> dict:
    count = len(rows)
    empty_count = sum(not str(row.get("response", "")).strip() for row in rows)

    candidate_values: list[bool] = []
    for row in rows:
        value = row.get("response_in_candidates")
        if value is None:
            candidates = row.get("metadata", {}).get("candidate_relations")
            if candidates:
                value = exact_match(str(row.get("response", "")), candidates)
        if value is not None:
            candidate_values.append(bool(value))

    token_values = [
        int(row["generated_token_count"])
        for row in rows
        if row.get("generated_token_count") is not None
    ]
    eos_values = [bool(row["ended_with_eos"]) for row in rows if "ended_with_eos" in row]
    raw_lengths = [len(str(row.get("raw_response", ""))) for row in rows]
    candidate_exact_count = sum(candidate_values)
    candidate_out_count = len(candidate_values) - candidate_exact_count
    return {
        "count": count,
        "empty_response_count": empty_count,
        "empty_response_rate": empty_count / count if count else 0.0,
        "candidate_evaluable_count": len(candidate_values),
        "candidate_exact_count": candidate_exact_count,
        "candidate_exact_rate": (
            candidate_exact_count / len(candidate_values) if candidate_values else 0.0
        ),
        "candidate_out_of_set_count": candidate_out_count,
        "candidate_out_of_set_rate": (
            candidate_out_count / len(candidate_values) if candidate_values else 0.0
        ),
        "generated_token_observed_count": len(token_values),
        "mean_generated_token_count": (
            sum(token_values) / len(token_values) if token_values else 0.0
        ),
        "eos_observed_count": len(eos_values),
        "eos_count": sum(eos_values),
        "eos_rate": sum(eos_values) / len(eos_values) if eos_values else 0.0,
        "mean_raw_response_length": (
            sum(raw_lengths) / len(raw_lengths) if raw_lengths else 0.0
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _vector_norm(vector: list[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _state_dynamics_stats(rows: list[dict]) -> dict:
    trace_lengths: list[float] = []
    initial_norms: list[float] = []
    final_norms: list[float] = []
    final_initial_ratios: list[float] = []
    consecutive_cosines: list[float] = []
    consecutive_relative_deltas: list[float] = []
    trace_present_count = 0
    transition_row_count = 0

    for row in rows:
        raw_trace = row.get("step_pooled_hidden_states")
        if not isinstance(raw_trace, list) or not raw_trace:
            continue
        trace: list[list[float]] = []
        for raw_vector in raw_trace:
            if not isinstance(raw_vector, list) or not raw_vector:
                trace = []
                break
            try:
                trace.append([float(value) for value in raw_vector])
            except (TypeError, ValueError):
                trace = []
                break
        if not trace or len({len(vector) for vector in trace}) != 1:
            continue

        trace_present_count += 1
        trace_lengths.append(float(len(trace)))
        norms = [_vector_norm(vector) for vector in trace]
        initial_norms.append(norms[0])
        final_norms.append(norms[-1])
        if norms[0] > 0.0:
            final_initial_ratios.append(norms[-1] / norms[0])

        row_has_transition = False
        for left, right, left_norm, right_norm in zip(
            trace, trace[1:], norms, norms[1:]
        ):
            row_has_transition = True
            dot = sum(x * y for x, y in zip(left, right))
            denominator = left_norm * right_norm
            consecutive_cosines.append(dot / denominator if denominator else 0.0)
            delta_norm = _vector_norm([y - x for x, y in zip(left, right)])
            consecutive_relative_deltas.append(
                delta_norm / left_norm if left_norm else 0.0
            )
        if row_has_transition:
            transition_row_count += 1

    count = len(rows)
    return {
        "count": count,
        "trace_present_count": trace_present_count,
        "trace_present_rate": trace_present_count / count if count else 0.0,
        "transition_row_count": transition_row_count,
        "mean_trace_length": _mean(trace_lengths),
        "mean_initial_hidden_norm": _mean(initial_norms),
        "mean_final_hidden_norm": _mean(final_norms),
        "mean_final_initial_norm_ratio": _mean(final_initial_ratios),
        "consecutive_transition_count": len(consecutive_cosines),
        "mean_consecutive_cosine": _mean(consecutive_cosines),
        "mean_consecutive_relative_delta": _mean(consecutive_relative_deltas),
    }


def _transition_summary(rows: list[dict]) -> dict[str, dict]:
    groups: dict[tuple[str, int, str], dict[tuple[str, str], dict[int, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        depth = int(row["recurrence_k"])
        if depth not in {1, 2}:
            continue
        split = str(row.get("metadata", {}).get("split", "unknown"))
        key = (split, _train_k(row), row.get("adapter_control", "correct"))
        question_key = (str(row.get("graph_id", "unknown")), str(row["question_id"]))
        groups[key][question_key][depth] = bool(row["correct"])

    output: dict[str, dict] = {}
    for (split, train_k, control), questions in sorted(groups.items()):
        counts = {
            "paired_count": 0,
            "both_correct": 0,
            "k1_correct_k2_wrong": 0,
            "k1_wrong_k2_correct": 0,
            "both_wrong": 0,
        }
        for outcomes in questions.values():
            if 1 not in outcomes or 2 not in outcomes:
                continue
            counts["paired_count"] += 1
            if outcomes[1] and outcomes[2]:
                counts["both_correct"] += 1
            elif outcomes[1] and not outcomes[2]:
                counts["k1_correct_k2_wrong"] += 1
            elif not outcomes[1] and outcomes[2]:
                counts["k1_wrong_k2_correct"] += 1
            else:
                counts["both_wrong"] += 1
        if counts["paired_count"]:
            output[
                f"split={split}|train_k={train_k}|adapter={control}"
            ] = counts
    return output


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
            "by_train_k_eval_k_and_adapter": {},
            "by_split_train_k_eval_k_and_adapter": {},
            "k1_k2_transitions": {},
            "output_quality": {"overall": _quality_stats([]), "by_train_k_eval_k_and_adapter": {}},
            "state_dynamics": {
                "overall": _state_dynamics_stats([]),
                "by_train_k_eval_k_and_adapter": {},
                "by_split_train_k_eval_k_and_adapter": {},
            },
        }

    hop_buckets: dict[tuple[int, int, str], list[bool]] = defaultdict(list)
    depth_buckets: dict[tuple[int, str], list[bool]] = defaultdict(list)
    split_depth_buckets: dict[tuple[str, int, str], list[bool]] = defaultdict(list)
    cross_buckets: dict[tuple[int, int, str], list[bool]] = defaultdict(list)
    split_cross_buckets: dict[tuple[str, int, int, str], list[bool]] = defaultdict(list)
    quality_buckets: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    state_buckets: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    split_state_buckets: dict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    known_hop_count = 0
    for row in rows:
        depth = int(row["recurrence_k"])
        train_k = _train_k(row)
        control = row.get("adapter_control", "correct")
        correct = bool(row["correct"])
        split = str(row.get("metadata", {}).get("split", "unknown"))
        depth_buckets[(depth, control)].append(correct)
        split_depth_buckets[(split, depth, control)].append(correct)
        cross_buckets[(train_k, depth, control)].append(correct)
        split_cross_buckets[(split, train_k, depth, control)].append(correct)
        quality_buckets[(train_k, depth, control)].append(row)
        state_buckets[(train_k, depth, control)].append(row)
        split_state_buckets[(split, train_k, depth, control)].append(row)
        hop = _known_hop(row)
        if hop is not None:
            known_hop_count += 1
            hop_buckets[(hop, depth, control)].append(correct)

    by_hop_and_k = {
        f"hop={hop}|k={depth}|adapter={control}": _accuracy_bucket(values)
        for (hop, depth, control), values in sorted(hop_buckets.items())
    }
    by_k_and_adapter = {
        f"k={depth}|adapter={control}": _accuracy_bucket(values)
        for (depth, control), values in sorted(depth_buckets.items())
    }
    by_split_k_and_adapter = {
        f"split={split}|k={depth}|adapter={control}": _accuracy_bucket(values)
        for (split, depth, control), values in sorted(split_depth_buckets.items())
    }
    by_train_k_eval_k_and_adapter = {
        f"train_k={train_k}|eval_k={depth}|adapter={control}": _accuracy_bucket(values)
        for (train_k, depth, control), values in sorted(cross_buckets.items())
    }
    by_split_train_k_eval_k_and_adapter = {
        f"split={split}|train_k={train_k}|eval_k={depth}|adapter={control}": _accuracy_bucket(values)
        for (split, train_k, depth, control), values in sorted(split_cross_buckets.items())
    }

    correct_rows = [row for row in rows if row.get("adapter_control", "correct") == "correct"]
    question_ids: set[tuple[int, str, str]] = set()
    successful_depths: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    hop_lookup: dict[tuple[int, str, str], int] = {}
    for row in correct_rows:
        question_key = (
            _train_k(row),
            str(row.get("graph_id", "unknown")),
            str(row["question_id"]),
        )
        question_ids.add(question_key)
        hop = _known_hop(row)
        if hop is not None:
            hop_lookup[question_key] = hop
        if bool(row["correct"]):
            successful_depths[question_key].append(int(row["recurrence_k"]))

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
    quality_by_cross = {
        f"train_k={train_k}|eval_k={depth}|adapter={control}": _quality_stats(bucket_rows)
        for (train_k, depth, control), bucket_rows in sorted(quality_buckets.items())
    }
    state_by_cross = {
        f"train_k={train_k}|eval_k={depth}|adapter={control}": _state_dynamics_stats(bucket_rows)
        for (train_k, depth, control), bucket_rows in sorted(state_buckets.items())
    }
    state_by_split_cross = {
        f"split={split}|train_k={train_k}|eval_k={depth}|adapter={control}": _state_dynamics_stats(bucket_rows)
        for (split, train_k, depth, control), bucket_rows in sorted(split_state_buckets.items())
    }
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
        "by_train_k_eval_k_and_adapter": by_train_k_eval_k_and_adapter,
        "by_split_train_k_eval_k_and_adapter": by_split_train_k_eval_k_and_adapter,
        "k1_k2_transitions": _transition_summary(rows),
        "output_quality": {
            "overall": _quality_stats(rows),
            "by_train_k_eval_k_and_adapter": quality_by_cross,
        },
        "state_dynamics": {
            "overall": _state_dynamics_stats(rows),
            "by_train_k_eval_k_and_adapter": state_by_cross,
            "by_split_train_k_eval_k_and_adapter": state_by_split_cross,
        },
    }
