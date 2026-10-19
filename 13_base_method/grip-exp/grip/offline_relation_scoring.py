"""Pure diagnostic utilities for offline relation candidate scoring."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


def build_pair_relation_index(
    graph_edges: Sequence[Sequence[str]],
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Map each textual ``(head, tail)`` pair to all observed relation labels."""
    pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in graph_edges:
        if len(edge) < 3:
            continue
        head, relation, tail = (str(value) for value in edge[:3])
        pair_relations[(head, tail)].add(relation)
    return {
        pair: tuple(sorted(relations))
        for pair, relations in pair_relations.items()
    }


def resolve_pair_metadata(
    record: Mapping[str, Any],
    graph_edges: Sequence[Sequence[str]],
    pair_relations: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    """Resolve head/tail only from explicit metadata or indexed graph edges."""
    edge_indices = record.get("source_edge_indices")
    if edge_indices is None and record.get("source_edge_index") is not None:
        edge_indices = [record["source_edge_index"]]
    edge_indices = list(edge_indices or [])
    indexed_edges = []
    for raw_index in edge_indices:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            indexed_edges = []
            break
        if index < 0 or index >= len(graph_edges) or len(graph_edges[index]) < 3:
            indexed_edges = []
            break
        indexed_edges.append(graph_edges[index])

    indexed_pairs = {
        (str(edge[0]), str(edge[2]))
        for edge in indexed_edges
    }
    if indexed_edges and len(indexed_pairs) == 1:
        head, tail = next(iter(indexed_pairs))
        return {
            "head_entity": head,
            "tail_entity": tail,
            "all_true_relations": list(pair_relations.get((head, tail), ())),
            "metadata_source": "source_edge_indices",
            "false_negative_filter_available": True,
        }

    triple = record.get("triple")
    if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
        head, tail = str(triple[0]), str(triple[2])
        return {
            "head_entity": head,
            "tail_entity": tail,
            "all_true_relations": list(pair_relations.get((head, tail), ())),
            "metadata_source": "triple",
            "false_negative_filter_available": True,
        }

    head, tail = record.get("head_entity"), record.get("tail_entity")
    if head is not None and tail is not None:
        pair = (str(head), str(tail))
        return {
            "head_entity": str(head),
            "tail_entity": str(tail),
            "all_true_relations": list(pair_relations.get(pair, ())),
            "metadata_source": "explicit_fields",
            "false_negative_filter_available": True,
        }

    return {
        "head_entity": None,
        "tail_entity": None,
        "all_true_relations": [],
        "metadata_source": "unavailable",
        "false_negative_filter_available": False,
    }


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def compute_candidate_diagnostics(
    candidate_relations: Sequence[str],
    scores: Sequence[float],
    gold_relation: str,
    known_true_relations: Sequence[str],
    temperature: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute raw score diagnostics without changing the model scores."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(candidate_relations) != len(scores):
        raise ValueError("candidate_relations and scores must have the same length")
    gold_indices = [
        index for index, relation in enumerate(candidate_relations)
        if relation == gold_relation
    ]
    if len(gold_indices) != 1:
        raise ValueError("Each QA must contain exactly one gold relation")

    numeric_scores = [float(score) for score in scores]
    gold_index = gold_indices[0]
    gold_score = numeric_scores[gold_index]
    known_true = set(known_true_relations)
    ordered_indices = sorted(
        range(len(candidate_relations)),
        key=lambda index: (-numeric_scores[index], index),
    )
    all_ranks = {index: rank for rank, index in enumerate(ordered_indices, 1)}

    valid_negative_indices = [
        index for index, relation in enumerate(candidate_relations)
        if index != gold_index and relation not in known_true
    ]
    negative_order = sorted(
        valid_negative_indices,
        key=lambda index: (-numeric_scores[index], index),
    )
    negative_ranks = {index: rank for rank, index in enumerate(negative_order, 1)}

    negative_mass: dict[int, float] = {}
    if valid_negative_indices:
        scaled = {
            index: numeric_scores[index] / temperature
            for index in valid_negative_indices
        }
        max_scaled = max(scaled.values())
        weights = {
            index: math.exp(value - max_scaled)
            for index, value in scaled.items()
        }
        denominator = sum(weights.values())
        negative_mass = {
            index: weight / denominator
            for index, weight in weights.items()
        }

    rows = []
    for index, relation in enumerate(candidate_relations):
        is_gold = index == gold_index
        if is_gold:
            is_valid_negative = False
            invalid_reason = "gold_relation"
        elif relation in known_true:
            is_valid_negative = False
            invalid_reason = "alternative_true_relation"
        else:
            is_valid_negative = True
            invalid_reason = None
        rows.append({
            "candidate_index": index,
            "candidate_relation": relation,
            "is_gold": is_gold,
            "is_valid_negative": is_valid_negative,
            "invalid_reason": invalid_reason,
            "candidate_score": numeric_scores[index],
            "gold_score": gold_score,
            "score_gap": gold_score - numeric_scores[index],
            "pairwise_confusion": (
                None if is_gold
                else _stable_sigmoid((numeric_scores[index] - gold_score) / temperature)
            ),
            "negative_mass": negative_mass.get(index),
            "candidate_rank_all": all_ranks[index],
            "negative_rank": negative_ranks.get(index),
        })

    hardest_index = max(
        valid_negative_indices,
        key=lambda index: (numeric_scores[index], -index),
        default=None,
    )
    negative_mass_sum = sum(negative_mass.values()) if negative_mass else None
    summary = {
        "gold_relation": gold_relation,
        "gold_score": gold_score,
        "gold_rank": all_ranks[gold_index],
        "number_of_candidates": len(candidate_relations),
        "number_of_valid_negatives": len(valid_negative_indices),
        "hardest_negative": (
            candidate_relations[hardest_index] if hardest_index is not None else None
        ),
        "hardest_negative_score": (
            numeric_scores[hardest_index] if hardest_index is not None else None
        ),
        "hardest_gap": (
            gold_score - numeric_scores[hardest_index]
            if hardest_index is not None else None
        ),
        "number_of_negatives_above_gold": sum(
            numeric_scores[index] > gold_score
            for index in valid_negative_indices
        ),
        "negative_mass_sum": negative_mass_sum,
        "negative_mass_error": (
            abs(negative_mass_sum - 1.0)
            if negative_mass_sum is not None else None
        ),
    }
    return rows, summary


def validate_candidate_diagnostics(
    rows_by_qa: Mapping[Any, Sequence[Mapping[str, Any]]],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate the invariants required by the offline diagnostic schema."""
    checks = {
        "one_gold_per_qa": True,
        "gold_score_consistent": True,
        "valid_negative_excludes_gold": True,
        "false_negative_marked": True,
        "negative_mass_sums_to_one": True,
        "score_gap_correct": True,
        "pairwise_confusion_bounded": True,
        "candidate_ranks_unique": True,
        "negative_ranks_unique": True,
    }
    failures: list[dict[str, Any]] = []

    def fail(check: str, qa_id: Any, detail: str) -> None:
        checks[check] = False
        failures.append({"check": check, "qa_id": qa_id, "detail": detail})

    for qa_id, rows in rows_by_qa.items():
        gold_rows = [row for row in rows if row.get("is_gold")]
        if len(gold_rows) != 1:
            fail("one_gold_per_qa", qa_id, f"found {len(gold_rows)} gold rows")
            continue
        gold_score = float(gold_rows[0]["gold_score"])
        if any(abs(float(row["gold_score"]) - gold_score) > tolerance for row in rows):
            fail("gold_score_consistent", qa_id, "gold_score differs within QA")
        if any(row.get("is_valid_negative") and row.get("is_gold") for row in rows):
            fail("valid_negative_excludes_gold", qa_id, "gold marked as valid negative")

        known_true = set(
            gold_rows[0].get(
                "all_true_relations",
                gold_rows[0].get("known_true_relations", ()),
            )
        )
        for row in rows:
            relation = row["candidate_relation"]
            if relation in known_true and not row.get("is_gold"):
                if row.get("is_valid_negative") or row.get("invalid_reason") != "alternative_true_relation":
                    fail("false_negative_marked", qa_id, relation)
            expected_gap = gold_score - float(row["candidate_score"])
            if abs(float(row["score_gap"]) - expected_gap) > tolerance:
                fail("score_gap_correct", qa_id, relation)
            confusion = row.get("pairwise_confusion")
            if confusion is not None and not 0.0 <= float(confusion) <= 1.0:
                fail("pairwise_confusion_bounded", qa_id, relation)

        all_ranks = [int(row["candidate_rank_all"]) for row in rows]
        if sorted(all_ranks) != list(range(1, len(rows) + 1)):
            fail("candidate_ranks_unique", qa_id, str(all_ranks))
        valid_rows = [row for row in rows if row.get("is_valid_negative")]
        negative_ranks = [int(row["negative_rank"]) for row in valid_rows]
        if sorted(negative_ranks) != list(range(1, len(valid_rows) + 1)):
            fail("negative_ranks_unique", qa_id, str(negative_ranks))
        if valid_rows:
            mass_sum = sum(float(row["negative_mass"]) for row in valid_rows)
            if abs(mass_sum - 1.0) > tolerance:
                fail("negative_mass_sums_to_one", qa_id, str(mass_sum))
        elif any(row.get("negative_mass") is not None for row in rows):
            fail("negative_mass_sums_to_one", qa_id, "mass exists without valid negatives")

    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "qa_count": len(rows_by_qa),
    }
