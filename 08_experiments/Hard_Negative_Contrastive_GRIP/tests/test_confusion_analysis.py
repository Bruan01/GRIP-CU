from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.confusion_analysis import (  # noqa: E402
    analyze_qa_group,
    analyze_score_groups,
    expand_confusion_db_row,
    k_for_coverage,
    percentile,
    topn_cumulative,
)
from hard_negative_grip.offline_scoring import build_qa_score_rows  # noqa: E402


def _rows() -> list[dict]:
    order = ["concept:gold", "concept:true", "concept:hard", "concept:mid", "concept:easy"]
    scores = {
        "concept:gold": -0.8,
        "concept:true": -0.7,
        "concept:hard": -0.9,
        "concept:mid": -1.4,
        "concept:easy": -3.0,
    }
    lengths = {rel: 1 for rel in order}
    rows, _summary = build_qa_score_rows(
        qa_id="task_qa:0",
        question="q",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=order,
        scores=scores,
        token_lengths=lengths,
        known_relations=["concept:true"],
        temperature=1.0,
        model_checkpoint="b1",
    )
    return rows


def test_percentile_and_k_coverage() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    masses = [0.50, 0.30, 0.15, 0.05]
    assert k_for_coverage(masses, 0.70) == 2
    assert k_for_coverage(masses, 0.80) == 2
    assert k_for_coverage(masses, 0.90) == 3
    assert k_for_coverage(masses, 0.99) == 4
    assert abs(topn_cumulative(masses, 3) - 0.95) < 1e-12


def test_analyze_qa_group_uses_valid_negatives_only() -> None:
    summary = analyze_qa_group(_rows())
    assert summary["n_valid_negatives"] == 3
    assert summary["hardest_negative"] == "concept:hard"
    assert abs(summary["hardest_negative_gap"] - 0.1) < 1e-12
    assert summary["K_80"] >= 1
    assert 0.0 < summary["top1_mass"] < summary["top3_mass"] <= 1.0 + 1e-12


def test_gap_and_confusion_buckets_count_inverted_gold() -> None:
    inverted, _summary = build_qa_score_rows(
        qa_id="task_qa:1",
        question="q",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=["concept:gold", "concept:hard", "concept:easy"],
        scores={"concept:gold": -1.0, "concept:hard": -0.2, "concept:easy": -4.0},
        token_lengths={"concept:gold": 1, "concept:hard": 1, "concept:easy": 1},
        known_relations=[],
        temperature=1.0,
        model_checkpoint="b1",
    )
    result = analyze_score_groups([inverted])
    gap_by_name = {row["bucket"]: row for row in result["gap_buckets"]}
    assert gap_by_name["gap < 0"]["candidate_count"] == 1
    assert gap_by_name["gap < 0"]["qa_count"] == 1
    confusion_by_name = {row["bucket"]: row for row in result["confusion_buckets"]}
    assert confusion_by_name["p > 0.50"]["candidate_count"] == 1
    assert result["hardest_negative_gap"]["qa_hardest_gap_lt_0"] == 1
    assert result["coverage_k"]["K_90"]["K_leq_1"]["qa_count"] == 1
    pair = result["relation_pairs"][0]
    assert pair["gold_relation"] == "concept:gold"
    assert pair["candidate_relation"] == "concept:hard"
    assert pair["fraction_candidate_beats_gold"] == 1.0


def test_expand_confusion_db_row_matches_score_schema() -> None:
    row = {
        "question_id": "task_qa:9",
        "positive_relation": "concept:gold",
        "matched_train_relation": "concept:gold",
        "entity_pair": ["a", "b"],
        "known_pair_relations": ["concept:gold", "concept:true"],
        "all_candidate_scores": [
            {"relation": "concept:gold", "score": -0.8, "rank": 2},
            {"relation": "concept:true", "score": -0.7, "rank": 1},
            {"relation": "concept:hard", "score": -0.9, "rank": 3},
        ],
        "b1_adapter": "frozen/b1",
        "split": "train",
    }
    rows, summary = expand_confusion_db_row(
        row,
        relation_order=["concept:gold", "concept:true", "concept:hard"],
    )
    by_rel = {item["candidate_relation"]: item for item in rows}
    assert by_rel["concept:true"]["is_valid_negative"] is False
    assert by_rel["concept:hard"]["is_valid_negative"] is True
    assert summary["hardest_negative"] == "concept:hard"
    dumped = json.dumps(rows[0])
    assert "qa_id" in dumped
