from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.offline_scoring import (  # noqa: E402
    ALTERNATIVE_TRUE_REASON,
    GOLD_INVALID_REASON,
    build_qa_score_rows,
    classify_candidate,
    extract_question_text,
    parse_matchable_relation_qa,
    pairwise_confusion,
    summarize_run,
    validate_candidate_rows,
)


def _qa(question: str, answer: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nGiven the context graph titled nell23k, please answer "
        f"the following question: {question} Response in the following format:"
        "<answer>[answer]</answer><|im_end|>\n"
        f"<|im_start|>assistant\n<answer>{answer}</answer><|im_end|>\n"
    )


def _toy_rows() -> tuple[list[dict], dict]:
    order = ["concept:gold", "concept:true", "concept:hard", "concept:easy"]
    scores = {
        "concept:gold": -0.8,
        "concept:true": -0.7,
        "concept:hard": -0.9,
        "concept:easy": -1.4,
    }
    lengths = {rel: index + 1 for index, rel in enumerate(order)}
    return build_qa_score_rows(
        qa_id="task_qa:0",
        question="what is the relation between a and b?",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=order,
        scores=scores,
        token_lengths=lengths,
        known_relations=["concept:gold", "concept:true"],
        temperature=1.0,
        model_checkpoint="frozen/b1/adapter",
        split="train",
    )


def test_extract_question_text_keeps_visible_prompt() -> None:
    text = _qa("what is the relation between alice and bob?", "concept:worksfor")
    assert extract_question_text(text) == "what is the relation between alice and bob?"
    item = parse_matchable_relation_qa(
        7,
        text,
        {"concept:worksfor": "concept:worksfor"},
    )
    assert item is not None
    assert item["qa_id"] == "task_qa:7"
    assert item["head_entity"] == "alice"
    assert item["tail_entity"] == "bob"
    assert parse_matchable_relation_qa(0, _qa("Is Detroit a city?", "Yes"), {}) is None


def test_classify_candidate_keeps_alternative_true_relation_scoreable() -> None:
    is_gold, is_valid, reason = classify_candidate(
        candidate="concept:true",
        gold="concept:gold",
        known_relations=["concept:gold", "concept:true"],
    )
    assert is_gold is False
    assert is_valid is False
    assert reason == ALTERNATIVE_TRUE_REASON


def test_pairwise_confusion_matches_two_way_softmax() -> None:
    gold, candidate = -0.8, -0.9
    value = pairwise_confusion(candidate, gold, temperature=1.0)
    expected = math.exp(candidate) / (math.exp(gold) + math.exp(candidate))
    assert abs(value - expected) < 1e-12
    assert 0.0 < value < 0.5
    inverted = pairwise_confusion(-0.5, -0.8, temperature=1.0)
    assert inverted > 0.5


def test_build_rows_mark_false_negatives_and_sum_negative_mass() -> None:
    rows, summary = _toy_rows()
    by_rel = {row["candidate_relation"]: row for row in rows}
    gold = by_rel["concept:gold"]
    true_rel = by_rel["concept:true"]
    hard = by_rel["concept:hard"]
    easy = by_rel["concept:easy"]

    assert gold["is_gold"] is True
    assert gold["is_valid_negative"] is False
    assert gold["invalid_reason"] == GOLD_INVALID_REASON
    assert gold["pairwise_confusion"] is None
    assert gold["negative_rank"] is None
    assert gold["negative_mass"] is None
    assert abs(gold["score_gap"]) < 1e-12

    assert true_rel["is_valid_negative"] is False
    assert true_rel["invalid_reason"] == ALTERNATIVE_TRUE_REASON
    assert true_rel["negative_mass"] is None
    assert true_rel["candidate_score"] == -0.7

    valid = [hard, easy]
    assert all(row["is_valid_negative"] for row in valid)
    mass = sum(float(row["negative_mass"]) for row in valid)
    assert abs(mass - 1.0) < 1e-12
    assert hard["negative_rank"] == 1
    assert easy["negative_rank"] == 2
    assert true_rel["candidate_rank_all"] == 1
    assert gold["candidate_rank_all"] == 2
    assert hard["candidate_rank_all"] == 3
    assert abs(hard["score_gap"] - 0.1) < 1e-12
    assert hard["score_gap"] > 0
    assert true_rel["score_gap"] < 0
    assert 0.0 < hard["pairwise_confusion"] < 1.0

    assert summary["gold_rank"] == 2
    assert summary["number_of_valid_negatives"] == 2
    assert summary["hardest_negative"] == "concept:hard"
    assert summary["number_of_negatives_above_gold"] == 0


def test_validate_candidate_rows_accepts_toy_and_rejects_bad_mass() -> None:
    rows, _summary = _toy_rows()
    order = ["concept:gold", "concept:true", "concept:hard", "concept:easy"]
    known = {("a", "b"): {"concept:gold", "concept:true"}}
    assert validate_candidate_rows(rows, relation_order=order, known_relations=known) == []
    broken = [dict(row) for row in rows]
    for row in broken:
        if row["candidate_relation"] == "concept:hard":
            row["negative_mass"] = 0.1
    failures = validate_candidate_rows(broken, relation_order=order, known_relations=known)
    assert any("negative_mass" in item for item in failures)


def test_summarize_run_counts_inverted_gold_rank() -> None:
    summaries = [
        {"gold_rank": 1, "number_of_negatives_above_gold": 0},
        {"gold_rank": 3, "number_of_negatives_above_gold": 2},
    ]
    stats = summarize_run(summaries, elapsed_sec=10.0, peak_gpu_bytes=2 * 1024**3, vocab_size=198, skipped=4)
    assert stats["qa_count"] == 2
    assert stats["total_candidates"] == 396
    assert stats["qa_per_sec"] == 0.2
    assert stats["mean_gold_rank"] == 2.0
    assert stats["qa_with_negative_above_gold"] == 1
    assert abs(stats["peak_gpu_gib"] - 2.0) < 1e-9
