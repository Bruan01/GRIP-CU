from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.confusion_audit import (  # noqa: E402
    SCORE_COMPARE_ATOL,
    audit_dump,
    check_group_completeness,
    check_group_mass,
    check_group_ranks,
    compare_score_groups,
    live_rescore_is_ok,
    math_row_failures,
    render_audit_markdown,
    summarize_comparisons,
    write_audit_outputs,
)
from hard_negative_grip.offline_scoring import (  # noqa: E402
    ALTERNATIVE_TRUE_REASON,
    build_qa_score_rows,
    pairwise_confusion,
)


def _toy_group(*, alt_true: bool = True) -> list[dict]:
    order = ["concept:gold", "concept:true", "concept:hard"]
    scores = {
        "concept:gold": -0.2,
        "concept:true": -0.5,
        "concept:hard": -1.1,
    }
    lengths = {rel: 2 for rel in order}
    known = ["concept:true"] if alt_true else []
    rows, _summary = build_qa_score_rows(
        qa_id="task_qa:0",
        question="what is the relation between a and b?",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=order,
        scores=scores,
        token_lengths=lengths,
        known_relations=known,
        temperature=1.0,
        model_checkpoint="runs/b1/adapter",
        split="train",
    )
    return rows


def test_math_and_ranks_match_saved_toy_rows() -> None:
    group = _toy_group()
    order = ["concept:gold", "concept:true", "concept:hard"]
    known = {("a", "b"): {"concept:gold", "concept:true"}}
    assert check_group_completeness(group, relation_order=order, known_relations=known) == []
    assert check_group_mass(group) == []
    assert check_group_ranks(group) == []
    for row in group:
        assert math_row_failures(row, 1.0) == []
        if not row["is_gold"]:
            expected = pairwise_confusion(
                float(row["candidate_score"]),
                float(row["gold_score"]),
                1.0,
            )
            assert abs(float(row["pairwise_confusion"]) - expected) < 1e-12
    hard = next(row for row in group if row["candidate_relation"] == "concept:hard")
    true_rel = next(row for row in group if row["candidate_relation"] == "concept:true")
    assert true_rel["is_valid_negative"] is False
    assert true_rel["invalid_reason"] == ALTERNATIVE_TRUE_REASON
    assert hard["is_valid_negative"] is True
    assert abs(float(hard["negative_mass"]) - 1.0) < 1e-12


def test_mass_and_rank_detect_corruption() -> None:
    group = _toy_group()
    broken_mass = [dict(row) for row in group]
    for row in broken_mass:
        if row["is_valid_negative"]:
            row["negative_mass"] = 0.3
    assert any("negative_mass" in item for item in check_group_mass(broken_mass))
    broken_rank = [dict(row) for row in group]
    for row in broken_rank:
        if row["candidate_relation"] == "concept:hard":
            row["candidate_rank_all"] = 99
    assert any("candidate_rank_all" in item for item in check_group_ranks(broken_rank))


def _write_task_and_kg(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "tasks.json"
    texts = [
        (
            "<|im_start|>user\nGiven the context graph titled nell23k, please answer "
            "the following question: what is the relation between a and b? "
            "Response in the following format:<answer>[answer]</answer><|im_end|>\n"
            "<|im_start|>assistant\n<answer>concept:gold</answer><|im_end|>\n"
        )
    ]
    task.write_text(json.dumps({"qa_samples": texts, "context_samples": ["graph"]}), encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.txt").write_text("a concept:gold b\nx concept:true y\nx concept:hard z\n", encoding="utf-8")
    (raw / "valid.txt").write_text("p concept:gold q\n", encoding="utf-8")
    (raw / "test.txt").write_text("a concept:true b\n", encoding="utf-8")
    return task, raw


def test_audit_dump_flags_test_structure_and_writes_revise(tmp_path: Path) -> None:
    task, raw = _write_task_and_kg(tmp_path)
    group = _toy_group()
    scores = tmp_path / "candidate_scores.jsonl"
    summary = tmp_path / "qa_summary.jsonl"
    metadata = tmp_path / "metadata.json"
    with scores.open("w", encoding="utf-8") as stream:
        for row in group:
            stream.write(json.dumps(row) + "\n")
    summary.write_text(
        json.dumps(
            {
                "qa_id": "task_qa:0",
                "gold_relation": "concept:gold",
                "matched_train_relation": "concept:gold",
                "gold_rank": 1,
                "number_of_negatives_above_gold": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "checkpoint": "runs/b1/adapter",
                "split": "train",
                "temperature": 1.0,
                "scoring_version": "offline_confusion_v1",
                "tokenizer": "toy",
                "dtype": "float32",
                "candidate_batch_size": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_dump(
        scores_path=scores,
        summary_path=summary,
        metadata_path=metadata,
        task_file=task,
        raw_dir=raw,
        seed=2026,
        alt_true_sample=10,
        valid_neg_sample=10,
        math_sample=10,
        rank_sample=10,
        scorer_qa_sample=1,
    )
    assert result["completeness_ok"] is True
    assert result["numerical"]["ok"] is True
    assert result["ranking"]["ok"] is True
    assert result["false_negative"]["ok"] is True
    assert result["false_negative"]["alternative_true"]["n_missing_from_kg"] == 0
    assert result["leakage"]["pair_overlap"]["alternative_true_from_test"] == 1
    assert result["final_status"] == "REVISE"
    assert "test_structure_in_train_filter" in result["issues"]
    output = tmp_path / "audit"
    written = write_audit_outputs(result, output)
    text = Path(written["report"]).read_text(encoding="utf-8")
    assert "# Final Status" in text
    assert "**REVISE**" in text
    assert "test KG structure" in text
    assert render_audit_markdown(result).startswith("# Dataset")
    assert result["issues"].count("live_7b_rescore_unavailable") == 1
    assert "live_7b_rescore_unavailable" in result["blocking_issues"]
    assert "scoring_inconsistency_live_20qa" not in result["blocking_issues"]


def test_compare_score_groups_and_live_rescore_gate() -> None:
    left = [
        {"candidate_relation": "concept:gold", "candidate_score": -0.2, "candidate_rank_all": 1},
        {"candidate_relation": "concept:hard", "candidate_score": -1.1, "candidate_rank_all": 2},
    ]
    right = [
        {"candidate_relation": "concept:gold", "candidate_score": -0.2, "candidate_rank_all": 1},
        {"candidate_relation": "concept:hard", "candidate_score": -1.1, "candidate_rank_all": 2},
    ]
    match = compare_score_groups(left, right)
    assert match["max_abs_error"] == 0.0
    assert match["n_rank_mismatch"] == 0
    assert match["output_order_mismatch"] is False
    drifted = compare_score_groups(
        left,
        [
            {"candidate_relation": "concept:hard", "candidate_score": -1.0, "candidate_rank_all": 2},
            {"candidate_relation": "concept:gold", "candidate_score": -0.2, "candidate_rank_all": 1},
        ],
    )
    assert drifted["max_abs_error"] > SCORE_COMPARE_ATOL
    assert drifted["output_order_mismatch"] is True
    summary = summarize_comparisons(
        [{"qa_id": "task_qa:0", **drifted}],
        label="training_vs_offline",
    )
    assert summary["output_order_mismatch"] is True
    assert live_rescore_is_ok(None) is False
    ok_payload = {
        "n_qa": 20,
        "training_vs_offline": {
            "n_qa": 20,
            "max_abs_error": 0.0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        },
        "training_run1_vs_run2": {
            "n_qa": 20,
            "max_abs_error": 0.0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        },
        "offline_run1_vs_run2": {
            "n_qa": 20,
            "max_abs_error": 0.0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        },
        "dump_vs_training": {
            "n_qa": 20,
            "max_abs_error": 0.0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        },
        "dump_vs_offline": {
            "n_qa": 20,
            "max_abs_error": 0.0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        },
    }
    assert live_rescore_is_ok(ok_payload) is True
    drifted_payload = dict(ok_payload)
    drifted_payload["dump_vs_training"] = {
        "n_qa": 20,
        "max_abs_error": 1e-3,
        "n_rank_mismatch_total": 0,
        "output_order_mismatch": False,
    }
    assert live_rescore_is_ok(drifted_payload) is False


def test_audit_dump_live_rescore_mismatch_is_blocking(tmp_path: Path) -> None:
    task, raw = _write_task_and_kg(tmp_path)
    group = _toy_group()
    scores = tmp_path / "candidate_scores.jsonl"
    summary = tmp_path / "qa_summary.jsonl"
    metadata = tmp_path / "metadata.json"
    live = tmp_path / "live_20qa_rescore.json"
    with scores.open("w", encoding="utf-8") as stream:
        for row in group:
            stream.write(json.dumps(row) + "\n")
    summary.write_text(
        json.dumps(
            {
                "qa_id": "task_qa:0",
                "gold_relation": "concept:gold",
                "matched_train_relation": "concept:gold",
                "gold_rank": 1,
                "number_of_negatives_above_gold": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "checkpoint": "runs/b1/adapter",
                "split": "train",
                "temperature": 1.0,
                "scoring_version": "offline_confusion_v1",
                "tokenizer": "toy",
                "dtype": "float32",
                "candidate_batch_size": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    live.write_text(
        json.dumps(
            {
                "n_qa": 20,
                "training_vs_offline": {
                    "n_qa": 20,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "n_rank_mismatch_total": 0,
                    "output_order_mismatch": False,
                },
                "training_run1_vs_run2": {
                    "n_qa": 20,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "n_rank_mismatch_total": 0,
                    "output_order_mismatch": False,
                },
                "offline_run1_vs_run2": {
                    "n_qa": 20,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "n_rank_mismatch_total": 0,
                    "output_order_mismatch": False,
                },
                "dump_vs_training": {
                    "n_qa": 20,
                    "max_abs_error": 0.02,
                    "mean_abs_error": 0.01,
                    "n_rank_mismatch_total": 0,
                    "output_order_mismatch": False,
                },
                "dump_vs_offline": {
                    "n_qa": 20,
                    "max_abs_error": 0.02,
                    "mean_abs_error": 0.01,
                    "n_rank_mismatch_total": 0,
                    "output_order_mismatch": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_dump(
        scores_path=scores,
        summary_path=summary,
        metadata_path=metadata,
        task_file=task,
        raw_dir=raw,
        live_rescore_path=live,
        seed=2026,
        alt_true_sample=10,
        valid_neg_sample=10,
        math_sample=10,
        rank_sample=10,
        scorer_qa_sample=1,
    )
    assert "scoring_inconsistency_live_20qa" in result["blocking_issues"]
    assert result["final_status"] == "REVISE"
