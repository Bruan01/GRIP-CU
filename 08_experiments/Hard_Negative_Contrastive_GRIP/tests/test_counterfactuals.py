from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.counterfactuals import (  # noqa: E402
    STRUCTURE_COUNTERFACTUAL_SOURCE,
    build_relation_profiles,
    relation_profile_similarity,
    sample_structure_counterfactual_group,
    write_bank_artifacts,
)


def _profiles():
    return build_relation_profiles(
        [
            ("h1", "gold", "t1"),
            ("h1", "gold", "t2"),
            ("h2", "gold", "t3"),
            ("h1", "near", "t4"),
            ("h2", "near", "t5"),
            ("h3", "far", "t6"),
            ("h4", "far", "t7"),
            ("h5", "far", "t8"),
        ]
    )


def _score_group() -> list[dict]:
    return [
        {
            "qa_id": "task_qa:3",
            "split": "train",
            "is_gold": True,
            "is_valid_negative": False,
            "candidate_relation": "gold",
            "matched_train_relation": "gold",
            "gold_relation": "gold",
            "gold_score": 0.1,
            "candidate_score": 0.1,
            "score_gap": 0.0,
        },
        {
            "qa_id": "task_qa:3",
            "split": "train",
            "is_gold": False,
            "is_valid_negative": True,
            "candidate_relation": "near",
            "candidate_score": 0.08,
            "gold_score": 0.1,
            "score_gap": 0.02,
            "pairwise_confusion": 0.48,
        },
        {
            "qa_id": "task_qa:3",
            "split": "train",
            "is_gold": False,
            "is_valid_negative": True,
            "candidate_relation": "far",
            "candidate_score": 0.01,
            "gold_score": 0.1,
            "score_gap": 0.09,
            "pairwise_confusion": 0.41,
        },
        {
            "qa_id": "task_qa:3",
            "split": "train",
            "is_gold": False,
            "is_valid_negative": True,
            "candidate_relation": "known_true",
            "candidate_score": 0.07,
            "gold_score": 0.1,
            "score_gap": 0.03,
            "pairwise_confusion": 0.47,
        },
    ]


def test_relation_profiles_are_deterministic_and_bounded() -> None:
    profiles = _profiles()
    first = relation_profile_similarity(profiles["gold"], profiles["near"])
    second = relation_profile_similarity(profiles["gold"], profiles["far"])
    assert 0.0 <= first <= 1.0
    assert 0.0 <= second <= 1.0
    assert first > second


def test_counterfactual_bank_filters_known_relation_and_is_reproducible() -> None:
    profiles = _profiles()
    first, audit = sample_structure_counterfactual_group(
        _score_group(),
        profiles,
        total_k=2,
        seed=2026,
        known_relations={"known_true"},
    )
    second, _ = sample_structure_counterfactual_group(
        _score_group(),
        profiles,
        total_k=2,
        seed=2026,
        known_relations={"known_true"},
    )
    assert first == second
    assert first["negative_source"] == STRUCTURE_COUNTERFACTUAL_SOURCE
    assert len(first["negative_relations"]) == 2
    assert "known_true" not in first["negative_relations"]
    assert all(item["counterfactual_valid"] is True for item in first["selected_negatives"])
    assert audit["selected_k"] == 2


def test_counterfactual_bank_rejects_non_train_score_groups() -> None:
    rows = _score_group()
    rows[0]["split"] = "valid"
    with pytest.raises(ValueError, match="train rows only"):
        sample_structure_counterfactual_group(rows, _profiles(), total_k=2)


def test_write_bank_artifacts_keeps_failures_and_metadata(tmp_path: Path) -> None:
    scores = tmp_path / "candidate_scores.jsonl"
    scores.write_text(
        "\n".join(json.dumps(row) for row in _score_group()) + "\n",
        encoding="utf-8",
    )
    metadata = write_bank_artifacts(
        score_path=scores,
        output_dir=tmp_path / "bank",
        profiles=_profiles(),
        total_k=2,
        seed=2026,
    )
    assert metadata["rows_written"] == 1
    assert (tmp_path / "bank" / "counterfactual_bank.jsonl").is_file()
    saved = json.loads(
        (tmp_path / "bank" / "counterfactual_bank.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert saved["requested_k"] == 2


def test_write_bank_artifacts_refuses_implicit_overwrite(tmp_path: Path) -> None:
    scores = tmp_path / "candidate_scores.jsonl"
    scores.write_text(
        "\n".join(json.dumps(row) for row in _score_group()) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "bank"
    write_bank_artifacts(
        score_path=scores,
        output_dir=output_dir,
        profiles=_profiles(),
        total_k=2,
        seed=2026,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bank_artifacts(
            score_path=scores,
            output_dir=output_dir,
            profiles=_profiles(),
            total_k=2,
            seed=2026,
        )
