from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HNG = Path(__file__).parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.error_memory_sampling import (  # noqa: E402
    build_error_memory,
    freeze_error_memory_bank,
    sample_error_memory_negatives,
)
from hard_negative_grip.offline_scoring import build_qa_score_rows  # noqa: E402
from hard_negative_grip.offline_mining import write_jsonl  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    build_qa_assets_from_task_texts,
    load_error_memory_manifest,
    validate_error_memory_manifest,
)


RELATIONS = [f"concept:r{i}" for i in range(12)]


def _group(qa_id: str, gold: str = "concept:r0") -> list[dict]:
    scores = {relation: -1.0 - 0.05 * index for index, relation in enumerate(RELATIONS)}
    scores["concept:r1"] = -0.9
    scores["concept:r2"] = -0.91
    return build_qa_score_rows(
        qa_id=qa_id,
        question="what is the relation between a and b?",
        head_entity="a",
        tail_entity="b",
        gold_relation=gold,
        matched_train_relation=gold,
        relation_order=RELATIONS,
        scores=scores,
        token_lengths={relation: 1 for relation in RELATIONS},
        known_relations=[],
        model_checkpoint="/frozen/b1/adapter",
        split="train",
    )[0]


def _rollout_rows() -> list[dict]:
    return [
        {
            "question_id": "task_qa:0",
            "split": "train",
            "positive_relation": "concept:r0",
            "matched_train_relation": "concept:r0",
            "rollout_hard_negative": "concept:r1",
            "rollout_hard_is_train_vocab": True,
        },
        {
            "question_id": "task_qa:1",
            "split": "train",
            "positive_relation": "concept:r0",
            "matched_train_relation": "concept:r0",
            "rollout_hard_negative": "concept:r1",
            "rollout_hard_is_train_vocab": True,
        },
        {
            "question_id": "task_qa:2",
            "split": "train",
            "positive_relation": "concept:r0",
            "matched_train_relation": "concept:r0",
            "rollout_hard_negative": "concept:r2",
            "rollout_hard_is_train_vocab": True,
        },
    ]


def test_error_memory_counts_train_only_relation_confusions() -> None:
    memory = build_error_memory(_rollout_rows(), RELATIONS)
    assert memory["counts"]["concept:r0"] == {
        "concept:r1": 2,
        "concept:r2": 1,
    }
    assert memory["stats"]["train_relation_errors"] == 3


def test_sampler_combines_memory_and_score_violation_with_uniform_floor() -> None:
    rows = _group("task_qa:0")
    memory = build_error_memory(_rollout_rows(), RELATIONS)
    manifest, audit = sample_error_memory_negatives(
        rows,
        memory,
        total_k=9,
        uniform_mix=1.0 / 3.0,
        seed=2026,
    )
    assert len(manifest["negative_relations"]) == 9
    assert manifest["negative_relations"] == manifest["hard_negative_relations"]
    assert manifest["uniform_negative_relations"] == []
    assert manifest["negative_source"].startswith("train_only_error_memory")
    assert audit["selected_error_memory_hits"] >= 1
    probabilities = [item["sampling_probability"] for item in manifest["selected_negatives"]]
    assert all(value > 0 for value in probabilities)


def test_freeze_and_train_loader_accept_only_train_only_bank(tmp_path: Path) -> None:
    groups = [_group(f"task_qa:{index}") for index in range(3)]
    scores = tmp_path / "candidate_scores.jsonl"
    write_jsonl(scores, [row for group in groups for row in group])
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"split": "train", "filter_splits": ["train"]}),
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout.jsonl"
    write_jsonl(rollout, _rollout_rows())
    output = tmp_path / "bank"
    result = freeze_error_memory_bank(
        scores_path=scores,
        metadata_path=metadata,
        rollout_path=rollout,
        output_dir=output,
        relation_order=RELATIONS,
    )
    assert result["n_qa"] == 3
    bank = load_error_memory_manifest(output / "error_memory_k.jsonl")
    validate_error_memory_manifest(bank)
    text = (
        "<|im_start|>user\nwhat is the relation between a and b?"
        "<|im_end|>\n<|im_start|>assistant\n<answer>concept:r0</answer>"
    )
    _texts, metas = build_qa_assets_from_task_texts(
        [text],
        seed=2026,
        listed_negative_source="error_memory",
        relation_order=RELATIONS,
        error_memory_manifest=bank,
        question_ids=["task_qa:0"],
    )
    assert metas[0]["listed_relations"] == bank["task_qa:0"]["negative_relations"]
    with pytest.raises(FileExistsError):
        freeze_error_memory_bank(
            scores_path=scores,
            metadata_path=metadata,
            rollout_path=rollout,
            output_dir=output,
            relation_order=RELATIONS,
        )


def test_freeze_rejects_all_split_score_metadata(tmp_path: Path) -> None:
    scores = tmp_path / "candidate_scores.jsonl"
    write_jsonl(scores, _group("task_qa:0"))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"split": "train", "filter_splits": ["train", "valid", "test"]}),
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout.jsonl"
    write_jsonl(rollout, _rollout_rows()[:1])
    with pytest.raises(ValueError, match="filter_splits"):
        freeze_error_memory_bank(
            scores_path=scores,
            metadata_path=metadata,
            rollout_path=rollout,
            output_dir=tmp_path / "bad",
            relation_order=RELATIONS,
        )
