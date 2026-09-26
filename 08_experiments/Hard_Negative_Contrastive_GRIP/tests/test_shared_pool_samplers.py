from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.offline_scoring import build_qa_score_rows  # noqa: E402
from hard_negative_grip.shared_pool_samplers import (  # noqa: E402
    decide_coverage_k,
    freeze_shared_pool_samplers,
    propose_coverage_tau,
    rank_valid_pool,
    sample_qa_variants,
    sample_random_k,
    sample_top_k,
    shared_valid_pool,
)


def _rows(
    *,
    qa_id: str,
    scores: dict[str, float],
    known: list[str] | None = None,
    order: list[str] | None = None,
) -> list[dict]:
    order = order or list(scores)
    lengths = {rel: 1 for rel in order}
    rows, _summary = build_qa_score_rows(
        qa_id=qa_id,
        question="q",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=order,
        scores=scores,
        token_lengths=lengths,
        known_relations=known or [],
        temperature=1.0,
        model_checkpoint="b1",
    )
    return rows


def test_shared_pool_drops_gold_and_alternative_true() -> None:
    rows = _rows(
        qa_id="task_qa:0",
        scores={
            "concept:gold": -0.2,
            "concept:true": -0.1,
            "concept:hard": -0.4,
            "concept:mid": -0.8,
            "concept:easy": -2.0,
        },
        known=["concept:true"],
    )
    pool = shared_valid_pool(rows)
    names = [row["candidate_relation"] for row in pool]
    assert names == ["concept:hard", "concept:mid", "concept:easy"]
    assert all(row["is_valid_negative"] for row in pool)


def test_random_and_top_k_share_the_same_pool() -> None:
    rows = _rows(
        qa_id="task_qa:0",
        scores={
            "concept:gold": -0.2,
            "concept:hard": -0.3,
            "concept:mid": -0.9,
            "concept:easy": -2.5,
            "concept:other": -1.5,
        },
    )
    sampled = sample_qa_variants(
        rows,
        k_fixed=2,
        tau=0.5,
        k_min=1,
        k_max=3,
        rng=random.Random(2026),
        seed=2026,
    )
    pool = {row["candidate_relation"] for row in shared_valid_pool(rows)}
    random_set = set(sampled["random_k"]["negative_relations"])
    top_set = set(sampled["top_k_hard"]["negative_relations"])
    adaptive_set = set(sampled["coverage_adaptive_k"]["negative_relations"])
    assert random_set <= pool
    assert top_set <= pool
    assert adaptive_set <= pool
    assert sampled["top_k_hard"]["hard_negative_relations"] == ["concept:hard", "concept:mid"]
    assert sampled["random_k"]["k"] == 2
    assert sampled["top_k_hard"]["k"] == 2
    assert "concept:gold" not in random_set | top_set | adaptive_set


def test_random_k_is_seed_reproducible_and_without_replacement() -> None:
    pool = [
        {"candidate_relation": f"r{i}", "candidate_score": -float(i), "negative_mass": 0.1}
        for i in range(10)
    ]
    first = sample_random_k(pool, 4, random.Random(7))
    second = sample_random_k(pool, 4, random.Random(7))
    third = sample_random_k(pool, 4, random.Random(8))
    names = [row["candidate_relation"] for row in first]
    assert names == [row["candidate_relation"] for row in second]
    assert names != [row["candidate_relation"] for row in third]
    assert len(names) == len(set(names)) == 4


def test_top_k_is_score_ranked() -> None:
    ranked = rank_valid_pool(
        [
            {"candidate_relation": "b", "candidate_score": -0.2, "negative_mass": 0.4},
            {"candidate_relation": "a", "candidate_score": -0.2, "negative_mass": 0.4},
            {"candidate_relation": "c", "candidate_score": -1.0, "negative_mass": 0.2},
        ]
    )
    assert [row["candidate_relation"] for row in ranked] == ["a", "b", "c"]
    assert [row["candidate_relation"] for row in sample_top_k(ranked, 2)] == ["a", "b"]


def test_coverage_adaptive_k_follows_per_qa_mass_not_fixed_nine() -> None:
    concentrated = _rows(
        qa_id="task_qa:0",
        scores={
            "concept:gold": -0.10,
            "concept:hard": -0.11,
            "concept:mid": -2.0,
            "concept:easy": -3.0,
            "concept:other": -4.0,
        },
    )
    diffuse = _rows(
        qa_id="task_qa:1",
        scores={
            "concept:gold": -1.0,
            "concept:a": -1.05,
            "concept:b": -1.10,
            "concept:c": -1.15,
            "concept:d": -1.20,
            "concept:e": -1.25,
            "concept:f": -1.30,
        },
    )
    tau = 0.50
    concentrated_k = decide_coverage_k(
        [
            float(row["negative_mass"])
            for row in sorted(
                shared_valid_pool(concentrated),
                key=lambda row: (-float(row["candidate_score"]), str(row["candidate_relation"])),
            )
        ],
        tau=tau,
        k_min=1,
        k_max=5,
    )
    diffuse_k = decide_coverage_k(
        [
            float(row["negative_mass"])
            for row in sorted(
                shared_valid_pool(diffuse),
                key=lambda row: (-float(row["candidate_score"]), str(row["candidate_relation"])),
            )
        ],
        tau=tau,
        k_min=1,
        k_max=5,
    )
    assert concentrated_k == 1
    assert diffuse_k > concentrated_k
    sampled = sample_qa_variants(
        concentrated,
        k_fixed=3,
        tau=tau,
        k_min=1,
        k_max=5,
        rng=random.Random(2026),
        seed=2026,
    )
    assert sampled["coverage_adaptive_k"]["k"] == 1
    assert sampled["coverage_adaptive_k"]["k"] != sampled["top_k_hard"]["k"]
    assert sampled["coverage_adaptive_k"]["covered_negative_mass"] >= tau - 1e-12


def test_coverage_tau_is_median_top9_mass() -> None:
    masses = [0.10, 0.20, 0.30, 0.40, 0.50]
    assert abs(propose_coverage_tau(masses) - 0.30) < 1e-12


def test_freeze_writes_three_shared_pool_manifests(tmp_path: Path) -> None:
    scores_path = tmp_path / "candidate_scores.jsonl"
    metadata_path = tmp_path / "metadata.json"
    output_dir = tmp_path / "samplers"
    groups = [
        _rows(
            qa_id="task_qa:0",
            scores={
                "concept:gold": -0.2,
                "concept:hard": -0.3,
                "concept:mid": -0.8,
                "concept:easy": -2.0,
            },
        ),
        _rows(
            qa_id="task_qa:1",
            scores={
                "concept:gold": -1.0,
                "concept:a": -1.05,
                "concept:b": -1.2,
                "concept:c": -1.4,
                "concept:d": -1.8,
            },
        ),
    ]
    with scores_path.open("w", encoding="utf-8") as stream:
        for group in groups:
            for row in group:
                stream.write(json.dumps(row) + "\n")
    metadata_path.write_text(json.dumps({"filter_splits": ["train"]}), encoding="utf-8")
    summary = freeze_shared_pool_samplers(
        scores_path=scores_path,
        output_dir=output_dir,
        metadata_path=metadata_path,
        k_fixed=2,
        k_min=1,
        k_max=3,
        seed=2026,
    )
    assert summary["n_qa"] == 2
    assert summary["filter_splits"] == ["train"]
    for variant in ("random_k", "top_k_hard", "coverage_adaptive_k"):
        path = output_dir / f"{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert {row["question_id"] for row in rows} == {"task_qa:0", "task_qa:1"}
        for row in rows:
            assert "concept:gold" not in row["negative_relations"]
            assert len(row["negative_relations"]) == len(set(row["negative_relations"]))
            assert row["negative_relations"] == (
                row["hard_negative_relations"] + row["uniform_negative_relations"]
                if variant != "random_k"
                else row["uniform_negative_relations"]
            )
    policy = json.loads((output_dir / "policy.json").read_text(encoding="utf-8"))
    assert policy["coverage_tau_source"] == "median_top9_negative_mass"
    assert 0.0 < policy["coverage_tau"] <= 1.0
    assert (output_dir / "FROZEN_SAMPLERS.md").is_file()


def test_freeze_rejects_all_split_dump(tmp_path: Path) -> None:
    scores_path = tmp_path / "candidate_scores.jsonl"
    metadata_path = tmp_path / "metadata.json"
    rows = _rows(
        qa_id="task_qa:0",
        scores={"concept:gold": -0.2, "concept:hard": -0.3, "concept:easy": -1.0},
    )
    with scores_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    metadata_path.write_text(
        json.dumps({"filter_splits": ["train", "valid", "test"]}),
        encoding="utf-8",
    )
    try:
        freeze_shared_pool_samplers(
            scores_path=scores_path,
            output_dir=tmp_path / "out",
            metadata_path=metadata_path,
        )
    except ValueError as exc:
        assert "train-only" in str(exc)
    else:
        raise AssertionError("all-split dump should be rejected")
