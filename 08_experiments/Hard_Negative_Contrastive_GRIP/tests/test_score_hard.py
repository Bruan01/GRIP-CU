from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.score_hard import select_score_hard_negatives  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    build_qa_assets_from_task_texts,
    known_pair_relations,
    load_score_hard_manifest,
    question_entity_pair,
)


def _qa(question: str, answer: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Given the context graph titled nell23k, please answer the following question: {question}"
        " Response in the following format:<answer>[answer]</answer>\n"
        f"<|im_start|>assistant\n<answer>{answer}</answer>"
    )


def test_score_hard_excludes_known_relations_but_scores_all_candidates() -> None:
    order = ["concept:gold", "concept:true_other", "concept:hard", "concept:uniform"]
    scores = {
        "concept:gold": 0.1,
        "concept:true_other": 0.99,
        "concept:hard": 0.8,
        "concept:uniform": 0.2,
    }
    hard, uniform = select_score_hard_negatives(
        "concept:gold",
        order,
        scores,
        total_k=2,
        hard_k=1,
        rng=random.Random(2026),
        excluded_relations={"concept:true_other"},
    )
    assert hard == ["concept:hard"]
    assert len(uniform) == 1
    assert "concept:true_other" not in hard + uniform
    assert set(scores) == set(order)


def test_known_pair_relations_reads_all_splits(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("a rel_train b\n", encoding="utf-8")
    (tmp_path / "valid.txt").write_text("a rel_valid b\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("a rel_test b\n", encoding="utf-8")
    assert known_pair_relations(tmp_path) == {
        ("a", "b"): {"rel_train", "rel_valid", "rel_test"}
    }


def test_score_hard_manifest_is_used_verbatim(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "question_id": "task_qa:0",
        "positive_relation": "concept:gold",
        "hard_negative_relations": ["concept:hard"],
        "uniform_negative_relations": ["concept:uniform"],
        "negative_relations": ["concept:hard", "concept:uniform"],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_score_hard_manifest(manifest)
    texts, metas = build_qa_assets_from_task_texts(
        [_qa("What is the relation between a and b?", "concept:gold")],
        seed=2026,
        listed_negative_source="score_hard",
        relation_order=["concept:gold", "concept:hard", "concept:uniform"],
        score_hard_manifest=loaded,
    )
    assert texts[0].endswith("<answer>concept:gold</answer>")
    assert metas[0]["listed_relations"] == ["concept:hard", "concept:uniform"]


def test_question_entity_pair_strips_word_node_prefix() -> None:
    text = _qa("What is the relation between word node a and word node b?", "concept:gold")
    assert question_entity_pair(text) == ("a", "b")


def test_rollout_hard_manifest_keeps_oov_negative(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "question_id": "task_qa:0",
        "positive_relation": "concept:gold",
        "hard_negative_relations": ["concept:unseenrelation"],
        "uniform_negative_relations": ["concept:uniform"],
        "negative_relations": ["concept:unseenrelation", "concept:uniform"],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_score_hard_manifest(manifest)
    _, metas = build_qa_assets_from_task_texts(
        [_qa("What is the relation between a and b?", "concept:gold")],
        seed=2026,
        listed_negative_source="rollout_hard",
        relation_order=["concept:gold", "concept:uniform"],
        score_hard_manifest=loaded,
    )
    assert metas[0]["listed_relations"] == [
        "concept:unseenrelation",
        "concept:uniform",
    ]
