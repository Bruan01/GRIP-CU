from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    assistant_gold,
    build_qa_assets_from_task_texts,
    canonical_rollout_candidate,
    is_grip_task_file,
    is_relation_gold,
    load_graph_record,
    parse_model_answer,
    sample_listed_negatives,
    train_relation_alias_index,
)


def _qa(question: str, answer: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nGiven the context graph titled nell23k, please answer "
        f"the following question: {question} Response in the following format:"
        "<answer>[answer]</answer><|im_end|>\n"
        f"<|im_start|>assistant\n<answer>{answer}</answer><|im_end|>\n"
    )


def test_standard_graph_record_is_the_only_eval_contract(tmp_path: Path) -> None:
    record = {
        "title": "nell23k",
        "graph": {"edge_list": [], "edge_index": [], "node_list": []},
        "questions": ["What is the relation between a and b?"],
        "answers": ["concept:r0"],
        "id": "nell23k",
    }
    path = tmp_path / "processed_test.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    assert load_graph_record(path)["answers"] == ["concept:r0"]
    path.write_text(json.dumps({"legacy_questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="standard GRIP graph record"):
        load_graph_record(path)


def test_task_file_detection_and_answer_parsing() -> None:
    text = _qa("what is the relation between a and b?", "concept:atdate")
    assert is_grip_task_file({"context_samples": ["a"], "qa_samples": [text]})
    assert assistant_gold(text) == "concept:atdate"
    assert parse_model_answer("noise <answer>concept:worksfor</answer>") == "concept:worksfor"
    assert parse_model_answer("concept:atdate") == "concept:atdate"
    assert assistant_answer_prefix(text).endswith("<answer>")


def test_relation_detection_excludes_entity_and_yes_no_answers() -> None:
    relation = _qa("what is the relation between x and y?", "concept:worksfor")
    entity = _qa("which entity has relation concept:worksfor to chrysler?", "concept_ceo")
    yes_no = _qa("Is Detroit a city?", "Yes")
    assert is_relation_gold(assistant_gold(relation), relation)
    assert not is_relation_gold(assistant_gold(entity), entity)
    assert not is_relation_gold(assistant_gold(yes_no), yes_no)


def test_fb_text_relation_aliases_are_canonicalized() -> None:
    question = _qa("what is the relation between x and y?", "Adjoining Locations")
    alias_index = train_relation_alias_index(["Adjoining Locations", "People Person"])
    assert canonical_rollout_candidate(
        "Adjoining Locations", relation_alias_index=alias_index
    ) == "Adjoining Locations"
    assert is_relation_gold(
        assistant_gold(question), question, relation_alias_index=alias_index
    )
    assert not is_relation_gold(
        "not a train relation", question, relation_alias_index=alias_index
    )


def test_random_control_uses_train_relation_vocabulary() -> None:
    order = ["concept:atdate", "concept:worksfor", "concept:haswife", "concept:capital"]
    texts = [
        _qa("what is the relation between a and b?", "concept:atdate"),
        _qa("what is the relation between c and d?", "worksfor"),
    ]
    _, metas = build_qa_assets_from_task_texts(
        texts,
        seed=2026,
        listed_negative_k=2,
        listed_negative_source="train_graph",
        relation_order=order,
    )
    assert len(metas[0]["listed_relations"]) == 2
    assert "concept:atdate" not in metas[0]["listed_relations"]
    assert metas[1]["matched_train_relation"] == "concept:worksfor"
    assert all(relation in order for relation in metas[0]["listed_relations"])


def test_error_memory_manifest_is_attached_by_original_task_id() -> None:
    texts = [
        _qa("what is the relation between a and b?", "concept:gold"),
        _qa("what is the relation between c and d?", "concept:other"),
    ]
    manifest = {
        "task_qa:0": {
            "question_id": "task_qa:0",
            "positive_relation": "concept:gold",
            "hard_negative_relations": ["concept:hard"],
            "uniform_negative_relations": [],
            "negative_relations": ["concept:hard"],
        },
        "task_qa:1": {
            "question_id": "task_qa:1",
            "positive_relation": "concept:other",
            "hard_negative_relations": ["concept:hard"],
            "uniform_negative_relations": [],
            "negative_relations": ["concept:hard"],
        },
    }
    _, metas = build_qa_assets_from_task_texts(
        texts,
        seed=2026,
        listed_negative_source="error_memory",
        relation_order=["concept:gold", "concept:other", "concept:hard"],
        error_memory_manifest=manifest,
        question_ids=["task_qa:0", "task_qa:1"],
    )
    assert metas[0]["listed_relations"] == ["concept:hard"]
    assert metas[1]["listed_relations"] == ["concept:hard"]


def test_basic_helpers_are_deterministic() -> None:
    vocab = [f"rel_{index}" for index in range(20)]
    first = sample_listed_negatives("rel_3", vocab, k=9, rng=random.Random(7))
    second = sample_listed_negatives("rel_3", vocab, k=9, rng=random.Random(7))
    assert first == second
    assert "rel_3" not in first
    assert train_relation_alias_index(["concept:atdate"])["atdate"] == "concept:atdate"
