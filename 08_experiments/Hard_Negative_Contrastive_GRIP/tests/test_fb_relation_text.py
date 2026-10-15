"""FB text relations must receive negatives without treating entities as relations."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import build_qa_assets_from_task_texts


def test_explicit_text_mapping_preserves_nell_default(tmp_path):
    (tmp_path / "train.txt").write_text("a /z b\nc /a d\ne /z f\n")
    mapping = tmp_path / "rel2text.json"
    mapping.write_text(json.dumps({"/z": "works for", "/a": "born in"}))
    assert load_train_relation_order(tmp_path) == ["/z", "/a"]
    assert load_train_relation_order(tmp_path, relation_text_file=mapping) == ["born in", "works for"]
    mapping.write_text(json.dumps({"/z": "same", "/a": "same"}))
    assert load_train_relation_order(tmp_path, relation_text_file=mapping) == ["same"]
    mapping.write_text(json.dumps({"/z": "", "/a": "valid"}))
    with pytest.raises(ValueError, match="non-empty"):
        load_train_relation_order(tmp_path, relation_text_file=mapping)


def test_spaced_relation_answers_receive_train_only_negatives():
    order = ["born in", "works for"] + [f"relation number {i}" for i in range(10)]
    samples = [
        "What is the relation between Alice and London? <answer>born in</answer>",
        "Who wrote this book? <answer>James Fenimore Cooper</answer>",
        "Which entity has this relation? <answer>works for</answer>",
        "What is the relation between A and B? <answer>unknown relation</answer>",
    ]
    _, metas = build_qa_assets_from_task_texts(samples, seed=2026, relation_order=order)
    negatives = metas[0]["listed_relations"]
    assert len(negatives) == len(set(negatives)) == 9
    assert "born in" not in negatives
    assert set(negatives) <= set(order)
    assert all(not meta["listed_relations"] for meta in metas[1:])
    _, again = build_qa_assets_from_task_texts(samples, seed=2026, relation_order=order)
    assert again == metas
