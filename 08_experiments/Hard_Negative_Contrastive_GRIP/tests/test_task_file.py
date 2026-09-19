from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    assistant_gold,
    build_qa_assets_from_task_texts,
    is_grip_task_file,
    is_relation_gold,
    match_train_relation,
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


def test_is_grip_task_file_requires_both_splits() -> None:
    assert is_grip_task_file({"context_samples": ["a"], "qa_samples": ["b"]})
    assert not is_grip_task_file({"context_samples": ["a"]})
    assert not is_grip_task_file([{"graph": {}}])


def test_assistant_gold_and_prefix_use_the_last_answer_span() -> None:
    text = _qa("what is the relation between a and b?", "concept:atdate")
    assert assistant_gold(text) == "concept:atdate"
    prefix = assistant_answer_prefix(text)
    assert prefix.endswith("<answer>")
    assert prefix.count("<answer>") == 2
    assert "concept:atdate" not in prefix


def test_relation_gold_keeps_nell_relations_and_drops_entities() -> None:
    rel = _qa("what is the relation between x and y?", "concept:worksfor")
    entity = _qa("which entity has the relation concept:worksfor to chrysler?", "concept_ceo_dieter_zetsche")
    yesno = _qa("Is Detroit a city?", "Yes")
    assert is_relation_gold(assistant_gold(rel), rel)
    assert not is_relation_gold(assistant_gold(entity), entity)
    assert not is_relation_gold(assistant_gold(yesno), yesno)


def test_sample_listed_negatives_are_deterministic_and_exclude_gold() -> None:
    vocab = [f"rel_{i}" for i in range(20)]
    import random

    first = sample_listed_negatives("rel_3", vocab, k=9, rng=random.Random(7))
    second = sample_listed_negatives("rel_3", vocab, k=9, rng=random.Random(7))
    other = sample_listed_negatives("rel_3", vocab, k=9, rng=random.Random(8))
    assert first == second
    assert "rel_3" not in first
    assert len(first) == 9
    assert first != other


def test_task_qa_assets_attach_negatives_only_to_relation_items() -> None:
    texts = [
        _qa("what is the relation between a and b?", "concept:atdate"),
        _qa("what is the relation between c and d?", "concept:worksfor"),
        _qa("Who wrote The Deerslayer?", "James Fenimore Cooper"),
        _qa("which entity has the relation concept:worksfor to chrysler?", "concept_ceo_dieter_zetsche"),
    ]
    out_texts, metas = build_qa_assets_from_task_texts(
        texts,
        seed=2026,
        listed_negative_k=1,
        listed_negative_source="qa_vocab",
    )
    assert out_texts == texts
    assert metas[0]["positive_relation"] == "concept:atdate"
    assert metas[0]["listed_relations"] == ["concept:worksfor"]
    assert metas[1]["listed_relations"] == ["concept:atdate"]
    assert metas[2]["listed_relations"] == []
    assert metas[3]["listed_relations"] == []
    assert metas[0]["prefix_text"].endswith("<answer>")


def test_train_graph_negatives_use_process_py_and_exclude_aliased_gold() -> None:
    import numpy as np
    from hard_negative_grip.official_lists import official_negatives, sample_official_candidates

    order = [
        "concept:atdate",
        "concept:worksfor",
        "concept:haswife",
        "concept:citycapitalofcountry",
    ]
    texts = [
        _qa("what is the relation between a and b?", "concept:atdate"),
        _qa("what is the relation between c and d?", "worksfor"),
        _qa("Who wrote The Deerslayer?", "James Fenimore Cooper"),
        _qa("what is the relation between e and f?", "not_a_train_relation"),
    ]
    stream = np.random.RandomState(2026)
    expected_first = official_negatives("concept:atdate", order, way=4, rng=stream)
    expected_second = official_negatives("concept:worksfor", order, way=4, rng=stream)

    out_texts, metas = build_qa_assets_from_task_texts(
        texts,
        seed=2026,
        listed_negative_k=3,
        listed_negative_source="train_graph",
        relation_order=order,
    )
    assert out_texts == texts
    assert metas[0]["listed_relations"] == expected_first
    assert "concept:atdate" not in metas[0]["listed_relations"]
    assert metas[0]["matched_train_relation"] == "concept:atdate"
    assert metas[1]["positive_relation"] == "worksfor"
    assert metas[1]["matched_train_relation"] == "concept:worksfor"
    assert metas[1]["listed_relations"] == expected_second
    assert "concept:worksfor" not in metas[1]["listed_relations"]
    assert "worksfor" not in metas[1]["listed_relations"]
    assert metas[2]["listed_relations"] == []
    assert metas[3]["listed_relations"] == []
    assert metas[3]["matched_train_relation"] is None
    assert all(rel in order for rel in metas[0]["listed_relations"] + metas[1]["listed_relations"])

    np.random.seed(7)
    global_first = sample_official_candidates("concept:atdate", order, way=4)
    private_first = sample_official_candidates(
        "concept:atdate", order, way=4, rng=np.random.RandomState(7)
    )
    assert global_first == private_first


def test_train_graph_requires_relation_order() -> None:
    texts = [_qa("what is the relation between a and b?", "concept:atdate")]
    try:
        build_qa_assets_from_task_texts(texts, seed=2026, listed_negative_source="train_graph")
    except ValueError as exc:
        assert "relation_order" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_match_train_relation_accepts_concept_alias() -> None:
    index = train_relation_alias_index(["concept:atdate", "concept:worksfor"])
    assert match_train_relation("concept:atdate", index) == "concept:atdate"
    assert match_train_relation("worksfor", index) == "concept:worksfor"
    assert match_train_relation("missing", index) is None


def test_embed_sim_negatives_use_train_vocab_and_prefer_neighbors() -> None:
    import numpy as np

    order = [
        "concept:atdate",
        "concept:worksfor",
        "concept:haswife",
        "concept:citycapitalofcountry",
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.05, 0.99],
        ]
    )
    texts = [
        _qa("what is the relation between a and b?", "concept:atdate"),
        _qa("what is the relation between c and d?", "worksfor"),
        _qa("Who wrote The Deerslayer?", "James Fenimore Cooper"),
        _qa("what is the relation between e and f?", "not_a_train_relation"),
    ]
    _, metas = build_qa_assets_from_task_texts(
        texts,
        seed=2026,
        listed_negative_k=2,
        listed_negative_source="embed_sim",
        relation_order=order,
        relation_embeddings=embeddings,
        embed_pool_size=2,
        embed_sample_temperature=0.05,
    )
    assert metas[0]["matched_train_relation"] == "concept:atdate"
    assert "concept:atdate" not in metas[0]["listed_relations"]
    assert set(metas[0]["listed_relations"]) == {
        "concept:worksfor",
        "concept:citycapitalofcountry",
    }
    assert metas[1]["matched_train_relation"] == "concept:worksfor"
    assert "concept:worksfor" not in metas[1]["listed_relations"]
    assert set(metas[1]["listed_relations"]) == {
        "concept:atdate",
        "concept:citycapitalofcountry",
    }
    assert metas[2]["listed_relations"] == []
    assert metas[3]["listed_relations"] == []
    assert all(rel in order for rel in metas[0]["listed_relations"] + metas[1]["listed_relations"])


def test_embed_sim_requires_embeddings() -> None:
    texts = [_qa("what is the relation between a and b?", "concept:atdate")]
    try:
        build_qa_assets_from_task_texts(
            texts,
            seed=2026,
            listed_negative_source="embed_sim",
            relation_order=["concept:atdate", "concept:worksfor"],
        )
    except ValueError as exc:
        assert "relation_embeddings" in str(exc)
    else:
        raise AssertionError("expected ValueError")
