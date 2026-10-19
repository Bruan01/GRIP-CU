from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip import (  # noqa: E402
    candidate_infonce_loss,
    hits_at_k,
    normalized_continuation_log_likelihood,
    parse_listed_relations,
    reciprocal_rank,
    summarize_candidate_scores,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    listed_relations_from_sample,
    load_train_relation_order,
    rewrite_question_with_official_list,
    sample_official_candidates,
    train_relation_insertion_order,
)


def test_prompt_candidates_and_official_lists_are_deterministic() -> None:
    question = (
        "What is the relation between word node alice and word node bob? "
        "Selected from the following candidate answers: rel_a; rel_b; rel_c."
    )
    assert parse_listed_relations(question) == ["rel_a", "rel_b", "rel_c"]
    order = train_relation_insertion_order(
        [("h", "rel_a", "t"), ("h", "rel_b", "t"), ("h", "rel_a", "u")]
    )
    assert order == ["rel_a", "rel_b"]
    import numpy as np

    np.random.seed(2026)
    first = sample_official_candidates("rel_a", order, way=2)
    np.random.seed(2026)
    assert first == sample_official_candidates("rel_a", order, way=2)
    sample = {"source_node": "alice", "target_node": "bob", "question": "old"}
    rewritten = rewrite_question_with_official_list(sample, first)
    assert listed_relations_from_sample(rewritten) == first


def test_nell23k_train_graph_has_198_concept_relations_when_available() -> None:
    train_file = DEFAULT_RAW_NELL23K / "train.txt"
    if not train_file.is_file():
        return
    order = load_train_relation_order(DEFAULT_RAW_NELL23K)
    assert len(order) == 198
    assert len(set(order)) == 198
    assert all(relation.startswith("concept:") for relation in order)


def test_candidate_infonce_and_metrics() -> None:
    positive = torch.tensor([3.0], requires_grad=True)
    negatives = torch.tensor([[1.0, 0.5]], requires_grad=True)
    assert candidate_infonce_loss(positive, negatives) < candidate_infonce_loss(
        torch.tensor([0.5]), negatives.detach()
    )
    with pytest.raises(ValueError):
        candidate_infonce_loss(positive, negatives, temperature=0.0)
    assert reciprocal_rank(2.0, [1.0, 0.0]) == 1.0
    assert hits_at_k(2.0, [1.0, 0.0], 1) == 1.0
    summary = summarize_candidate_scores(
        [{"positive_score": 2.0, "negative_scores": [1.0], "kind": "error_memory"}],
        ks=(1,),
    )
    assert summary["error_memory.hits@1"] == 1.0


def test_normalized_continuation_scoring_ignores_prefix() -> None:
    input_ids = torch.tensor([[9, 8, 1, 2], [7, 6, 3, 4]])
    logits = torch.full((2, 4, 10), -10.0)
    logits[0, 1, 1] = 4.0
    logits[0, 2, 2] = 4.0
    logits[1, 1, 3] = 4.0
    logits[1, 2, 4] = 4.0
    scores = normalized_continuation_log_likelihood(
        logits, input_ids, torch.tensor([2, 2])
    )
    assert torch.allclose(scores[0], scores[1])
