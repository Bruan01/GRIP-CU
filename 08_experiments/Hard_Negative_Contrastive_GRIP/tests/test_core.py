from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip import (  # noqa: E402
    adapter_contrastive_loss,
    candidate_infonce_loss,
    generate_hard_negatives,
    generate_random_negatives,
    hits_at_k,
    margin_ranking_loss,
    reciprocal_rank,
    summarize_candidate_scores,
    normalized_continuation_log_likelihood,
)


def test_candidate_families_are_filtered_and_deterministic() -> None:
    graph = [
        ("alice", "likes", "bob"),
        ("alice", "visits", "dave"),
        ("bob", "knows", "carol"),
        ("dave", "knows", "erin"),
        ("carol", "likes", "dave"),
    ]
    all_known = graph + [("alice", "likes", "carol")]
    kwargs = dict(
        positive_relation="likes",
        head="alice",
        tail="bob",
        relations=["likes", "knows", "visits", "owns"],
        graph_triples=graph,
        all_known_triples=all_known,
        num_per_kind=2,
        path_hops=2,
    )
    first = generate_hard_negatives(**kwargs)
    second = generate_hard_negatives(**kwargs)

    assert first == second
    assert all(candidate.relation != "likes" for candidate in first)
    assert all(
        (candidate.head, candidate.relation, candidate.tail) not in set(all_known)
        for candidate in first
    )
    assert {candidate.kind for candidate in first} == {
        "uniform_relation",
        "tail_range_relation",
        "path_relation",
    }
    assert all(
        candidate.kind != "path_relation" or candidate.structural_distance is not None
        for candidate in first
    )


def test_tail_range_relation_shares_tails_with_positive() -> None:
    graph = [
        ("alice", "likes", "bob"),
        ("alice", "likes", "dave"),
        ("carol", "knows", "bob"),
        ("erin", "visits", "frank"),
    ]
    kwargs = dict(
        positive_relation="likes",
        head="alice",
        tail="bob",
        relations=["likes", "knows", "visits"],
        graph_triples=graph,
        all_known_triples=graph,
        num_per_kind=4,
        path_hops=2,
    )
    hard = generate_hard_negatives(**kwargs)
    tail_range = {
        candidate.relation
        for candidate in hard
        if candidate.kind == "tail_range_relation"
    }
    # "knows" shares tail "bob" with "likes"; "visits" does not.
    assert "knows" in tail_range
    assert "visits" not in tail_range


def test_random_control_is_filtered_and_seeded() -> None:
    graph = [("a", "r", "b"), ("b", "s", "c")]
    kwargs = dict(
        positive_relation="r",
        head="a",
        tail="b",
        relations=["r", "s", "t"],
        graph_triples=graph,
        all_known_triples=graph + [("a", "r", "c")],
        num_negatives=3,
        seed=12,
    )
    first = generate_random_negatives(**kwargs)
    second = generate_random_negatives(**kwargs)
    assert first == second
    assert all(item.kind == "random" for item in first)
    assert all(
        (item.head, item.relation, item.tail) not in set(kwargs["all_known_triples"])
        for item in first
    )


def test_zero_candidates_and_invalid_arguments() -> None:
    kwargs = dict(
        positive_relation="r",
        head="a",
        tail="b",
        relations=["r", "s"],
        graph_triples=[("a", "r", "b")],
    )
    assert generate_hard_negatives(num_per_kind=0, **kwargs) == []
    try:
        generate_hard_negatives(path_hops=0, **kwargs)
    except ValueError as error:
        assert "path_hops" in str(error)
    else:
        raise AssertionError("path_hops=0 must be rejected")


def test_losses_prefer_a_higher_positive_score() -> None:
    positive = torch.tensor([3.0], requires_grad=True)
    negatives = torch.tensor([[1.0, 0.5]], requires_grad=True)
    low_positive = torch.tensor([0.5])

    info_good = candidate_infonce_loss(positive, negatives)
    info_bad = candidate_infonce_loss(low_positive, negatives.detach())
    margin_good = margin_ranking_loss(positive, negatives)
    adapter_good = adapter_contrastive_loss(positive, negatives)

    assert info_good < info_bad
    assert margin_good.item() == 0.0
    assert adapter_good.item() == info_good.item()
    info_good.backward()
    assert positive.grad is not None
    assert negatives.grad is not None


def test_candidate_metrics_and_loss_input_validation() -> None:
    positive = torch.tensor([1.0])
    negatives = torch.tensor([[0.0]])
    for function, kwargs in (
        (candidate_infonce_loss, {"temperature": 0.0}),
        (margin_ranking_loss, {"margin": -0.1}),
    ):
        try:
            function(positive, negatives, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid loss hyperparameter must be rejected")

    assert reciprocal_rank(2.0, [1.0, 0.0]) == 1.0
    assert hits_at_k(2.0, [1.0, 0.0], 1) == 1.0
    assert hits_at_k(2.0, [3.0, 0.0], 1) == 0.0
    summary = summarize_candidate_scores(
        [
            {"positive_score": 2.0, "negative_scores": [1.0], "kind": "path_relation"},
            {"positive_score": 0.0, "negative_scores": [1.0], "kind": "random"},
        ],
        ks=(1,),
    )
    assert summary["count"] == 2
    assert summary["path_relation.hits@1"] == 1.0
    assert summary["random.hits@1"] == 0.0


def test_normalized_continuation_scoring_ignores_prefix() -> None:
    input_ids = torch.tensor([[9, 8, 1, 2], [7, 6, 3, 4]])
    logits = torch.full((2, 4, 10), -10.0)
    logits[0, 1, 1] = 4.0
    logits[0, 2, 2] = 4.0
    logits[1, 1, 3] = 4.0
    logits[1, 2, 4] = 4.0
    from hard_negative_grip import normalized_continuation_log_likelihood

    scores = normalized_continuation_log_likelihood(
        logits,
        input_ids,
        torch.tensor([2, 2]),
    )
    assert torch.allclose(scores[0], scores[1])
