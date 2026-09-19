from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.embed_negatives import (  # noqa: E402
    RelationNeighborIndex,
    cosine_similarity_matrix,
    l2_normalize,
    mean_offdiag_cosine,
    load_relation_embeddings,
    sample_embed_negatives,
    save_relation_embeddings,
    softmax,
    top_neighbors,
    weighted_sample_without_replacement,
)


def test_cosine_ranks_nearer_vectors_first() -> None:
    relations = ["gold", "near", "far"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    neighbors = top_neighbors("gold", relations, embeddings, k=2)
    assert [row["relation"] for row in neighbors] == ["near", "far"]
    assert neighbors[0]["cosine"] > neighbors[1]["cosine"]


def test_embed_negatives_prefer_similar_and_exclude_gold() -> None:
    relations = ["gold", "near_a", "near_b", "mid", "far_a", "far_b"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.6, 0.4],
            [0.0, 1.0],
            [0.1, 0.99],
        ]
    )
    sampled = sample_embed_negatives(
        "gold",
        relations,
        embeddings,
        k=2,
        rng=np.random.RandomState(2026),
        pool_size=3,
        temperature=0.05,
    )
    again = sample_embed_negatives(
        "gold",
        relations,
        embeddings,
        k=2,
        rng=np.random.RandomState(2026),
        pool_size=3,
        temperature=0.05,
    )
    assert sampled == again
    assert "gold" not in sampled
    assert len(sampled) == 2
    assert set(sampled).issubset({"near_a", "near_b", "mid"})
    assert "far_a" not in sampled
    assert "far_b" not in sampled


def test_neighbor_index_caches_unique_golds_not_gram_matrix() -> None:
    relations = [f"rel_{i}" for i in range(32)]
    rng = np.random.RandomState(0)
    embeddings = rng.normal(size=(32, 8))
    index = RelationNeighborIndex(relations, embeddings, pool_size=5)
    first = index.pool("rel_0")
    second = index.pool("rel_0")
    assert first is second
    assert len(index._pools) == 1
    index.pool("rel_1")
    assert set(index._pools) == {"rel_0", "rel_1"}
    sampled = index.sample("rel_0", k=3, rng=np.random.RandomState(1))
    assert len(sampled) == 3
    assert "rel_0" not in sampled


def test_mean_offdiag_matches_dense_gram_without_materializing_it() -> None:
    embeddings = l2_normalize(np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]))
    dense = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(dense, np.nan)
    assert np.isclose(mean_offdiag_cosine(embeddings), np.nanmean(dense))


def test_weighted_sample_is_deterministic() -> None:
    items = ["a", "b", "c", "d"]
    weights = np.array([0.5, 0.3, 0.15, 0.05])
    first = weighted_sample_without_replacement(
        items, weights, k=3, rng=np.random.RandomState(7)
    )
    second = weighted_sample_without_replacement(
        items, weights, k=3, rng=np.random.RandomState(7)
    )
    other = weighted_sample_without_replacement(
        items, weights, k=3, rng=np.random.RandomState(8)
    )
    assert first == second
    assert len(first) == 3
    assert first != other


def test_softmax_temperature_concentrates() -> None:
    scores = np.array([1.0, 0.1, 0.0])
    sharp = softmax(scores, 0.05)
    flat = softmax(scores, 10.0)
    assert sharp[0] > flat[0]
    assert np.isclose(sharp.sum(), 1.0)
    assert np.isclose(flat.sum(), 1.0)


def test_align_and_roundtrip_embedding_file(tmp_path: Path) -> None:
    relations = ["concept:atdate", "concept:worksfor", "concept:haswife"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "rels.npz"
    save_relation_embeddings(path, relations, embeddings, {"note": "unit"})
    loaded_rels, loaded_emb = load_relation_embeddings(path)
    assert loaded_rels == relations
    order = ["concept:haswife", "concept:atdate"]
    aligned = align_embeddings(order, loaded_rels, loaded_emb)
    assert aligned.shape == (2, 3)
    assert np.allclose(aligned[0], embeddings[2])
    assert np.allclose(aligned[1], embeddings[0])
    assert (path.with_suffix(".json")).is_file()
