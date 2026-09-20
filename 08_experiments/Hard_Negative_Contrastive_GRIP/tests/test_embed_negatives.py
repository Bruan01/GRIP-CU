from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.embed_negatives import (  # noqa: E402
    RelationNeighborIndex,
    align_embeddings,
    all_but_the_top,
    apply_linear_transform,
    cosine_similarity_matrix,
    geometry_summary,
    l2_normalize,
    mean_offdiag_cosine,
    load_relation_embeddings,
    pca_partial_whiten,
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


def _cone_table(n: int = 24, dim: int = 6) -> np.ndarray:
    """Relation embeddings like Stage 1: a shared direction plus a small per-row offset."""
    rng = np.random.RandomState(2026)
    shared = np.ones(dim) * 3.0
    return shared + rng.normal(scale=0.05, size=(n, dim))


def _top_energy_fraction(centered: np.ndarray, k: int) -> float:
    singular = np.linalg.svd(centered, compute_uv=False)
    return float((singular[:k] ** 2).sum() / (singular**2).sum())


def test_center_removes_the_shared_cone() -> None:
    table = _cone_table()
    raw = l2_normalize(table)
    centered = l2_normalize(apply_linear_transform(table, mode="center"))
    dense = cosine_similarity_matrix(centered)
    np.fill_diagonal(dense, np.nan)
    assert mean_offdiag_cosine(raw) > 0.9
    # The residual has no common direction left: cosines scatter around zero.
    assert abs(mean_offdiag_cosine(centered)) < 0.1
    assert abs(np.nanmean(dense)) < 0.1


def test_partial_whitening_alpha_zero_is_plain_pca() -> None:
    table = _cone_table()
    projected = pca_partial_whiten(table, k=4, alpha=0.0)
    centered = table - table.mean(axis=0, keepdims=True)
    assert projected.shape == (table.shape[0], 4)
    assert np.isclose(
        (projected**2).sum(), (centered**2).sum() * _top_energy_fraction(centered, 4)
    )


def test_full_whitening_at_rank_minus_one_erases_the_ranking() -> None:
    """k=n-1 with alpha=1 makes every row equidistant: the sampler goes uniform."""
    rng = np.random.RandomState(0)
    table = rng.normal(size=(8, 16))
    whitened = pca_partial_whiten(table, k=7, alpha=1.0)
    normalized = l2_normalize(whitened)
    norms = np.linalg.norm(whitened, axis=1)
    gram = normalized @ normalized.T
    off = ~np.eye(8, dtype=bool)
    assert np.allclose(norms, norms[0])  # whitening equalizes every row norm
    assert gram[off].std() < 1e-9  # so all off-diagonal cosines are identical
    report = geometry_summary(normalized, pool_size=6, temperature=0.1)
    # No relation is preferred over another: the sampler is uniform in the pool.
    # (Top-1 is then decided by float tie-breaking, so hub counts are meaningless.)
    assert np.isclose(report["mean_effective_negatives"], 6.0)
    assert report["std_offdiag_cosine"] < 1e-9


def test_partial_whitening_sharpens_the_softmax_versus_raw() -> None:
    table = _cone_table(n=32, dim=8)
    raw = l2_normalize(table)
    whitened = l2_normalize(pca_partial_whiten(table, k=16, alpha=0.5))
    raw_report = geometry_summary(raw, pool_size=10, temperature=0.1)
    whitened_report = geometry_summary(whitened, pool_size=10, temperature=0.1)
    # Raw cosines sit in a narrow band, so the sampler is effectively uniform.
    assert raw_report["mean_effective_negatives"] > 0.98 * 10
    assert whitened_report["mean_effective_negatives"] < raw_report["mean_effective_negatives"]


def test_all_but_the_top_drops_the_leading_direction() -> None:
    table = _cone_table(n=16, dim=5)
    centered = all_but_the_top(table, remove=1)
    _, singular, basis = np.linalg.svd(
        table - table.mean(axis=0, keepdims=True), full_matrices=False
    )
    assert np.allclose(centered @ basis[0], 0.0, atol=1e-9)
    assert np.isclose((centered**2).sum(), (singular**2).sum() - singular[0] ** 2)


def test_apply_linear_transform_rejects_unknown_mode() -> None:
    table = _cone_table(n=4, dim=3)
    assert np.allclose(apply_linear_transform(table, mode="raw"), table)
    try:
        apply_linear_transform(table, mode="nope")
    except ValueError as error:
        assert "nope" in str(error)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("apply_linear_transform accepted an unknown mode")


def test_geometry_summary_matches_dense_gram_statistics() -> None:
    table = l2_normalize(_cone_table(n=20, dim=6))
    report = geometry_summary(table, pool_size=5, temperature=0.2)
    dense = cosine_similarity_matrix(table)
    off = ~np.eye(20, dtype=bool)
    assert np.isclose(report["mean_offdiag_cosine"], dense[off].mean())
    assert np.isclose(report["std_offdiag_cosine"], dense[off].std())
    assert report["n_relations"] == 20
    assert 1.0 <= report["mean_effective_negatives"] <= 5.0
