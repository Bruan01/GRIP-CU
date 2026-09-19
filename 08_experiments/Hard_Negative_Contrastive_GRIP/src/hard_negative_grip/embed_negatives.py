"""Embed the train-graph relation vocabulary and sample similar negatives.

Vocabulary A is the official NELL23K train-relation insertion order (198).
Embeddings are L2-normalized vectors, one per relation. Training listed
negatives then prefer cosine-similar relations instead of a uniform
``process.py`` permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_EMBED_POOL_SIZE = 40
DEFAULT_EMBED_TEMPERATURE = 0.1
DEFAULT_NEIGHBOR_K = 10


def l2_normalize(embeddings: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape [n, dim], got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarities. Input may be unnormalized."""
    normalized = l2_normalize(embeddings)
    return normalized @ normalized.T


def align_embeddings(
    relation_order: list[str],
    stored_relations: list[str],
    stored_embeddings: np.ndarray,
) -> np.ndarray:
    """Reindex a stored embedding table onto ``relation_order``."""
    if len(stored_relations) != len(stored_embeddings):
        raise ValueError(
            f"stored relations ({len(stored_relations)}) != embeddings "
            f"({len(stored_embeddings)})"
        )
    index = {rel: i for i, rel in enumerate(stored_relations)}
    missing = [rel for rel in relation_order if rel not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} train relations missing from embedding file: {preview}"
        )
    rows = [index[rel] for rel in relation_order]
    return np.asarray(stored_embeddings, dtype=np.float64)[rows]


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    shifted = (np.asarray(scores, dtype=np.float64) - np.max(scores)) / temperature
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(scores), dtype=np.float64) / max(len(scores), 1)
    return weights / total


def weighted_sample_without_replacement(
    items: list[str],
    weights: np.ndarray,
    *,
    k: int,
    rng: np.random.RandomState,
) -> list[str]:
    remaining = list(items)
    remaining_weights = np.asarray(weights, dtype=np.float64).copy()
    if len(remaining) != len(remaining_weights):
        raise ValueError("items and weights must have the same length")
    chosen: list[str] = []
    take = min(k, len(remaining))
    for _ in range(take):
        if remaining_weights.sum() <= 0 or not np.isfinite(remaining_weights.sum()):
            remaining_weights = np.ones(len(remaining), dtype=np.float64)
        probs = remaining_weights / remaining_weights.sum()
        idx = int(rng.choice(len(remaining), p=probs))
        chosen.append(remaining.pop(idx))
        remaining_weights = np.delete(remaining_weights, idx)
    return chosen


def sample_embed_negatives(
    gold: str,
    relation_order: list[str],
    similarity: np.ndarray,
    *,
    k: int,
    rng: np.random.RandomState,
    pool_size: int = DEFAULT_EMBED_POOL_SIZE,
    temperature: float = DEFAULT_EMBED_TEMPERATURE,
) -> list[str]:
    """Sample ``k`` negatives, preferring cosine-similar train relations."""
    if gold not in relation_order:
        raise ValueError(f"gold relation absent from train vocabulary: {gold}")
    if k <= 0:
        return []
    gold_index = relation_order.index(gold)
    scores = []
    for index, rel in enumerate(relation_order):
        if index == gold_index:
            continue
        scores.append((float(similarity[gold_index, index]), rel))
    if not scores:
        return []
    scores.sort(key=lambda item: (-item[0], item[1]))
    pool = scores[: min(pool_size, len(scores))]
    items = [rel for _, rel in pool]
    weights = softmax(np.array([sim for sim, _ in pool], dtype=np.float64), temperature)
    return weighted_sample_without_replacement(items, weights, k=k, rng=rng)


def top_neighbors(
    gold: str,
    relation_order: list[str],
    similarity: np.ndarray,
    *,
    k: int = DEFAULT_NEIGHBOR_K,
) -> list[dict]:
    if gold not in relation_order:
        raise ValueError(f"gold relation absent from train vocabulary: {gold}")
    gold_index = relation_order.index(gold)
    ranked = sorted(
        (
            (float(similarity[gold_index, index]), rel)
            for index, rel in enumerate(relation_order)
            if index != gold_index
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [{"relation": rel, "cosine": sim} for sim, rel in ranked[:k]]


def sampled_cosine_stats(
    gold: str,
    negatives: list[str],
    relation_order: list[str],
    similarity: np.ndarray,
) -> dict:
    gold_index = relation_order.index(gold)
    values = [
        float(similarity[gold_index, relation_order.index(rel)])
        for rel in negatives
        if rel in relation_order
    ]
    if not values:
        return {"count": 0, "mean_cosine": 0.0}
    return {"count": len(values), "mean_cosine": float(np.mean(values))}


def save_relation_embeddings(
    path: Path,
    relations: list[str],
    embeddings: np.ndarray,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(
        path,
        relations=np.asarray(relations, dtype=object),
        embeddings=matrix,
    )
    sidecar = path.with_suffix(".json")
    payload = dict(metadata or {})
    payload.update(
        {
            "path": str(path),
            "n_relations": len(relations),
            "dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        }
    )
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_relation_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(Path(path), allow_pickle=True)
    relations = [str(item) for item in payload["relations"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(relations):
        raise ValueError(
            f"bad embedding table in {path}: relations={len(relations)} "
            f"embeddings={embeddings.shape}"
        )
    return relations, embeddings
