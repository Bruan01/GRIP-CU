"""Embed the train-graph relation vocabulary and sample similar negatives.

Vocabulary A is the official NELL23K train-relation insertion order (198).
Each relation is one L2-normalized vector of size ``d``. Training listed
negatives then prefer cosine-similar relations instead of a uniform
``process.py`` permutation.

``k=9`` is the InfoNCE set size (official 10-way = 1 gold + 9 distractors).
It does not grow with ``|A|``. Similarity never materializes an ``n x n``
matrix: one gold is one matvec ``E @ e_gold`` (``O(n d)``), then keep a
``top-M`` pool and sample ``k`` from it. Unique golds are cached, so 3253
QA items with 198 golds cost 198 matvecs, not 3253. For ``n=10_000`` cache
``n x d`` embeddings plus ``|golds| x M`` neighbors, not ``n x n``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_EMBED_POOL_SIZE = 40
DEFAULT_EMBED_TEMPERATURE = 0.1
DEFAULT_NEIGHBOR_K = 10
DEFAULT_WHITEN_MODE = "pca_whiten"
DEFAULT_WHITEN_ALPHA = 0.5

TRANSFORM_MODES = ("raw", "center", "allbuttop", "pca_whiten")


def l2_normalize(embeddings: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape [n, dim], got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Dense ``n x n`` cosine table. Tests / tiny vocabs only; do not use at 10k."""
    normalized = l2_normalize(embeddings)
    return normalized @ normalized.T


def cosine_row(normalized: np.ndarray, gold_index: int) -> np.ndarray:
    """Cosine of one gold against the whole table. ``O(n d)``, no ``n x n``."""
    return normalized @ normalized[gold_index]


def mean_offdiag_cosine(normalized: np.ndarray) -> float:
    """Mean pairwise cosine without forming the Gram matrix. ``O(n d)``."""
    n = int(normalized.shape[0])
    if n < 2:
        return 0.0
    col_sum = normalized.sum(axis=0)
    total = float(col_sum @ col_sum)
    return (total - n) / (n * (n - 1))


def _centered_svd(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape [n, dim], got {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"need at least 2 rows to fit a transform, got {matrix.shape[0]}")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    return centered, singular, basis


def pca_partial_whiten(
    embeddings: np.ndarray,
    *,
    k: int | None = None,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Center, keep the top-``k`` principal directions, rescale them by ``sigma ** -alpha``.

    ``alpha=0`` is a plain PCA projection; ``alpha=1`` is full whitening of the
    retained subspace. ``k=n-1`` with ``alpha=1`` drives the Gram matrix to the
    identity (every relation equidistant, cosine carries no information), so for
    negative sampling keep ``alpha<1`` or ``k`` well below ``n-1``. Rows are left
    un-normalized; ``l2_normalize`` afterwards for cosine.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    centered, singular, basis = _centered_svd(embeddings)
    take = len(singular) if k is None else min(int(k), len(singular))
    if take <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    projected = centered @ basis[:take].T
    if alpha == 0:
        return projected
    sigma = np.maximum(singular[:take] / np.sqrt(max(centered.shape[0] - 1, 1)), eps)
    return projected / (sigma**alpha)


def all_but_the_top(embeddings: np.ndarray, *, remove: int = 1) -> np.ndarray:
    """Drop the top ``remove`` principal directions (Mu & Viswanath style)."""
    if remove < 0:
        raise ValueError(f"remove must be >= 0, got {remove}")
    centered, singular, basis = _centered_svd(embeddings)
    take = min(int(remove), len(singular) - 1)
    if take <= 0:
        return centered
    return centered - (centered @ basis[:take].T) @ basis[:take]


def apply_linear_transform(
    embeddings: np.ndarray,
    *,
    mode: str = "raw",
    k: int | None = None,
    alpha: float = DEFAULT_WHITEN_ALPHA,
    remove: int = 1,
) -> np.ndarray:
    """Apply one of the ``TRANSFORM_MODES`` to a relation embedding table."""
    if mode not in TRANSFORM_MODES:
        raise ValueError(f"unknown transform mode {mode!r}; expected one of {TRANSFORM_MODES}")
    if mode == "raw":
        return np.asarray(embeddings, dtype=np.float64)
    if mode == "center":
        return np.asarray(embeddings, dtype=np.float64) - np.asarray(
            embeddings, dtype=np.float64
        ).mean(axis=0, keepdims=True)
    if mode == "allbuttop":
        return all_but_the_top(embeddings, remove=remove)
    return pca_partial_whiten(embeddings, k=k, alpha=alpha)


def geometry_summary(
    embeddings: np.ndarray,
    *,
    pool_size: int = DEFAULT_EMBED_POOL_SIZE,
    temperature: float = DEFAULT_EMBED_TEMPERATURE,
) -> dict:
    """Scale of the sampling space: cone, hub concentration, effective negatives.

    Materializes the Gram matrix, so this is for diagnostics on the 198-relation
    train vocabulary, not for the training loop. ``effective_negatives`` is
    ``1 / sum(w ** 2)`` of the sampler softmax over the top-``pool_size``
    neighbours, averaged over golds. A value equal to ``pool_size`` means uniform
    sampling inside the pool; values near 1 mean the sampler always takes the same
    relation.
    """
    normalized = l2_normalize(embeddings)
    n = int(normalized.shape[0])
    gram = normalized @ normalized.T
    off = ~np.eye(n, dtype=bool)
    effective: list[float] = []
    top1 = np.zeros(n, dtype=int)
    for i in range(n):
        row = gram[i].copy()
        row[i] = -np.inf
        top = np.argsort(-row)[: max(1, min(pool_size, n - 1))]
        top1[int(top[0])] += 1
        weights = softmax(row[top], temperature)
        effective.append(1.0 / float(np.sum(weights**2)))
    hub_index = int(np.argmax(top1))
    return {
        "n_relations": n,
        "dim": int(normalized.shape[1]),
        "mean_offdiag_cosine": float(gram[off].mean()),
        "std_offdiag_cosine": float(gram[off].std()),
        "pool_size": int(pool_size),
        "temperature": float(temperature),
        "mean_effective_negatives": float(np.mean(effective)),
        "top1_hub_count": int(top1[hub_index]),
        "top1_hub_index": hub_index,
    }



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


def top_neighbor_pool(
    gold: str,
    relation_order: list[str],
    embeddings: np.ndarray,
    *,
    pool_size: int,
) -> list[tuple[str, float]]:
    """Return the ``pool_size`` nearest relations to ``gold``.

    One matvec against the embedding table, then ``argpartition``. Does not
    allocate an ``n x n`` similarity matrix.
    """
    if gold not in relation_order:
        raise ValueError(f"gold relation absent from train vocabulary: {gold}")
    if pool_size <= 0:
        return []
    gold_index = relation_order.index(gold)
    normalized = l2_normalize(embeddings)
    scores = cosine_row(normalized, gold_index)
    scores = scores.copy()
    scores[gold_index] = -np.inf
    n = len(relation_order)
    take = min(pool_size, n - 1)
    if take <= 0:
        return []
    if take >= n - 1:
        candidate_idx = [i for i in range(n) if i != gold_index]
    else:
        candidate_idx = np.argpartition(-scores, take - 1)[:take].tolist()
    ranked = sorted(
        ((float(scores[i]), relation_order[i]) for i in candidate_idx),
        key=lambda item: (-item[0], item[1]),
    )
    return [(rel, sim) for sim, rel in ranked[:take]]


def sample_from_pool(
    pool: list[tuple[str, float]],
    *,
    k: int,
    rng: np.random.RandomState,
    temperature: float = DEFAULT_EMBED_TEMPERATURE,
) -> list[str]:
    if k <= 0 or not pool:
        return []
    items = [rel for rel, _ in pool]
    weights = softmax(np.array([sim for _, sim in pool], dtype=np.float64), temperature)
    return weighted_sample_without_replacement(items, weights, k=k, rng=rng)


def sample_embed_negatives(
    gold: str,
    relation_order: list[str],
    embeddings: np.ndarray,
    *,
    k: int,
    rng: np.random.RandomState,
    pool_size: int = DEFAULT_EMBED_POOL_SIZE,
    temperature: float = DEFAULT_EMBED_TEMPERATURE,
) -> list[str]:
    """Sample ``k`` negatives, preferring cosine-similar train relations."""
    pool = top_neighbor_pool(
        gold, relation_order, embeddings, pool_size=pool_size
    )
    return sample_from_pool(pool, k=k, rng=rng, temperature=temperature)


def top_neighbors(
    gold: str,
    relation_order: list[str],
    embeddings: np.ndarray,
    *,
    k: int = DEFAULT_NEIGHBOR_K,
) -> list[dict]:
    pool = top_neighbor_pool(gold, relation_order, embeddings, pool_size=k)
    return [{"relation": rel, "cosine": sim} for rel, sim in pool]


class RelationNeighborIndex:
    """Cache top-M neighbors per gold. Storage is ``|golds| x M``, not ``n x n``."""

    def __init__(
        self,
        relation_order: list[str],
        embeddings: np.ndarray,
        *,
        pool_size: int = DEFAULT_EMBED_POOL_SIZE,
    ) -> None:
        self.relation_order = list(relation_order)
        self.normalized = l2_normalize(embeddings)
        if self.normalized.shape[0] != len(self.relation_order):
            raise ValueError(
                f"embeddings ({self.normalized.shape[0]}) != vocab ({len(self.relation_order)})"
            )
        self.pool_size = int(pool_size)
        self._pools: dict[str, list[tuple[str, float]]] = {}

    def pool(self, gold: str) -> list[tuple[str, float]]:
        cached = self._pools.get(gold)
        if cached is not None:
            return cached
        built = top_neighbor_pool(
            gold,
            self.relation_order,
            self.normalized,
            pool_size=self.pool_size,
        )
        self._pools[gold] = built
        return built

    def sample(
        self,
        gold: str,
        *,
        k: int,
        rng: np.random.RandomState,
        temperature: float = DEFAULT_EMBED_TEMPERATURE,
    ) -> list[str]:
        return sample_from_pool(self.pool(gold), k=k, rng=rng, temperature=temperature)

    def sampled_cosines(self, gold: str, negatives: list[str]) -> list[float]:
        lookup = {rel: sim for rel, sim in self.pool(gold)}
        return [float(lookup[rel]) for rel in negatives if rel in lookup]


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
