from __future__ import annotations

from collections.abc import Iterable


def reciprocal_rank(positive_score: float, negative_scores: Iterable[float]) -> float:
    """Compute reciprocal rank with ties assigned the conservative worst rank."""
    rank = 1 + sum(score >= positive_score for score in negative_scores)
    return 1.0 / rank


def hits_at_k(positive_score: float, negative_scores: Iterable[float], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    rank = 1 + sum(score >= positive_score for score in negative_scores)
    return float(rank <= k)


def summarize_candidate_scores(
    rows: Iterable[dict],
    *,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float | int]:
    """Aggregate candidate MRR and Hits@K from score rows.

    Each row contains a scalar ``positive_score`` and an iterable
    ``negative_scores``. Rows may optionally contain ``kind``; per-kind metrics
    are emitted to make family ablations auditable.
    """
    materialized = list(rows)
    if not materialized:
        return {"count": 0, "mrr": 0.0, **{f"hits@{k}": 0.0 for k in ks}}

    groups: dict[str, list[dict]] = {"all": materialized}
    for row in materialized:
        kind = str(row.get("kind", "unknown"))
        groups.setdefault(kind, []).append(row)

    summary: dict[str, float | int] = {"count": len(materialized)}
    for name, group in groups.items():
        prefix = "" if name == "all" else f"{name}."
        summary[f"{prefix}mrr"] = sum(
            reciprocal_rank(row["positive_score"], row["negative_scores"])
            for row in group
        ) / len(group)
        for k in ks:
            summary[f"{prefix}hits@{k}"] = sum(
                hits_at_k(row["positive_score"], row["negative_scores"], k)
                for row in group
            ) / len(group)
        if name != "all":
            summary[f"{prefix}count"] = len(group)
    return summary
