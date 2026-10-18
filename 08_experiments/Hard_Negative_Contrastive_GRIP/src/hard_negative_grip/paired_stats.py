"""Paired significance tests and error decomposition for two fixed adapters.

Everything here is CPU-only and depends on numpy alone. Uncertainty covers
question sampling for one trained pair; it says nothing about training seeds.
"""
from __future__ import annotations

import math
import re
import string
from collections.abc import Iterable, Sequence

import numpy as np

ERROR_CATEGORIES = (
    "correct", "wrong_in_list", "wrong_off_list_in_vocab", "wrong_abstain", "wrong_oov",
)
# Refusal-style outputs ("I don't know", "no direct relation ..."), after normalization.
_ABSTAIN = re.compile(
    r"^(i dont know|i do not know|unknown|none|not related|no relation|no direct relation\b.*)$"
)


def normalize_answer(answer: str) -> str:
    """Same normalization as grip-exp ``evaluation.utils.normalize_answer``."""
    text = str(answer).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    mapping = {"man": "men", "woman": "women"}
    return " ".join(mapping.get(word, word) for word in text.split())


def _log_binom_pmf(k: int, n: int) -> float:
    """log P(X = k) for X ~ Binomial(n, 1/2)."""
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)
    )


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def mcnemar_exact(gains: int, regressions: int) -> dict:
    """Two-sided exact McNemar test on the discordant pairs.

    ``gains``: right correct, left wrong. ``regressions``: the reverse.
    Returns the p-value plus log10(p), since p underflows on large samples.
    """
    n = gains + regressions
    if n == 0:
        return {"discordant": 0, "p_value": 1.0, "log10_p_value": 0.0, "chi2_cc": 0.0}
    k = min(gains, regressions)
    log_tail = _logsumexp([_log_binom_pmf(i, n) for i in range(k + 1)])
    log_p = min(0.0, math.log(2.0) + log_tail)
    chi2 = (abs(gains - regressions) - 1) ** 2 / n
    return {
        "discordant": n,
        "p_value": math.exp(log_p),
        "log10_p_value": log_p / math.log(10.0),
        "chi2_cc": chi2,
    }


def ranks_from_scores(scores: dict, gold: str) -> int:
    """Conservative rank: ties with the gold count against it (matches metrics.py)."""
    positive = scores[gold]
    return 1 + sum(1 for rel, value in scores.items() if rel != gold and value >= positive)


def stratified_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    strata: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 20261018,
    chunk: int = 500,
) -> dict:
    """Paired question bootstrap of mean(right) - mean(left), resampled within strata.

    Returns 95% percentile intervals for left, right, and the difference.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    strata = np.asarray(strata)
    if left.shape != right.shape or left.shape != strata.shape:
        raise ValueError("left, right, strata must share one shape")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    n = len(left)
    stats = {"left": [], "right": [], "delta": []}
    remaining = draws
    while remaining > 0:
        size = min(chunk, remaining)
        sum_left = np.zeros(size)
        sum_right = np.zeros(size)
        for index in groups:
            picks = index[rng.integers(0, len(index), size=(size, len(index)))]
            sum_left += left[picks].sum(axis=1)
            sum_right += right[picks].sum(axis=1)
        stats["left"].append(sum_left / n)
        stats["right"].append(sum_right / n)
        stats["delta"].append((sum_right - sum_left) / n)
        remaining -= size
    result = {}
    for key, chunks in stats.items():
        values = np.concatenate(chunks)
        low, high = np.quantile(values, [0.025, 0.975])
        result[f"{key}_ci95"] = [float(low), float(high)]
    result["draws"] = draws
    result["seed"] = seed
    return result


def categorize(row: dict, vocab_normalized: set[str]) -> str:
    """Assign one prediction to an error category.

    List and vocabulary membership use the same normalization as EM, so a
    prediction counted correct is never also counted out of list.
    """
    if row["correct"]:
        return "correct"
    predicted = normalize_answer(row.get("response", ""))
    listed = {normalize_answer(rel) for rel in row.get("candidates", [])}
    if predicted in listed:
        return "wrong_in_list"
    if predicted in vocab_normalized:
        return "wrong_off_list_in_vocab"
    if _ABSTAIN.match(predicted):
        return "wrong_abstain"
    return "wrong_oov"


def category_counts(categories: Iterable[str]) -> dict[str, int]:
    counts = {name: 0 for name in ERROR_CATEGORIES}
    for name in categories:
        counts[name] += 1
    return counts


def transition_matrix(left: Sequence[str], right: Sequence[str]) -> dict[str, dict[str, int]]:
    """Counts of (left category -> right category) on the same questions."""
    matrix = {a: {b: 0 for b in ERROR_CATEGORIES} for a in ERROR_CATEGORIES}
    for a, b in zip(left, right, strict=True):
        matrix[a][b] += 1
    return matrix
