import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_negative_grip.paired_stats import (  # noqa: E402
    categorize,
    category_counts,
    mcnemar_exact,
    normalize_answer,
    ranks_from_scores,
    stratified_bootstrap,
    transition_matrix,
)


def test_mcnemar_matches_small_exact_case():
    # 9 vs 1 discordant: p = 2 * P(X<=1 | n=10, 1/2) = 2 * 11/1024.
    result = mcnemar_exact(9, 1)
    assert math.isclose(result["p_value"], 22 / 1024, rel_tol=1e-9)
    assert mcnemar_exact(5, 5)["p_value"] == 1.0
    assert mcnemar_exact(0, 0)["p_value"] == 1.0


def test_mcnemar_log_p_survives_underflow():
    result = mcnemar_exact(3000, 1000)
    assert result["p_value"] == 0.0 or result["p_value"] < 1e-200
    assert result["log10_p_value"] < -200


def test_bootstrap_ci_contains_point_estimate_and_is_paired():
    rng = np.random.default_rng(0)
    left = rng.integers(0, 2, 2000).astype(float)
    right = left.copy()
    right[:200] = 1.0
    strata = np.array(["a", "b"] * 1000)
    out = stratified_bootstrap(left, right, strata, draws=2000, seed=1)
    delta = (right - left).mean()
    lo, hi = out["delta_ci95"]
    assert lo <= delta <= hi
    assert lo >= 0.0  # right never worse than left on any question


def test_identical_arms_give_zero_width_delta():
    left = np.array([1.0, 0.0, 1.0, 1.0])
    out = stratified_bootstrap(left, left.copy(), np.array(["x"] * 4), draws=200, seed=3)
    assert out["delta_ci95"] == [0.0, 0.0]


def test_rank_ties_are_conservative():
    assert ranks_from_scores({"g": -1.0, "a": -1.0, "b": -2.0}, "g") == 2
    assert ranks_from_scores({"g": -0.5, "a": -1.0}, "g") == 1


def test_categorize_uses_em_normalization():
    vocab = {normalize_answer(x) for x in ["Award Winners", "Cause of Death", "Child Organizations"]}
    base = {"candidates": ["Award Winners", "Cause of Death"]}
    assert categorize({**base, "correct": True, "response": "x"}, vocab) == "correct"
    assert categorize({**base, "correct": False, "response": "cause of death,"}, vocab) == "wrong_in_list"
    assert categorize({**base, "correct": False, "response": "Child Organizations"}, vocab) == "wrong_off_list_in_vocab"
    assert categorize({**base, "correct": False, "response": "Made Up"}, vocab) == "wrong_oov"
    assert categorize({**base, "correct": False, "response": "I Don't Know"}, vocab) == "wrong_abstain"
    assert categorize(
        {**base, "correct": False, "response": "No direct relation based on provided context"}, vocab
    ) == "wrong_abstain"


def test_counts_and_transitions():
    left = ["correct", "wrong_in_list", "wrong_oov"]
    right = ["correct", "correct", "wrong_oov"]
    assert category_counts(left)["wrong_in_list"] == 1
    matrix = transition_matrix(left, right)
    assert matrix["wrong_in_list"]["correct"] == 1
    assert matrix["wrong_oov"]["wrong_oov"] == 1
