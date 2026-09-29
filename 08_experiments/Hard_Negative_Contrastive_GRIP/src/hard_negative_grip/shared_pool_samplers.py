"""Frozen shared-pool listed-contrastive samplers.

Random-K / Top-K Hard / Coverage-Adaptive K plus the Prompt-6 Soft-Mix and
Calibrated mixture. All variants read the same train-only valid-negative pool
from a Prompt-4/7 score dump. They never rescore and never consult valid/test
KG structure.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from .confusion_analysis import (
    analyze_qa_group,
    iter_qa_groups_from_scores,
    k_for_coverage,
    percentile,
    summarize_values,
    valid_negatives,
    write_json,
    write_jsonl,
)
from .official_lists import load_train_relation_order
from .score_hard import merge_negative_sources
from .task_file import (
    assistant_gold,
    build_qa_assets_from_task_texts,
    is_grip_task_file,
    load_json_payload,
    load_score_hard_manifest,
    original_question_ids_from_payload,
    task_qa_id,
)

CONTROL_SAMPLER_VARIANTS = ("random_k", "top_k_hard", "coverage_adaptive_k")
MIXTURE_SAMPLER_VARIANTS = ("soft_mix", "calibrated")
SAMPLER_VARIANTS = CONTROL_SAMPLER_VARIANTS + MIXTURE_SAMPLER_VARIANTS
DEFAULT_K_FIXED = 9
DEFAULT_K_UNIFORM = 6
DEFAULT_K_SOFT = 3
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 20
DEFAULT_SEED = 2026
DEFAULT_TAU_SOURCE = "median_top9_negative_mass"
DEFAULT_SOFT_RHO = 0.33
DEFAULT_LAMBDA_0 = 1.0
DEFAULT_LAMBDA_MIN = 0.25
DEFAULT_LAMBDA_BETA = 0.5
DEFAULT_LISTED_BATCH = 1
DEFAULT_LISTED_ACCUM = 512
DEFAULT_LISTED_EPOCHS = 10
DEFAULT_MIN_TRAIN_QA = 3253
RETIRED_UNDERFIT_TRAIN = "64-QA / 10-step listed smoke"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_valid_pool(rows: list[dict]) -> list[dict]:
    """Valid negatives in dump order. That order is the shared population."""
    pool = valid_negatives(rows)
    seen: set[str] = set()
    ordered: list[dict] = []
    for row in pool:
        relation = str(row["candidate_relation"])
        if not relation:
            raise ValueError(f"{row.get('qa_id')}: empty valid-negative relation")
        if relation in seen:
            raise ValueError(f"{row.get('qa_id')}: duplicate valid-negative {relation!r}")
        seen.add(relation)
        ordered.append(row)
    return ordered


def rank_valid_pool(pool: list[dict]) -> list[dict]:
    """Hardest first: higher candidate_score, then relation name."""
    return sorted(
        pool,
        key=lambda row: (-float(row["candidate_score"]), str(row["candidate_relation"])),
    )


def pool_relations(pool: Iterable[dict]) -> list[str]:
    return [str(row["candidate_relation"]) for row in pool]


def pool_masses_desc(ranked: list[dict]) -> list[float]:
    return [float(row["negative_mass"]) for row in ranked]


def covered_mass(ranked: list[dict], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    return float(sum(float(row["negative_mass"]) for row in ranked[:k]))


def decide_coverage_k(
    masses_desc: list[float],
    *,
    tau: float,
    k_min: int,
    k_max: int,
) -> int:
    """Smallest K covering ``tau`` of negative mass, then clamp to ``[k_min, k_max]``."""
    if k_min < 0:
        raise ValueError(f"k_min must be non-negative, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, got {k_max} < {k_min}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if not masses_desc:
        return 0
    raw = k_for_coverage(masses_desc, tau)
    return max(k_min, min(k_max, raw, len(masses_desc)))


def sample_random_k(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    take = min(k, len(pool))
    if take == 0:
        return []
    if take == len(pool):
        return list(pool)
    indexes = rng.sample(range(len(pool)), take)
    return [pool[index] for index in indexes]


def sample_top_k(ranked: list[dict], k: int) -> list[dict]:
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return list(ranked[: min(k, len(ranked))])


def confusion_masses(pool: list[dict]) -> list[float]:
    """Normalize ``negative_mass`` over the valid pool so the values sum to 1."""
    masses = [max(float(row["negative_mass"]), 0.0) for row in pool]
    total = sum(masses)
    if total <= 0:
        if not masses:
            return []
        return [1.0 / len(masses)] * len(masses)
    return [mass / total for mass in masses]


def mixture_probabilities(masses: list[float], *, rho: float) -> list[float]:
    """``p(r) = (1-rho)/|P| + rho m(r)``. ``rho=0`` is uniform; ``rho=1`` is mass."""
    if rho < 0 or rho > 1:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    n = len(masses)
    if n == 0:
        return []
    uniform = 1.0 / n
    return [(1.0 - rho) * uniform + rho * mass for mass in masses]


def sample_weighted_without_replacement(
    pool: list[dict],
    k: int,
    rng: random.Random,
    probabilities: list[float],
    *,
    excluded: set[str] | None = None,
) -> list[dict]:
    """Draw ``k`` rows without replacement from a discrete mixture over the pool."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if len(probabilities) != len(pool):
        raise ValueError(
            f"probabilities has {len(probabilities)} entries, expected {len(pool)}"
        )
    blocked = excluded or set()
    remaining = [
        (row, float(prob))
        for row, prob in zip(pool, probabilities)
        if str(row["candidate_relation"]) not in blocked and float(prob) > 0
    ]
    take = min(k, len(remaining))
    chosen: list[dict] = []
    for _ in range(take):
        total = sum(prob for _row, prob in remaining)
        if total <= 0:
            break
        draw = rng.random() * total
        cursor = 0.0
        index = len(remaining) - 1
        for candidate_index, (_row, prob) in enumerate(remaining):
            cursor += prob
            if draw <= cursor:
                index = candidate_index
                break
        chosen.append(remaining.pop(index)[0])
    return chosen


def sample_soft_mix(
    pool: list[dict],
    *,
    k_uniform: int,
    k_soft: int,
    rho: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """6 uniform + 3 mixture draws, without replacement, gold already excluded."""
    if k_uniform < 0 or k_soft < 0:
        raise ValueError(f"k_uniform/k_soft must be non-negative, got {k_uniform}/{k_soft}")
    n_valid = len(pool)
    take_uniform = min(k_uniform, n_valid)
    uniform_rows = sample_random_k(pool, take_uniform, rng)
    remaining = n_valid - len(uniform_rows)
    take_soft = min(k_soft, remaining)
    if take_soft == 0:
        return uniform_rows, []
    masses = confusion_masses(pool)
    probabilities = mixture_probabilities(masses, rho=rho)
    excluded = {str(row["candidate_relation"]) for row in uniform_rows}
    soft_rows = sample_weighted_without_replacement(
        pool,
        take_soft,
        rng,
        probabilities,
        excluded=excluded,
    )
    return uniform_rows, soft_rows


def clip_calibrated_lambda(
    concentration: float,
    *,
    lambda_0: float = DEFAULT_LAMBDA_0,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    beta: float = DEFAULT_LAMBDA_BETA,
) -> float:
    """``lambda_q = lambda_0 * clip(1 - beta * C_q, lambda_min/lambda_0, 1)``."""
    if lambda_0 <= 0:
        raise ValueError(f"lambda_0 must be positive, got {lambda_0}")
    if lambda_min < 0 or lambda_min > lambda_0:
        raise ValueError(f"lambda_min must be in [0, lambda_0], got {lambda_min}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    raw = 1.0 - beta * float(concentration)
    lower = lambda_min / lambda_0
    clipped = min(1.0, max(lower, raw))
    return float(lambda_0 * clipped)


def listed_update_budget(
    n_samples: int,
    *,
    batch: int = DEFAULT_LISTED_BATCH,
    requested_accum: int = DEFAULT_LISTED_ACCUM,
    epochs: float = DEFAULT_LISTED_EPOCHS,
) -> dict:
    """Match ``training_arguments``: clamp accum when the slice is smaller than it."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if requested_accum < 1:
        raise ValueError(f"requested_accum must be >= 1, got {requested_accum}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    effective_accum = requested_accum
    if batch * requested_accum > n_samples:
        effective_accum = max(1, n_samples // batch)
    steps_per_epoch = max(n_samples // (batch * effective_accum), 1) if n_samples else 0
    total_steps = int(epochs * steps_per_epoch)
    clamped = effective_accum != requested_accum
    too_small = n_samples <= DEFAULT_MIN_TRAIN_QA
    return {
        "n_samples": n_samples,
        "batch": batch,
        "requested_accum": requested_accum,
        "effective_accum": effective_accum,
        "epochs": float(epochs),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "accum_clamped": clamped,
        "underfit": bool(clamped or too_small),
    }


def refuse_underfit_listed_budget(
    n_samples: int,
    *,
    batch: int = DEFAULT_LISTED_BATCH,
    requested_accum: int = DEFAULT_LISTED_ACCUM,
    epochs: float = DEFAULT_LISTED_EPOCHS,
) -> dict:
    """Block the retired 64-QA protocol before a listed GPU job starts."""
    budget = listed_update_budget(
        n_samples,
        batch=batch,
        requested_accum=requested_accum,
        epochs=epochs,
    )
    if budget["underfit"]:
        if budget["accum_clamped"]:
            raise ValueError(
                f"{n_samples} QA clamps accum {requested_accum} -> {budget['effective_accum']} "
                f"and yields {budget['total_steps']} update steps. That is {RETIRED_UNDERFIT_TRAIN}, "
                "not listed training. Wire the frozen manifests onto the full paper task file, "
                f"then train Random-K with accum={requested_accum}."
            )
        raise ValueError(
            f"{n_samples} QA is a matchable-only or smoke slice "
            f"({budget['total_steps']} update steps). Train Random-K on the full paper "
            f"task file with accum={requested_accum}."
        )
    return budget


def assert_score_hard_listed_training_budget(
    n_samples: int,
    *,
    listed_negative_source: str,
    skip_train: bool = False,
    allow_underfit: bool = False,
    batch: int = DEFAULT_LISTED_BATCH,
    requested_accum: int = DEFAULT_LISTED_ACCUM,
    epochs: float = DEFAULT_LISTED_EPOCHS,
) -> dict | None:
    """Refuse shared-pool listed training that is not the paper-task budget.

    Other negative sources keep their smoke/pilot launchers. ``skip_train`` and
    ``allow_underfit`` are only for evaluating an already-finished adapter.
    """
    if skip_train or allow_underfit or listed_negative_source != "score_hard":
        return None
    return refuse_underfit_listed_budget(
        n_samples,
        batch=batch,
        requested_accum=requested_accum,
        epochs=epochs,
    )


def propose_coverage_tau(topn_masses: list[float]) -> float:
    """Freeze tau as the median Top-K_fixed mass so the median QA still uses K=9."""
    if not topn_masses:
        raise ValueError("cannot freeze coverage tau from an empty Top-N mass list")
    tau = percentile(topn_masses, 50)
    if tau is None:
        raise ValueError("median Top-N mass is undefined")
    return float(tau)


def qa_identity(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty QA group")
    gold_rows = [row for row in rows if row.get("is_gold")]
    if len(gold_rows) != 1:
        raise ValueError(f"{rows[0].get('qa_id')}: expected exactly one gold row")
    gold = gold_rows[0]
    return {
        "question_id": str(rows[0]["qa_id"]),
        "positive_relation": str(gold["gold_relation"]),
        "matched_train_relation": str(gold["matched_train_relation"]),
        "gold_score": float(gold["candidate_score"]),
        "n_candidates": len(rows),
    }


def _manifest_row(
    *,
    identity: dict,
    variant: str,
    hard: list[str],
    uniform: list[str],
    extra: dict,
) -> dict:
    negatives = merge_negative_sources(hard, uniform)
    row = {
        "question_id": identity["question_id"],
        "positive_relation": identity["positive_relation"],
        "matched_train_relation": identity["matched_train_relation"],
        "negative_source": f"shared_valid_pool_{variant}",
        "shared_pool": "train_only_valid_negatives",
        "hard_negative_relations": hard,
        "uniform_negative_relations": uniform,
        "negative_relations": negatives,
        "k": len(negatives),
        "lambda_candidate": extra.get("lambda_candidate", DEFAULT_LAMBDA_0),
    }
    row.update(extra)
    if negatives != merge_negative_sources(hard, uniform):
        raise ValueError(f"{identity['question_id']}: negative provenance mismatch")
    if identity["positive_relation"] in negatives:
        raise ValueError(f"{identity['question_id']}: gold leaked into negatives")
    if len(negatives) != len(set(negatives)):
        raise ValueError(f"{identity['question_id']}: duplicate negatives")
    return row


def sample_qa_variants(
    rows: list[dict],
    *,
    k_fixed: int,
    tau: float,
    k_min: int,
    k_max: int,
    rng: random.Random,
    seed: int,
    k_uniform: int = DEFAULT_K_UNIFORM,
    k_soft: int = DEFAULT_K_SOFT,
    rho: float = DEFAULT_SOFT_RHO,
    lambda_0: float = DEFAULT_LAMBDA_0,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
) -> dict:
    identity = qa_identity(rows)
    pool = shared_valid_pool(rows)
    ranked = rank_valid_pool(pool)
    masses = pool_masses_desc(ranked)
    n_valid = len(pool)
    k_random = min(k_fixed, n_valid)
    k_top = min(k_fixed, n_valid)
    k_adaptive = decide_coverage_k(masses, tau=tau, k_min=k_min, k_max=k_max)
    analysis = analyze_qa_group(rows)
    concentration = float(analysis["top9_mass"])
    lambda_q = clip_calibrated_lambda(
        concentration,
        lambda_0=lambda_0,
        lambda_min=lambda_min,
        beta=lambda_beta,
    )

    random_rows = sample_random_k(pool, k_random, rng)
    top_rows = sample_top_k(ranked, k_top)
    adaptive_rows = sample_top_k(ranked, k_adaptive)
    mix_uniform, mix_soft = sample_soft_mix(
        pool,
        k_uniform=k_uniform,
        k_soft=k_soft,
        rho=rho,
        rng=rng,
    )
    mix_covered = sum(float(row["negative_mass"]) for row in [*mix_uniform, *mix_soft])
    per_qa = {
        "qa_id": identity["question_id"],
        "positive_relation": identity["positive_relation"],
        "matched_train_relation": identity["matched_train_relation"],
        "n_valid_negatives": n_valid,
        "k_random": k_random,
        "k_top": k_top,
        "k_adaptive": k_adaptive,
        "k_uniform": len(mix_uniform),
        "k_soft": len(mix_soft),
        "k_mix": len(mix_uniform) + len(mix_soft),
        "adaptive_covered_mass": covered_mass(ranked, k_adaptive),
        "random_covered_mass": sum(float(row["negative_mass"]) for row in random_rows),
        "top_covered_mass": covered_mass(ranked, k_top),
        "soft_mix_covered_mass": mix_covered,
        "top1_mass": analysis["top1_mass"],
        "top9_mass": analysis["top9_mass"],
        "top20_mass": analysis["top20_mass"],
        "concentration_c_q": concentration,
        "lambda_q": lambda_q,
        "K_80": analysis["K_80"],
        "K_90": analysis["K_90"],
        "hardest_negative": analysis["hardest_negative"],
        "hardest_negative_gap": analysis["hardest_negative_gap"],
    }
    shared_extra = {
        "n_valid_negatives": n_valid,
        "k_fixed": k_fixed,
        "filter_splits": ["train"],
        "source_dump_field": "is_valid_negative",
        "lambda_candidate": lambda_0,
    }
    mix_extra = {
        **shared_extra,
        "selection_rule": "k_uniform_uniform_plus_k_soft_mass_mixture",
        "selection_seed": seed,
        "k_uniform": k_uniform,
        "k_soft": k_soft,
        "soft_rho": rho,
        "covered_negative_mass": mix_covered,
        "concentration_c_q": concentration,
    }
    return {
        "per_qa": per_qa,
        "random_k": _manifest_row(
            identity=identity,
            variant="random_k",
            hard=[],
            uniform=pool_relations(random_rows),
            extra={
                **shared_extra,
                "selection_rule": "uniform_without_replacement",
                "selection_seed": seed,
                "covered_negative_mass": per_qa["random_covered_mass"],
            },
        ),
        "top_k_hard": _manifest_row(
            identity=identity,
            variant="top_k_hard",
            hard=pool_relations(top_rows),
            uniform=[],
            extra={
                **shared_extra,
                "selection_rule": "top_candidate_score_then_relation_name",
                "selection_seed": None,
                "covered_negative_mass": per_qa["top_covered_mass"],
            },
        ),
        "coverage_adaptive_k": _manifest_row(
            identity=identity,
            variant="coverage_adaptive_k",
            hard=pool_relations(adaptive_rows),
            uniform=[],
            extra={
                **shared_extra,
                "selection_rule": "smallest_top_k_covering_tau_clamped",
                "selection_seed": None,
                "coverage_tau": tau,
                "k_min": k_min,
                "k_max": k_max,
                "raw_coverage_k": k_for_coverage(masses, tau) if masses else 0,
                "covered_negative_mass": per_qa["adaptive_covered_mass"],
            },
        ),
        "soft_mix": _manifest_row(
            identity=identity,
            variant="soft_mix",
            hard=pool_relations(mix_soft),
            uniform=pool_relations(mix_uniform),
            extra={
                **mix_extra,
                "lambda_candidate": lambda_0,
            },
        ),
        "calibrated": _manifest_row(
            identity=identity,
            variant="calibrated",
            hard=pool_relations(mix_soft),
            uniform=pool_relations(mix_uniform),
            extra={
                **mix_extra,
                "lambda_candidate": lambda_q,
                "lambda_0": lambda_0,
                "lambda_min": lambda_min,
                "lambda_beta": lambda_beta,
            },
        ),
    }


def load_source_metadata(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def freeze_shared_pool_samplers(
    *,
    scores_path: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    k_fixed: int = DEFAULT_K_FIXED,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    seed: int = DEFAULT_SEED,
    tau: float | None = None,
    k_uniform: int = DEFAULT_K_UNIFORM,
    k_soft: int = DEFAULT_K_SOFT,
    rho: float = DEFAULT_SOFT_RHO,
    lambda_0: float = DEFAULT_LAMBDA_0,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
    variants: tuple[str, ...] = CONTROL_SAMPLER_VARIANTS,
) -> dict:
    """Read the train-only dump, freeze tau, and write immutable manifests."""
    if k_fixed < 0:
        raise ValueError(f"k_fixed must be non-negative, got {k_fixed}")
    unknown = [name for name in variants if name not in SAMPLER_VARIANTS]
    if unknown:
        raise ValueError(f"unknown sampler variants: {unknown}")
    if not variants:
        raise ValueError("variants must be non-empty")
    source_meta = load_source_metadata(metadata_path)
    filter_splits = source_meta.get("filter_splits") or ["train"]
    if list(filter_splits) != ["train"]:
        raise ValueError(
            "shared-pool samplers require a train-only dump; "
            f"got filter_splits={filter_splits}"
        )

    groups = list(iter_qa_groups_from_scores(scores_path))
    if not groups:
        raise ValueError(f"{scores_path} contains no QA groups")

    per_qa_analysis = [analyze_qa_group(group) for group in groups]
    top9_masses = [float(row["top9_mass"]) for row in per_qa_analysis]
    frozen_tau = float(tau) if tau is not None else propose_coverage_tau(top9_masses)
    rng = random.Random(seed)

    sampled = [
        sample_qa_variants(
            group,
            k_fixed=k_fixed,
            tau=frozen_tau,
            k_min=k_min,
            k_max=k_max,
            rng=rng,
            seed=seed,
            k_uniform=k_uniform,
            k_soft=k_soft,
            rho=rho,
            lambda_0=lambda_0,
            lambda_min=lambda_min,
            lambda_beta=lambda_beta,
        )
        for group in groups
    ]
    per_qa = [item["per_qa"] for item in sampled]
    manifests = {variant: [item[variant] for item in sampled] for variant in variants}

    k_adaptive_values = [int(row["k_adaptive"]) for row in per_qa]
    topn_names = [
        "top1_mass",
        "top9_mass",
        "top20_mass",
        "random_covered_mass",
        "top_covered_mass",
        "adaptive_covered_mass",
        "soft_mix_covered_mass",
        "concentration_c_q",
        "lambda_q",
    ]
    topn_stats = {
        name: summarize_values([float(row[name]) for row in per_qa]) for name in topn_names
    }
    k_stats = {
        "k_fixed": k_fixed,
        "k_uniform": k_uniform,
        "k_soft": k_soft,
        "k_adaptive": summarize_values([float(value) for value in k_adaptive_values]),
        "k_adaptive_eq_k_fixed": sum(1 for value in k_adaptive_values if value == k_fixed),
        "k_adaptive_lt_k_fixed": sum(1 for value in k_adaptive_values if value < k_fixed),
        "k_adaptive_gt_k_fixed": sum(1 for value in k_adaptive_values if value > k_fixed),
        "k_mix": summarize_values([float(row["k_mix"]) for row in per_qa]),
        "k_80": summarize_values([float(row["K_80"]) for row in per_qa]),
        "k_90": summarize_values([float(row["K_90"]) for row in per_qa]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {variant: output_dir / f"{variant}.jsonl" for variant in variants}
    per_qa_path = output_dir / "per_qa.jsonl"
    policy_path = output_dir / "policy.json"
    stats_path = output_dir / "global_statistics.json"
    report_path = output_dir / "FROZEN_SAMPLERS.md"

    write_jsonl(per_qa_path, per_qa)
    for variant, path in paths.items():
        write_jsonl(path, manifests[variant])

    policy = {
        "source_scores": str(scores_path),
        "source_metadata": None if metadata_path is None else str(metadata_path),
        "source_scores_sha256": file_sha256(scores_path),
        "filter_splits": ["train"],
        "shared_pool": "is_valid_negative from train-only dump",
        "n_qa": len(groups),
        "k_fixed": k_fixed,
        "k_uniform": k_uniform,
        "k_soft": k_soft,
        "k_min": k_min,
        "k_max": k_max,
        "coverage_tau": frozen_tau,
        "coverage_tau_source": DEFAULT_TAU_SOURCE if tau is None else "cli_override",
        "soft_rho": rho,
        "lambda_0": lambda_0,
        "lambda_min": lambda_min,
        "lambda_beta": lambda_beta,
        "seed": seed,
        "variants": list(variants),
        "selection_rules": {
            "random_k": "uniform_without_replacement from shared valid pool, K=k_fixed",
            "top_k_hard": "top k_fixed by candidate_score, relation name breaks ties",
            "coverage_adaptive_k": (
                "K_i = clip(smallest K covering coverage_tau of negative_mass, "
                "k_min, k_max); then take top K_i"
            ),
            "soft_mix": (
                f"{k_uniform} uniform + {k_soft} draws from "
                f"(1-rho)/|P| + rho * m(r), rho={rho}, without replacement"
            ),
            "calibrated": (
                "same negatives as soft_mix; lambda_q = lambda_0 * "
                "clip(1 - beta * top9_mass, lambda_min/lambda_0, 1)"
            ),
        },
        "do_not_train_from": "results/runs/20260923_offline_confusion_full",
        "training_protocol": {
            "wire_first": True,
            "next_variant": "soft_mix" if "soft_mix" in variants else "random_k",
            "train_qa": "full paper task file; listed negatives from frozen sampler IDs",
            "min_train_qa": DEFAULT_MIN_TRAIN_QA,
            "expected_paper_qa": 12014,
            "accum": DEFAULT_LISTED_ACCUM,
            "epochs": DEFAULT_LISTED_EPOCHS,
            "expected_listed_steps": 230,
            "do_not_train": RETIRED_UNDERFIT_TRAIN,
            "do_not_train_matchable_only": 3253,
            "do_not_rescore": "3253x198",
            "do_not_expand_underfit_overnight": True,
        },
    }
    write_json(policy_path, policy)

    global_payload = {
        "n_qa": len(groups),
        "n_valid_negatives": sum(int(row["n_valid_negatives"]) for row in per_qa),
        "coverage_tau": frozen_tau,
        "topn_negative_mass": topn_stats,
        "k_statistics": k_stats,
        "hardest_negative_gap": summarize_values(
            [
                float(row["hardest_negative_gap"])
                for row in per_qa
                if row["hardest_negative_gap"] is not None
            ]
        ),
    }
    write_json(stats_path, global_payload)

    manifest_hashes = {variant: file_sha256(path) for variant, path in paths.items()}
    report = render_sampler_report(
        policy=policy,
        stats=global_payload,
        manifest_paths=paths,
        manifest_hashes=manifest_hashes,
        per_qa_path=per_qa_path,
        policy_path=policy_path,
        stats_path=stats_path,
    )
    report_path.write_text(report, encoding="utf-8")

    summary = {
        **policy,
        "output_dir": str(output_dir),
        "policy_path": str(policy_path),
        "per_qa_path": str(per_qa_path),
        "global_statistics_path": str(stats_path),
        "report_path": str(report_path),
        "manifests": {variant: str(path) for variant, path in paths.items()},
        "manifest_sha256": manifest_hashes,
        "k_statistics": k_stats,
        "top9_mass": topn_stats["top9_mass"],
        "lambda_q": topn_stats["lambda_q"],
    }
    return summary


def render_sampler_report(
    *,
    policy: dict,
    stats: dict,
    manifest_paths: dict[str, Path],
    manifest_hashes: dict[str, str],
    per_qa_path: Path,
    policy_path: Path,
    stats_path: Path,
) -> str:
    k_stats = stats["k_statistics"]
    top9 = stats["topn_negative_mass"]["top9_mass"]
    adaptive = k_stats["k_adaptive"]
    next_variant = policy["training_protocol"]["next_variant"]
    mixture = "soft_mix" in policy["variants"] or "calibrated" in policy["variants"]
    lines = [
        "# Frozen shared-pool samplers",
        "",
        "These manifests are the only negative lists for listed-contrastive",
        "controls. They all read the train-only valid-negative pool. Do not",
        "resample live and do not use the all-split dump under",
        "`20260923_offline_confusion_full`.",
        "",
        "## Source",
        "",
        f"- Scores: `{policy['source_scores']}`",
        f"- SHA256: `{policy['source_scores_sha256']}`",
        f"- Filter splits: `{policy['filter_splits']}`",
        f"- QA count: {policy['n_qa']}",
        "",
        "## Frozen policy",
        "",
        f"- Shared pool: `{policy['shared_pool']}`",
        f"- Random-K / Top-K Hard `k_fixed`: {policy['k_fixed']}",
        f"- Coverage-Adaptive `tau`: {policy['coverage_tau']:.6f} ({policy['coverage_tau_source']})",
        f"- Coverage-Adaptive clamp: `[{policy['k_min']}, {policy['k_max']}]`",
        f"- Random seed: {policy['seed']}",
    ]
    if mixture:
        lines.extend(
            [
                f"- Soft-Mix: `{policy.get('k_uniform', DEFAULT_K_UNIFORM)}` uniform + "
                f"`{policy.get('k_soft', DEFAULT_K_SOFT)}` mixture, rho="
                f"{policy.get('soft_rho', DEFAULT_SOFT_RHO)}",
                f"- Calibrated: lambda_0={policy.get('lambda_0', DEFAULT_LAMBDA_0)}, "
                f"lambda_min={policy.get('lambda_min', DEFAULT_LAMBDA_MIN)}, "
                f"beta={policy.get('lambda_beta', DEFAULT_LAMBDA_BETA)}",
            ]
        )
    lines.extend(
        [
            "",
            "Coverage-Adaptive K is **not** K80/K90 over the full 197-way mass. Those",
            "diagnostics stay in the analysis tables (median K80 is far above 9). The",
            "training tau is the median Top-9 `negative_mass`, so the median QA still",
            "uses K=9, concentrated QAs shrink K, and diffuse QAs grow K up to `k_max`.",
            "",
            "## Train-only mass / Top-N",
            "",
            f"- Top-9 mass mean={top9['mean']:.4f} median={top9['median']:.4f} "
            f"p25={top9['p25']:.4f} p75={top9['p75']:.4f}",
            f"- Adaptive K mean={adaptive['mean']:.3f} median={adaptive['median']:.1f} "
            f"min={adaptive['min']} max={adaptive['max']}",
            f"- Adaptive K < 9: {k_stats['k_adaptive_lt_k_fixed']}",
            f"- Adaptive K = 9: {k_stats['k_adaptive_eq_k_fixed']}",
            f"- Adaptive K > 9: {k_stats['k_adaptive_gt_k_fixed']}",
            "",
        ]
    )
    if mixture and "lambda_q" in stats["topn_negative_mass"]:
        lambda_stats = stats["topn_negative_mass"]["lambda_q"]
        lines.append(
            f"- Calibrated lambda_q mean={lambda_stats['mean']:.4f} "
            f"median={lambda_stats['median']:.4f} min={lambda_stats['min']:.4f} "
            f"max={lambda_stats['max']:.4f}"
        )
        lines.append("")
    lines.extend(
        [
            "## Manifests",
            "",
        ]
    )
    for variant, path in manifest_paths.items():
        lines.append(f"- `{variant}`: `{path}` sha256=`{manifest_hashes[variant]}`")
    lines.extend(
        [
            "",
            f"- Policy: `{policy_path}`",
            f"- Per-QA table: `{per_qa_path}`",
            f"- Global statistics: `{stats_path}`",
            "",
            "## Selection rules",
            "",
            "- Random-K: uniform sample of `k_fixed` from the shared pool.",
            "- Top-K Hard: the `k_fixed` highest `candidate_score` rows in that pool.",
            "- Coverage-Adaptive K: smallest top-K whose cumulative `negative_mass`",
            "  reaches `tau`, clamped to `[k_min, k_max]`.",
            "- Soft-Mix: 6 uniform + 3 confusion-mixture draws, without replacement.",
            "- Calibrated: same negatives as Soft-Mix, with per-QA InfoNCE weight",
            "  `lambda_q = lambda_0 * clip(1 - beta * top9_mass, lambda_min/lambda_0, 1)`.",
            "",
            "## Training protocol",
            "",
            f"- Do not train `{policy['training_protocol']['do_not_train']}`.",
            "- Do not train the 3253-QA matchable-only slice.",
            "- CPU-wire the frozen manifests onto the full paper task file before any listed GPU job.",
            f"- The next listed run is `{next_variant}` on that full task file "
            f"(~{policy['training_protocol'].get('expected_paper_qa', 12014)} QA, "
            f"`accum={policy['training_protocol']['accum']}`, "
            f"`epochs={policy['training_protocol']['epochs']}`, "
            f"~{policy['training_protocol'].get('expected_listed_steps', 230)} steps).",
            "- Do not rescore the 3253×198 dump. Do not turn the 64-QA shot into an overnight run.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def verify_listed_manifest_wiring(
    *,
    task_path: Path,
    manifest_paths: dict[str, Path],
    raw_dir: Path,
    expected_n: int | None = None,
    expected_listed: int | None = None,
    variants: tuple[str, ...] | None = None,
) -> dict:
    """Attach frozen sampler manifests to a task file without loading a model."""
    payload = load_json_payload(task_path)
    if not is_grip_task_file(payload):
        raise ValueError(f"{task_path} is not a GRIP task file")
    qa_texts = list(payload["qa_samples"])
    question_ids = original_question_ids_from_payload(payload, len(qa_texts))
    identity = "original"
    if question_ids is None:
        if len(qa_texts) <= DEFAULT_MIN_TRAIN_QA:
            raise ValueError(
                f"{task_path} is missing original_question_ids; "
                "a naive prefix slice would attach the wrong frozen negatives"
            )
        question_ids = [task_qa_id(index) for index in range(len(qa_texts))]
        identity = "positional"
    if expected_n is not None and len(qa_texts) != expected_n:
        raise ValueError(f"{task_path} has {len(qa_texts)} QA rows, expected {expected_n}")
    relation_order = load_train_relation_order(raw_dir)
    golds = {question_id: assistant_gold(text) for question_id, text in zip(question_ids, qa_texts)}
    resolved_variants = tuple(variants) if variants is not None else tuple(manifest_paths)
    if not resolved_variants:
        raise ValueError("variants must be non-empty")
    unknown = [name for name in resolved_variants if name not in SAMPLER_VARIANTS]
    if unknown:
        raise ValueError(f"unknown sampler variants: {unknown}")
    missing_paths = [name for name in resolved_variants if name not in manifest_paths]
    if missing_paths:
        raise ValueError(f"missing manifest paths for {missing_paths}")
    variants_out: dict[str, dict] = {}
    for variant in resolved_variants:
        path = manifest_paths[variant]
        manifest = load_score_hard_manifest(path)
        if identity == "original":
            missing = [question_id for question_id in question_ids if question_id not in manifest]
            if missing:
                raise ValueError(f"{path} is missing {len(missing)} subset rows, e.g. {missing[0]}")
        mismatches = [
            question_id
            for question_id in question_ids
            if question_id in manifest
            and str(manifest[question_id]["positive_relation"]) != golds[question_id]
        ]
        if mismatches:
            question_id = mismatches[0]
            raise ValueError(
                f"{path} gold mismatch for {question_id}: "
                f"manifest={manifest[question_id]['positive_relation']!r} "
                f"task={golds[question_id]!r}"
            )
        _, metas = build_qa_assets_from_task_texts(
            qa_texts,
            seed=DEFAULT_SEED,
            listed_negative_source="score_hard",
            relation_order=relation_order,
            score_hard_manifest=manifest,
            question_ids=question_ids,
        )
        listed_counts = [len(meta["listed_relations"]) for meta in metas]
        missing_attached = [
            question_id
            for question_id, count in zip(question_ids, listed_counts)
            if question_id in manifest and count < 1
        ]
        if missing_attached:
            raise ValueError(f"{path} attached no negatives for {missing_attached[0]}")
        n_listed = sum(1 for count in listed_counts if count > 0)
        if expected_listed is not None and n_listed != expected_listed:
            raise ValueError(
                f"{path} attached listed negatives for {n_listed} QA rows, "
                f"expected {expected_listed}"
            )
        listed_only = [count for count in listed_counts if count > 0]
        if not listed_only:
            raise ValueError(f"{path} attached no listed negatives")
        attached = {
            meta["question_id"]: list(meta["listed_relations"])
            for meta in metas
            if meta["listed_relations"]
        }
        lambda_values = [
            float(meta["lambda_candidate"])
            for meta in metas
            if meta.get("lambda_candidate") is not None
        ]
        variants_out[variant] = {
            "manifest": str(path),
            "manifest_sha256": file_sha256(path),
            "n_qa": len(metas),
            "n_listed": n_listed,
            "k_min": min(listed_only),
            "k_max": max(listed_only),
            "k_mean": float(sum(listed_only) / len(listed_only)),
            "n_lambda": len(lambda_values),
            "lambda_min": min(lambda_values) if lambda_values else None,
            "lambda_max": max(lambda_values) if lambda_values else None,
            "question_ids": list(question_ids),
            "negatives_by_id": attached,
        }
    first_ids = variants_out[resolved_variants[0]]["question_ids"]
    for variant in resolved_variants[1:]:
        if variants_out[variant]["question_ids"] != first_ids:
            raise ValueError(f"{variant} question IDs drifted from {resolved_variants[0]}")
    listed_counts_first = variants_out[resolved_variants[0]]["n_listed"]
    return {
        "task_file": str(task_path),
        "n_qa": len(qa_texts),
        "n_listed": listed_counts_first,
        "question_id_source": identity,
        "question_ids": list(question_ids),
        "variants": {
            variant: {
                key: value
                for key, value in payload.items()
                if key not in {"negatives_by_id", "question_ids"}
            }
            for variant, payload in variants_out.items()
        },
        "same_question_ids": True,
        "note": (
            "CPU wiring check only. Listed-contrastive reads these frozen "
            "negative lists via --listed_negative_source score_hard. "
            "Do not train the retired 64-QA / 10-step protocol; train the "
            "requested frozen variant on the full paper task file after this check."
        ),
        "_negatives_by_id": {
            variant: payload["negatives_by_id"] for variant, payload in variants_out.items()
        },
    }


def compact_wiring_payload(payload: dict) -> dict:
    """Drop the 12k-ID list from logs / wiring.json; keep counts and K stats."""
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"question_ids", "_negatives_by_id"}
    }
    ids = payload.get("question_ids") or []
    compact["question_id_head"] = list(ids[:3])
    compact["question_id_tail"] = list(ids[-3:]) if ids else []
    return compact
