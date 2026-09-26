"""Frozen Random-K / Top-K Hard / Coverage-Adaptive K samplers.

All three variants read the same train-only valid-negative pool from a
Prompt-4/7 score dump. They never rescore and never consult valid/test KG
structure. Coverage-Adaptive K is derived from each QA's ``negative_mass``
distribution rather than a global fixed K.
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
from .score_hard import merge_negative_sources

SAMPLER_VARIANTS = ("random_k", "top_k_hard", "coverage_adaptive_k")
DEFAULT_K_FIXED = 9
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 20
DEFAULT_SEED = 2026
DEFAULT_TAU_SOURCE = "median_top9_negative_mass"


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
) -> dict:
    identity = qa_identity(rows)
    pool = shared_valid_pool(rows)
    ranked = rank_valid_pool(pool)
    masses = pool_masses_desc(ranked)
    n_valid = len(pool)
    k_random = min(k_fixed, n_valid)
    k_top = min(k_fixed, n_valid)
    k_adaptive = decide_coverage_k(masses, tau=tau, k_min=k_min, k_max=k_max)

    random_rows = sample_random_k(pool, k_random, rng)
    top_rows = sample_top_k(ranked, k_top)
    adaptive_rows = sample_top_k(ranked, k_adaptive)

    analysis = analyze_qa_group(rows)
    per_qa = {
        "qa_id": identity["question_id"],
        "positive_relation": identity["positive_relation"],
        "matched_train_relation": identity["matched_train_relation"],
        "n_valid_negatives": n_valid,
        "k_random": k_random,
        "k_top": k_top,
        "k_adaptive": k_adaptive,
        "adaptive_covered_mass": covered_mass(ranked, k_adaptive),
        "random_covered_mass": sum(float(row["negative_mass"]) for row in random_rows),
        "top_covered_mass": covered_mass(ranked, k_top),
        "top1_mass": analysis["top1_mass"],
        "top9_mass": analysis["top9_mass"],
        "top20_mass": analysis["top20_mass"],
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
) -> dict:
    """Read the train-only dump, freeze tau, and write three immutable manifests."""
    if k_fixed < 0:
        raise ValueError(f"k_fixed must be non-negative, got {k_fixed}")
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
        )
        for group in groups
    ]
    per_qa = [item["per_qa"] for item in sampled]
    manifests = {
        variant: [item[variant] for item in sampled] for variant in SAMPLER_VARIANTS
    }

    k_adaptive_values = [int(row["k_adaptive"]) for row in per_qa]
    topn_stats = {
        name: summarize_values([float(row[name]) for row in per_qa])
        for name in ("top1_mass", "top9_mass", "top20_mass", "random_covered_mass", "top_covered_mass", "adaptive_covered_mass")
    }
    k_stats = {
        "k_fixed": k_fixed,
        "k_adaptive": summarize_values([float(value) for value in k_adaptive_values]),
        "k_adaptive_eq_k_fixed": sum(1 for value in k_adaptive_values if value == k_fixed),
        "k_adaptive_lt_k_fixed": sum(1 for value in k_adaptive_values if value < k_fixed),
        "k_adaptive_gt_k_fixed": sum(1 for value in k_adaptive_values if value > k_fixed),
        "k_80": summarize_values([float(row["K_80"]) for row in per_qa]),
        "k_90": summarize_values([float(row["K_90"]) for row in per_qa]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        variant: output_dir / f"{variant}.jsonl" for variant in SAMPLER_VARIANTS
    }
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
        "k_min": k_min,
        "k_max": k_max,
        "coverage_tau": frozen_tau,
        "coverage_tau_source": DEFAULT_TAU_SOURCE if tau is None else "cli_override",
        "seed": seed,
        "variants": list(SAMPLER_VARIANTS),
        "selection_rules": {
            "random_k": "uniform_without_replacement from shared valid pool, K=k_fixed",
            "top_k_hard": "top k_fixed by candidate_score, relation name breaks ties",
            "coverage_adaptive_k": (
                "K_i = clip(smallest K covering coverage_tau of negative_mass, "
                "k_min, k_max); then take top K_i"
            ),
        },
        "do_not_train_from": "results/runs/20260923_offline_confusion_full",
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
    lines = [
        "# Frozen shared-pool samplers",
        "",
        "These manifests are the only negative lists for the next listed-contrastive",
        "control: Random-K vs Top-K Hard vs Coverage-Adaptive K. They all read the",
        "train-only valid-negative pool. Do not resample live and do not use the",
        "all-split dump under `20260923_offline_confusion_full`.",
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
        "## Manifests",
        "",
    ]
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
            "",
        ]
    )
    return "\n".join(lines) + "\n"
