#!/usr/bin/env python3
"""Freeze shared-pool listed-contrastive samplers from a train-only dump.

CPU-only. Does not load a language model, does not rescore, and does not train.
Control variants (Random-K / Top-K Hard / Coverage-Adaptive K) stay frozen.
Soft-Mix / Calibrated write additional manifests from the same valid-negative
pool without resampling the already-frozen control lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.shared_pool_samplers import (  # noqa: E402
    CONTROL_SAMPLER_VARIANTS,
    DEFAULT_K_FIXED,
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_K_SOFT,
    DEFAULT_K_UNIFORM,
    DEFAULT_LAMBDA_0,
    DEFAULT_LAMBDA_BETA,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_SEED,
    DEFAULT_SOFT_RHO,
    SAMPLER_VARIANTS,
    freeze_shared_pool_samplers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--k_fixed", type=int, default=DEFAULT_K_FIXED)
    parser.add_argument("--k_min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k_max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--k_uniform", type=int, default=DEFAULT_K_UNIFORM)
    parser.add_argument("--k_soft", type=int, default=DEFAULT_K_SOFT)
    parser.add_argument("--rho", type=float, default=DEFAULT_SOFT_RHO)
    parser.add_argument("--lambda_0", type=float, default=DEFAULT_LAMBDA_0)
    parser.add_argument("--lambda_min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda_beta", type=float, default=DEFAULT_LAMBDA_BETA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override coverage tau. Default: median Top-k_fixed negative mass.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(CONTROL_SAMPLER_VARIANTS),
        help="Subset of random_k / top_k_hard / coverage_adaptive_k / soft_mix / calibrated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.variants if name not in SAMPLER_VARIANTS]
    if unknown:
        raise ValueError(f"unknown sampler variants: {unknown}")
    summary = freeze_shared_pool_samplers(
        scores_path=args.scores,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        k_fixed=args.k_fixed,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
        tau=args.tau,
        k_uniform=args.k_uniform,
        k_soft=args.k_soft,
        rho=args.rho,
        lambda_0=args.lambda_0,
        lambda_min=args.lambda_min,
        lambda_beta=args.lambda_beta,
        variants=tuple(args.variants),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
