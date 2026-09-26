#!/usr/bin/env python3
"""Freeze Random-K / Top-K Hard / Coverage-Adaptive K from a train-only dump.

CPU-only. Does not load a language model, does not rescore, and does not train.
The three manifests share the same valid-negative pool from the train-only
``is_valid_negative`` labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.shared_pool_samplers import (  # noqa: E402
    DEFAULT_K_FIXED,
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_SEED,
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override coverage tau. Default: median Top-k_fixed negative mass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = freeze_shared_pool_samplers(
        scores_path=args.scores,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        k_fixed=args.k_fixed,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
        tau=args.tau,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
