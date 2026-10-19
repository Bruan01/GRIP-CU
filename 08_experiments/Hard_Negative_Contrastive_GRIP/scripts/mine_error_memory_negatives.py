"""Freeze GRIP-conditioned error-memory negatives from train-only artifacts.

This is intentionally CPU-only. It consumes a completed B1 candidate-score
dump and a completed B1 free-generation rollout manifest; it never loads a
model and never trains. The output is an immutable K=9 JSONL bank plus the
error memory, policy, and audit records used by the WSL launcher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.error_memory_sampling import (  # noqa: E402
    DEFAULT_ERROR_WEIGHT,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_TOTAL_K,
    DEFAULT_UNIFORM_MIX,
    freeze_error_memory_bank,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    load_train_relation_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--rollout_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help=f"NELL23K raw split directory; default convention is {DEFAULT_RAW_NELL23K}.",
    )
    parser.add_argument("--total_k", type=int, default=DEFAULT_TOTAL_K)
    parser.add_argument("--uniform_mix", type=float, default=DEFAULT_UNIFORM_MIX)
    parser.add_argument("--error_weight", type=float, default=DEFAULT_ERROR_WEIGHT)
    parser.add_argument("--violation_weight", type=float, default=1.0)
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=DEFAULT_SAMPLING_TEMPERATURE,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing frozen output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_k != DEFAULT_TOTAL_K:
        raise ValueError("the current GRIP decision-set contract is fixed K=9")
    relation_order = load_train_relation_order(args.raw_dir)
    result = freeze_error_memory_bank(
        scores_path=args.scores,
        metadata_path=args.metadata,
        rollout_path=args.rollout_manifest,
        output_dir=args.output_dir,
        relation_order=relation_order,
        total_k=args.total_k,
        uniform_mix=args.uniform_mix,
        error_weight=args.error_weight,
        violation_weight=args.violation_weight,
        sampling_temperature=args.sampling_temperature,
        seed=args.seed,
        min_support=args.min_support,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
