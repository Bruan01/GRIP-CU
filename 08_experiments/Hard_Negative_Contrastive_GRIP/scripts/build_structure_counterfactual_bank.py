#!/usr/bin/env python3
"""Freeze structure-aware counterfactual negatives from a train-only B1 dump.

The input is the existing full-vocabulary ``candidate_scores.jsonl`` produced
by ``score_relation_candidates_offline.py``.  No model is loaded and no
training is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.counterfactuals import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    DEFAULT_TOTAL_K,
    DEFAULT_UNIFORM_MIX,
    DEFAULT_VIOLATION_WEIGHT,
    DEFAULT_STRUCTURE_WEIGHT,
    load_train_relation_profiles,
    write_bank_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--relation_text_file",
        type=Path,
        default=None,
        help="Raw-relation to task-label mapping, required for FB15K237-2.",
    )
    parser.add_argument(
        "--strip_first_underscore",
        action="store_true",
        help="Use WN18RR's relation-label conversion: '_'.join(parts[1:]).",
    )
    parser.add_argument("--total_k", type=int, default=DEFAULT_TOTAL_K)
    parser.add_argument("--uniform_mix", type=float, default=DEFAULT_UNIFORM_MIX)
    parser.add_argument(
        "--structure_weight", type=float, default=DEFAULT_STRUCTURE_WEIGHT
    )
    parser.add_argument(
        "--violation_weight", type=float, default=DEFAULT_VIOLATION_WEIGHT
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing frozen output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_k < 1:
        raise ValueError("--total_k must be positive")
    profiles = load_train_relation_profiles(
        args.raw_dir,
        relation_text_file=args.relation_text_file,
        strip_first_underscore=args.strip_first_underscore,
    )
    metadata = write_bank_artifacts(
        score_path=args.scores,
        output_dir=args.output_dir,
        profiles=profiles,
        total_k=args.total_k,
        uniform_mix=args.uniform_mix,
        structure_weight=args.structure_weight,
        violation_weight=args.violation_weight,
        temperature=args.temperature,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
