#!/usr/bin/env python3
"""CPU check: listed-contrastive can attach frozen sampler manifests.

Does not load a language model. Confirms the task file keeps original
``task_qa:N`` identities and that each requested variant supplies a
non-empty negative list. Default is the full frozen list, not a 64-QA slice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.official_lists import DEFAULT_RAW_NELL23K  # noqa: E402
from hard_negative_grip.shared_pool_samplers import (  # noqa: E402
    SAMPLER_VARIANTS,
    compact_wiring_payload,
    verify_listed_manifest_wiring,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--sampler_dir", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_NELL23K)
    parser.add_argument(
        "--expected_n",
        type=int,
        default=None,
        help="QA rows in the task file. Omit to skip the count check.",
    )
    parser.add_argument(
        "--expected_listed",
        type=int,
        default=None,
        help="Rows that must receive frozen negatives. Omit to skip.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(SAMPLER_VARIANTS),
        help="Subset of random_k / top_k_hard / coverage_adaptive_k.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.variants if name not in SAMPLER_VARIANTS]
    if unknown:
        raise ValueError(f"unknown sampler variants: {unknown}")
    payload = verify_listed_manifest_wiring(
        task_path=args.task_file,
        manifest_paths={
            variant: args.sampler_dir / f"{variant}.jsonl" for variant in args.variants
        },
        raw_dir=args.raw_dir,
        expected_n=args.expected_n,
        expected_listed=args.expected_listed,
        variants=tuple(args.variants),
    )
    compact = compact_wiring_payload(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
