#!/usr/bin/env python3
"""CPU check: listed-contrastive can attach frozen sampler manifests.

Does not load a language model. Confirms the smoke subset keeps original
``task_qa:N`` identities and that Random-K / Top-K Hard / Coverage-Adaptive K
each supply a non-empty negative list for every selected QA.
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
    verify_listed_manifest_wiring,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--sampler_dir", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_NELL23K)
    parser.add_argument("--expected_n", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify_listed_manifest_wiring(
        task_path=args.task_file,
        manifest_paths={
            variant: args.sampler_dir / f"{variant}.jsonl" for variant in SAMPLER_VARIANTS
        },
        raw_dir=args.raw_dir,
        expected_n=args.expected_n,
    )
    payload.pop("_negatives_by_id", None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
