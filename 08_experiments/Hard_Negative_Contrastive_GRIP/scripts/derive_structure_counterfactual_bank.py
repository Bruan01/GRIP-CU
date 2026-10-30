#!/usr/bin/env python3
"""Derive an exact-K bank from a frozen structure-counterfactual max-K bank.

This CPU-only step reuses the deterministic prefix of each without-replacement
candidate sequence. It performs no B1 scoring and no model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.counterfactuals import (  # noqa: E402
    derive_structure_counterfactual_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--total_k", type=int, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing derived bank outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = derive_structure_counterfactual_bank(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        total_k=args.total_k,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
