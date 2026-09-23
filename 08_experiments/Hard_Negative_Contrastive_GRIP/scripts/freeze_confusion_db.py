#!/usr/bin/env python3
"""Validate a mined full-vocab score table and freeze it as a confusion DB.

This is offline. It does not load the 7B teacher. Later listed sampling should
read the frozen JSONL and never rescore candidates live.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.confusion_db import (  # noqa: E402
    DEFAULT_B1_ADAPTER,
    DEFAULT_SEED,
    freeze_confusion_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Mined score-hard JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Frozen confusion DB JSONL.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--b1_adapter", default=DEFAULT_B1_ADAPTER)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = freeze_confusion_db(
        task_path=args.task_file,
        raw_dir=args.raw_dir,
        input_jsonl=args.input,
        output_jsonl=args.output,
        metadata_path=args.metadata,
        b1_adapter=args.b1_adapter,
        candidate_batch_size=args.candidate_batch_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
