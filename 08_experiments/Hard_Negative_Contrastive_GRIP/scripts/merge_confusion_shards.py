#!/usr/bin/env python3
"""Merge sharded offline confusion dumps into one deterministic JSONL pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.offline_mining import merge_confusion_shards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_confusion_shards(args.shard_dirs, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
