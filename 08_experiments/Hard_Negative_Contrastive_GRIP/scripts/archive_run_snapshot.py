#!/usr/bin/env python3
"""Write a never-overwritten lightweight snapshot of a finished run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_negative_grip.archive_results import archive_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--archive_root", type=Path, default=None)
    parser.add_argument("--stamp", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = archive_run(args.run_dir, archive_root=args.archive_root, stamp=args.stamp)
    print(dest)


if __name__ == "__main__":
    main()
