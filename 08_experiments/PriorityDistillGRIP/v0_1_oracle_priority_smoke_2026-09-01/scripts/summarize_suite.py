#!/usr/bin/env python3
"""Aggregate completed PriorityDistill runs and apply the oracle gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from priority_distill.config import load_config
from priority_distill.suite import write_suite_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_priority_smoke.json")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    payload = write_suite_outputs(args.run_root.expanduser().resolve(), load_config(args.config.expanduser().resolve()))
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
