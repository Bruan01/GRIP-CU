#!/usr/bin/env python3
"""Aggregate completed StructuredLoRA method runs and apply the go/no-go gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from structured_lora.config import load_config
from structured_lora.suite import write_suite_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_prefix_smoke.json")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.expanduser().resolve())
    payload = write_suite_outputs(args.run_root.expanduser().resolve(), config)
    print(f"decision={payload['gate']['status']} runs={payload['run_count']} root={args.run_root}")


if __name__ == "__main__":
    main()
