#!/usr/bin/env python3
"""Run one method/seed of the StructuredLoRA oracle-prefix experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from structured_lora.config import load_config
from structured_lora.experiment import run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_prefix_smoke.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    config = load_config(config_path)
    summary = run_one(
        repo_root=REPO_ROOT,
        experiment_root=EXPERIMENT_ROOT,
        config_path=config_path,
        config=config,
        method=args.method,
        seed=args.seed,
        output_dir=output_dir,
        model_override=args.model_name_or_path,
    )
    print(
        f"complete method={summary['method']} seed={summary['seed']} "
        f"test_accuracy={summary['metrics']['test']['accuracy']:.6f} output={output_dir}"
    )


if __name__ == "__main__":
    main()
