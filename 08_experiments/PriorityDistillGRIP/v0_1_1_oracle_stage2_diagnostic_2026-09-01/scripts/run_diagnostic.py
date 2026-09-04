#!/usr/bin/env python3
"""Run the Oracle Stage-2 diagnostic and save checkpoint/evaluation curves."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from priority_distill.config import load_config
from priority_distill.experiment import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_stage2_diagnostic.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    config = load_config(config_path)
    summary = run_one(
        repo_root=REPO_ROOT,
        config_path=config_path,
        config=config,
        output_dir=output_dir,
        model_override=args.model_name_or_path,
    )
    final = summary["diagnostic_metrics_file"]
    print(f"complete diagnostic seed={summary['seed']} stage2_epochs={len(summary['training']['stage2_epochs'])} metrics={final} output={output_dir}")


if __name__ == "__main__":
    main()
