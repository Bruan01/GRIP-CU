#!/usr/bin/env python3
"""Launch the registered method matrix in isolated subprocesses."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from structured_lora.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_prefix_smoke.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--full-seeds", action="store_true", help="Run all registered seeds instead of the one-seed smoke.")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs and replace only incomplete run folders.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    methods = args.methods or config["methods"]
    unknown = sorted(set(methods) - set(config["methods"]))
    if unknown:
        raise ValueError(f"methods absent from config: {unknown}")
    seeds = config["training"]["seeds" if args.full_seeds else "smoke_seeds"]
    run_id = args.run_id or f"wsl3090_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = EXPERIMENT_ROOT / "results" / "runs" / run_id
    commands: list[list[str]] = []
    for method in methods:
        for seed in seeds:
            output_dir = run_root / method / f"seed_{seed}"
            if args.resume and (output_dir / "run_summary.json").is_file():
                print(f"SKIP completed {method} seed={seed} output={output_dir}")
                continue
            replace_partial = args.resume and output_dir.exists() and any(output_dir.iterdir())
            command = [
                args.python_executable,
                str(EXPERIMENT_ROOT / "scripts/run_experiment.py"),
                "--config",
                str(config_path),
                "--method",
                method,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            if args.model_name_or_path:
                command += ["--model-name-or-path", args.model_name_or_path]
            if args.overwrite or replace_partial:
                command.append("--overwrite")
            commands.append(command)

    print(f"run_root={run_root}")
    for command in commands:
        print("COMMAND", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=EXPERIMENT_ROOT)
    summary_command = [
        args.python_executable,
        str(EXPERIMENT_ROOT / "scripts/summarize_suite.py"),
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
    ]
    print("COMMAND", shlex.join(summary_command), flush=True)
    if not args.dry_run:
        subprocess.run(summary_command, check=True, cwd=EXPERIMENT_ROOT)


if __name__ == "__main__":
    main()
