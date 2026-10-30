#!/usr/bin/env python3
"""Evaluate the finalized validation winner on test with the same generation protocol."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import search_validation_hparams as hpo
from hard_negative_grip.decode_io import eval_samples
from hard_negative_grip.task_file import load_graph_record
from train_listed_contrastive import evaluate_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_dir", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--trial_number", type=int, default=None,
                        help="Explicit completed trial for exploratory test comparison")
    parser.add_argument("--wait_for_pipeline", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    for name in ("search_dir", "eval_file", "output_dir", "model_cache_dir"):
        setattr(args, name, getattr(args, name).resolve())
    selected = hpo.read_json(args.search_dir / "best.json")
    if not selected["finalized"] or selected["status"] not in ("COMPLETE", "BUDGET_EXHAUSTED"):
        raise ValueError("Validation selection must be finalized before test evaluation")
    best = selected["best"]
    if args.trial_number is not None:
        best = next((row for row in hpo.read_json(args.search_dir / "trials.json")
                     if row["number"] == args.trial_number), None)
    if not best or best["state"] != "COMPLETE":
        raise ValueError("Requested trial must be complete")
    trial = Path(best["directory"])
    adapter = trial / "listed/adapter"
    frozen = hpo.read_json(args.search_dir / "search_config.json")
    if frozen["files"].get(str(args.eval_file)) != hpo.sha256(args.eval_file):
        raise ValueError("Evaluation dataset differs from the frozen search input")
    command = hpo.read_json(trial / "trial.json")["command"]
    record = load_graph_record(args.eval_file)
    samples = eval_samples(record, ("test",))
    if not samples:
        raise ValueError("Test split is empty")
    config = {
        "selection": best,
        "selection_sha256": hpo.sha256(args.search_dir / "best.json"),
        "eval_file": str(args.eval_file),
        "eval_sha256": hpo.sha256(args.eval_file),
        "adapter": str(adapter),
        "adapter_sha256": hpo.sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": hpo.sha256(adapter / "adapter_config.json"),
        "model_name": command[command.index("--model_name") + 1],
        "gen_max_length": int(command[command.index("--gen_max_length") + 1]),
        "official_drop_last_line": "--official_drop_last_line" in command,
        "splits": ["test"], "test_count": len(samples), "training": False,
    }
    if best["number"] != selected["best"]["number"]:
        config["selection_role"] = "exploratory_test_comparison"
        config["validation_selected_trial"] = selected["best"]["number"]
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Launch through configs/run_validation_selected_test.sh in tmux")
    with (Path(frozen["pipeline_root"]) / "pipeline.lock").open("a") as lock:
        print("[test] waiting for exclusive pipeline access", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX if args.wait_for_pipeline else fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        hpo.freeze_text(args.output_dir / "test_config.json", json.dumps(config, sort_keys=True, indent=2) + "\n")
        eval_args = SimpleNamespace(model_name=config["model_name"],
            model_cache_dir=str(args.model_cache_dir), gen_max_length=config["gen_max_length"],
            official_drop_last_line=config["official_drop_last_line"], eval_splits=["test"], no_resume=False)
        print(f"[test] trial={trial.name}; test questions={len(samples)}; resume enabled", flush=True)
        summary = evaluate_variant(eval_args, record, adapter, args.output_dir)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
