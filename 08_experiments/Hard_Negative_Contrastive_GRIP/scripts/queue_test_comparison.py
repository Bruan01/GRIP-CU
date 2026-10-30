#!/usr/bin/env python3
"""Serial test comparison of the four completed three-epoch validation trials."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import search_validation_hparams as hpo

ORDER = (1, 3, 2, 4)


def output_path(search, number, winner):
    return search / "selected_test" if number == winner else search / "test_comparison" / f"trial_{number:03d}"


def publish(search, records, winner, status):
    rows = []
    for number in ORDER:
        trial = next(row for row in records if row["number"] == number)
        directory = output_path(search, number, winner)
        row = {"number": number, "params": trial["params"],
               "validation_em": trial["validation_em"], "validation_selected": number == winner,
               "output_dir": str(directory), "test_em": None, "state": "PENDING"}
        summary_path = directory / "summary.json"
        config_path = directory / "test_config.json"
        if summary_path.exists() and config_path.exists():
            summary = hpo.read_json(summary_path)["test"]
            config = hpo.read_json(config_path)
            if (config["selection"]["number"] != number or summary["count"] != config["test_count"]
                    or summary["official_harness_count"] != config["test_count"] - 1):
                raise ValueError(f"Incomplete/mismatched test result for trial {number}")
            row.update(state="COMPLETE", test_em=summary["official_harness_em"])
        rows.append(row)
    result = {"status": status, "validation_selected_trial": winner,
              "interpretation": "Across-model test comparison is exploratory; validation selection remains fixed",
              "rows": rows}
    directory = search / "test_comparison"
    hpo.write_json(directory / "comparison.json", result)
    lines = ["# Test comparison\n\n",
             "Across-model test comparison is exploratory. The validation-selected model remains fixed.\n\n",
             f"Status: {status}\n\n",
             "| Trial | Method | Scale | Epochs | Validation EM | Test EM (official harness) | Validation selected |\n",
             "| --- | --- | --- | --- | --- | --- | --- |\n"]
    for row in rows:
        p = row["params"]
        score = "pending" if row["test_em"] is None else f'{100 * row["test_em"]:.4f}%'
        lines.append(f'| {row["number"]} | {p["method"]} | {p["lambda_scale"]} | {p["epochs"]} | '
                     f'{100 * row["validation_em"]:.4f}% | {score} | {row["validation_selected"]} |\n')
    hpo.atomic_text(directory / "COMPARISON.md", "".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_dir", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    search = args.search_dir.resolve()
    selected = hpo.read_json(search / "best.json")
    if not selected["finalized"]:
        raise ValueError("Validation selection is not finalized")
    winner = selected["best"]["number"]
    if winner not in ORDER:
        raise ValueError("Validation winner is outside the requested four-trial comparison")
    records = hpo.read_json(search / "trials.json")
    for number in ORDER:
        trial = next(row for row in records if row["number"] == number)
        if trial["state"] != "COMPLETE" or trial["params"]["epochs"] != 3:
            raise ValueError("Only completed three-epoch trials may enter this queue")
    commands = []
    for number in ORDER:
        directory = output_path(search, number, winner)
        commands.append((number, directory, [sys.executable, str(Path(__file__).with_name("eval_validation_selected.py")),
            "--search_dir", str(search), "--eval_file", str(args.eval_file.resolve()),
            "--output_dir", str(directory), "--model_cache_dir", str(args.model_cache_dir.resolve()),
            "--trial_number", str(number), "--wait_for_pipeline"]))
    if args.dry_run:
        for _, _, command in commands:
            subprocess.run(command + ["--dry_run"], check=True)
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Launch with configs/run_test_comparison_queue.sh in tmux")
    directory = search / "test_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "queue.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hpo.freeze_text(directory / "queue_config.json", json.dumps({
            "order": ORDER, "validation_selected_trial": winner,
            "selection_sha256": hpo.sha256(search / "best.json"),
            "commands": [command for _, _, command in commands]}, sort_keys=True, indent=2) + "\n")
        publish(search, records, winner, "RUNNING")
        try:
            for number, output, command in commands:
                output.mkdir(parents=True, exist_ok=True)
                print(f"[queue] trial={number} waiting/running; log={output / 'run.log'}", flush=True)
                with (output / "run.log").open("a") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=(lock.fileno(),))
                publish(search, records, winner, "RUNNING")
        except Exception:
            publish(search, records, winner, "INTERRUPTED")
            raise
        rows = publish(search, records, winner, "COMPLETE")
        if any(row["state"] != "COMPLETE" for row in rows):
            raise RuntimeError("Queue exited without complete summaries")
        print(f"[queue] complete: {directory / 'COMPARISON.md'}", flush=True)


if __name__ == "__main__":
    main()
