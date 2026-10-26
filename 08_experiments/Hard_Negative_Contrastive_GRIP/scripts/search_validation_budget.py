#!/usr/bin/env python3
"""Fixed 60-hour validation search, adopting an already running 10-epoch anchor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import search_validation_hparams as hpo

SCREEN = [("soft_mix", 1.0), ("soft_mix", 0.5),
          ("calibrated", 1.0), ("calibrated", 0.5)]
ANCHOR = {"method": "soft_mix", "lambda_scale": 1.0, "epochs": 10}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input_file", "eval_file", "s1_adapter", "calibrated_manifest",
                 "raw_dir", "model_cache_dir", "search_dir", "pipeline_root", "source_search_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source_supervisor_pid", type=int, required=True)
    parser.add_argument("--source_training_pid", type=int, required=True)
    parser.add_argument("--started_at", required=True, help="UTC ISO timestamp of the original launch")
    parser.add_argument("--budget_hours", type=float, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    started = datetime.fromisoformat(args.started_at.replace("Z", "+00:00"))
    if started.tzinfo is None:
        parser.error("started_at must include a timezone")
    args.started_at = started.astimezone(timezone.utc).isoformat()
    if not math.isfinite(args.budget_hours) or not 0 < args.budget_hours <= 72:
        parser.error("budget_hours must be positive and <= 72")
    if args.source_search_dir == args.search_dir:
        parser.error("Use a new search_dir; the source study is archived unchanged")
    return args


def deadline(args):
    return datetime.fromisoformat(args.started_at).timestamp() + args.budget_hours * 3600


def remaining(args):
    return max(0.0, deadline(args) - time.time())


def identity(pid):
    """Linux PID identity, including start ticks to detect PID reuse."""
    try:
        base = Path(f"/proc/{pid}")
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "command": (base / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except FileNotFoundError:
        return None


def same_process(saved):
    return saved is not None and identity(saved["pid"]) == saved


def checked_signal(saved, sig):
    if same_process(saved):
        os.kill(saved["pid"], sig)


def stop_adopted_processes(args):
    path = args.search_dir / "handoff.json"
    if path.exists():
        saved = hpo.read_json(path)
        checked_signal(saved["training"], signal.SIGTERM)
        checked_signal(saved["supervisor"], signal.SIGTERM)
        checked_signal(saved["supervisor"], signal.SIGCONT)


def source_trial(args):
    return args.source_search_dir / "trial_000"


def verify_source(args):
    """Refuse to adopt a run with different training inputs or hyperparameters."""
    trial_dir = source_trial(args)
    trial = hpo.read_json(trial_dir / "trial.json")
    if trial["params"] != ANCHOR:
        raise ValueError("Only the predeclared Soft-Mix scale=1, epochs=10 anchor can be adopted")
    child_args = argparse.Namespace(**vars(args))
    child_args.eval_file = args.source_search_dir / "validation_only.json"
    expected = hpo.training_command(child_args, ANCHOR, trial_dir)
    if expected[-1] == "--skip_train":
        expected.pop()
    if trial["command"] != expected:
        raise ValueError("Source training command differs from the new protocol")
    frozen = hpo.read_json(args.source_search_dir / "search_config.json")
    if frozen["seed"] != args.seed:
        raise ValueError("Source seed differs")
    # Verify all old input/source/S1 hashes. The old runner itself stays unchanged.
    for path, expected_hash in frozen["files"].items():
        if hpo.sha256(path) != expected_hash:
            raise ValueError(f"Source input/code changed: {path}")
    if hpo.validation_record(args.eval_file) != hpo.read_json(child_args.eval_file):
        raise ValueError("Source validation input differs")
    return trial


def frozen_config(args):
    paths = hpo.required_files(args) + [Path(__file__).resolve(), Path(hpo.__file__).resolve(),
                                      hpo.TRAIN_SCRIPT]
    paths += sorted((hpo.HNG / "src/hard_negative_grip").glob("*.py"))
    paths += [args.source_search_dir / "search_config.json", source_trial(args) / "trial.json",
              source_trial(args) / "negatives.jsonl", args.source_search_dir / "validation_only.json"]
    return {"version": 1, "protocol": "budget60_v1", "seed": args.seed,
            "started_at": args.started_at, "deadline_utc": datetime.fromtimestamp(
                deadline(args), timezone.utc).isoformat(), "budget_hours": args.budget_hours,
            "source_search_dir": str(args.source_search_dir),
            "source_supervisor_pid": args.source_supervisor_pid,
            "source_training_pid": args.source_training_pid,
            "python": sys.executable, "model_cache_dir": str(args.model_cache_dir),
            "pipeline_root": str(args.pipeline_root),
            "anchor": ANCHOR, "screen": [{"method": m, "lambda_scale": s, "epochs": 3}
                                          for m, s in SCREEN],
            "confirmation": {"epochs": 5, "select_from": "screen_only", "tie": "earlier_trial",
                             "restart_from": "shared_S1"},
            "max_total_epochs": 27, "max_trials": 6,
            "metric": "validation.official_harness_em", "test_evaluation": False,
            "files": {str(p): hpo.sha256(p) for p in paths}}


def handoff(args):
    """Pause only the old search parent; its GPU child keeps its original tmux."""
    path = args.search_dir / "handoff.json"
    if path.exists():
        saved = hpo.read_json(path)
    else:
        supervisor = identity(args.source_supervisor_pid)
        training = identity(args.source_training_pid)
        if supervisor:
            command = supervisor["command"]
            if (str(Path(hpo.__file__).resolve()) not in command or "--search_dir" not in command
                    or command[command.index("--search_dir") + 1] != str(args.source_search_dir)):
                raise ValueError("Source supervisor PID does not match the source study")
        if training and training["command"] != hpo.read_json(source_trial(args) / "trial.json")["command"]:
            raise ValueError("Source training PID does not match the source trial")
        saved = {"supervisor": supervisor, "training": training,
                 "reason": "User requested a 2-3 day budget before any HPO validation score",
                 "original_study": "Superseded; trial_000 result adopted, queued trials not executed"}
        hpo.write_json(path, saved)
    checked_signal(saved["supervisor"], signal.SIGSTOP)
    print("[budget] Old search parent paused; existing training continues unchanged", flush=True)
    while same_process(saved["training"]):
        if remaining(args) <= 0:
            checked_signal(saved["training"], signal.SIGTERM)
            # Release the stopped parent's locks too. Checkpoint files remain intact.
            checked_signal(saved["supervisor"], signal.SIGTERM)
            checked_signal(saved["supervisor"], signal.SIGCONT)
            raise TimeoutError("Deadline reached during adopted anchor; latest checkpoint retained")
        time.sleep(min(5, remaining(args)))
    # Child has exited, so the suspended parent can no longer advance the old queue.
    checked_signal(saved["supervisor"], signal.SIGTERM)
    checked_signal(saved["supervisor"], signal.SIGCONT)
    hpo.write_json(args.search_dir / "source_status.json", {
        "state": "CHILD_EXITED", "original_study": str(args.source_search_dir),
        "old_queue": "superseded", "source_result": str(source_trial(args) / "listed/summary.json")})


def publish(args, records, status):
    hpo.write_json(args.search_dir / "trials.json", records)
    completed = [r for r in records if r["state"] == "COMPLETE"]
    best = max(completed, key=lambda r: (r["validation_em"], -r["number"])) if completed else None
    hpo.write_json(args.search_dir / "best.json", {
        "metric": "validation.official_harness_em", "status": status,
        "finalized": status in ("COMPLETE", "BUDGET_EXHAUSTED"), "best": best,
        "best_by_method": {m: max((r for r in completed if r["params"]["method"] == m),
                                  key=lambda r: (r["validation_em"], -r["number"]), default=None)
                           for m in ("soft_mix", "calibrated")}})
    hpo.write_json(args.search_dir / "run_status.json", {
        "status": status, "completed": len(completed), "planned": 6,
        "remaining_seconds": remaining(args), "deadline_utc": datetime.fromtimestamp(
            deadline(args), timezone.utc).isoformat()})
    lines = ["# 60-hour validation search\n\n",
             "Supersedes the old 8-trial plan before its first validation result. Test is never used.\n\n",
             f"Status: {status}. Only COMPLETE full-validation results are eligible.\n\n",
             "| Trial | State | Parameters | Validation EM | Selected |\n",
             "| --- | --- | --- | --- | --- |\n"]
    for row in records:
        value = "" if row.get("validation_em") is None else f'{100 * row["validation_em"]:.4f}%'
        lines.append(f'| {row["number"]} | {row["state"]} | {json.dumps(row["params"], sort_keys=True)} '
                     f'| {value} | {"yes" if row == best else ""} |\n')
    hpo.atomic_text(args.search_dir / "VARIANTS.md", "".join(lines))


def parameters(number, records):
    if number == 0:
        return dict(ANCHOR)
    if number <= len(SCREEN):
        method, scale = SCREEN[number - 1]
        return {"method": method, "lambda_scale": scale, "epochs": 3}
    # Compare like-for-like 3-epoch trials; retain 10-epoch anchor in overall selection.
    screens = [r for r in records if 1 <= r["number"] <= 4 and r["state"] == "COMPLETE"]
    if len(screens) != 4:
        raise ValueError("All four screens must complete before confirmation")
    best = max(screens, key=lambda r: (r["validation_em"], -r["number"]))
    return dict(best["params"], epochs=5)


def run_search(args, execute=None, adopt=None):
    args.search_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        own_lock = stack.enter_context((args.search_dir / "search.lock").open("a"))
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = frozen_config(args)
        hpo.freeze_text(args.search_dir / "search_config.json", json.dumps(config, sort_keys=True, indent=2) + "\n")
        record = hpo.validation_record(args.eval_file)
        if len(record["recurrent_questions"]) != hpo.EXPECTED_VALIDATION_COUNT:
            raise ValueError("Full NELL23K validation required")
        validation_file = args.search_dir / "validation_only.json"
        hpo.freeze_text(validation_file, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        records = hpo.read_json(args.search_dir / "trials.json") if (args.search_dir / "trials.json").exists() else []
        if len(records) > 6 or any(r["number"] != i for i, r in enumerate(records)):
            raise ValueError("Corrupt trial ledger")
        if len(records) == 6 and all(r["state"] == "COMPLETE" for r in records):
            publish(args, records, "COMPLETE")
            return
        if remaining(args) <= 0:
            stop_adopted_processes(args)
            publish(args, records, "BUDGET_EXHAUSTED")
            return
        publish(args, records, "WAITING_FOR_ANCHOR")
        try:
            (adopt or handoff)(args)
        except TimeoutError:
            publish(args, records, "BUDGET_EXHAUSTED")
            return
        pipeline_lock = stack.enter_context((args.pipeline_root / "pipeline.lock").open("a"))
        while True:
            try:
                fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if remaining(args) <= 0:
                    publish(args, records, "BUDGET_EXHAUSTED")
                    return
                time.sleep(min(2, remaining(args)))
        # The original child inherited both old locks. No next GPU task starts until release.
        if not records:
            summary_path = source_trial(args) / "listed/summary.json"
            try:
                value = hpo.validation_value(summary_path, len(record["recurrent_questions"]))
            except (FileNotFoundError, KeyError, ValueError):
                # A machine/process failure during the anchor resumes its original checkpoint.
                child_args = argparse.Namespace(**vars(args))
                child_args.eval_file = args.source_search_dir / "validation_only.json"
                command = hpo.training_command(child_args, ANCHOR, source_trial(args))
                print("[budget] Anchor incomplete; resume original directory/checkpoint", flush=True)
                try:
                    if execute is None:
                        with (source_trial(args) / "run.log").open("a") as log:
                            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(own_lock.fileno(), pipeline_lock.fileno()),
                                           timeout=remaining(args))
                    else:
                        execute(command, source_trial(args))
                    value = hpo.validation_value(summary_path, len(record["recurrent_questions"]))
                except (subprocess.TimeoutExpired, TimeoutError):
                    publish(args, records, "BUDGET_EXHAUSTED")
                    return
                except Exception:
                    publish(args, records, "INTERRUPTED")
                    raise
            records.append({"number": 0, "params": dict(ANCHOR), "state": "COMPLETE",
                            "validation_em": value, "directory": str(source_trial(args)),
                            "adopted": True, "summary_sha256": hpo.sha256(source_trial(args) / "listed/summary.json")})
            publish(args, records, "RUNNING")
        for number in range(1, 6):
            params = parameters(number, records)
            if number < len(records):
                row = records[number]
                if row["params"] != params:
                    raise ValueError("Recorded parameters differ from frozen schedule")
                if row["state"] == "COMPLETE":
                    continue
            else:
                row = {"number": number, "params": params, "state": "RUNNING", "validation_em": None,
                       "directory": str(args.search_dir / f"trial_{number:03d}")}
                records.append(row)
            if remaining(args) <= 0:
                row["state"] = "BUDGET_EXHAUSTED"
                publish(args, records, "BUDGET_EXHAUSTED")
                return
            directory = Path(row["directory"])
            directory.mkdir(exist_ok=True)
            hpo.freeze_manifest(args.calibrated_manifest, directory / "negatives.jsonl", params)
            child_args = argparse.Namespace(**vars(args))
            child_args.eval_file = validation_file
            command = hpo.training_command(child_args, params, directory)
            hpo.write_json(directory / "trial.json", {"number": number, "params": params, "command": command})
            row["state"] = "RUNNING"
            publish(args, records, "RUNNING")
            print(f"[budget] trial={number} params={params} remaining_hours={remaining(args) / 3600:.2f}\n"
                  f"{shlex.join(command)}", flush=True)
            try:
                if execute is None:
                    with (directory / "run.log").open("a") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(own_lock.fileno(), pipeline_lock.fileno()), timeout=remaining(args))
                else:
                    execute(command, directory)
                row["validation_em"] = hpo.validation_value(directory / "listed/summary.json", len(record["recurrent_questions"]))
            except (subprocess.TimeoutExpired, TimeoutError) as exc:
                row["state"] = "BUDGET_EXHAUSTED"
                row["last_error"] = str(exc)
                publish(args, records, "BUDGET_EXHAUSTED")
                return
            except Exception as exc:
                row.setdefault("attempt_errors", []).append(str(exc))
                publish(args, records, "INTERRUPTED")
                raise
            row["state"] = "COMPLETE"
            row["last_error"] = None
            publish(args, records, "RUNNING")
        publish(args, records, "COMPLETE")
        print(f"[budget] complete; selected configuration: {args.search_dir / 'best.json'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    verify_source(args)
    if args.dry_run:
        print(json.dumps({"max_total_epochs": 27, "max_trials": 6, "budget_hours": args.budget_hours,
                          "remaining_hours": remaining(args) / 3600, "test_evaluation": False,
                          "anchor": ANCHOR, "screen": SCREEN, "confirmation_epochs": 5,
                          "missing_prerequisites": [str(p) for p in hpo.required_files(args) if not p.is_file()]}, indent=2))
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Use configs/run_validation_budget.sh to launch inside tmux")
    run_search(args)


if __name__ == "__main__":
    main()
