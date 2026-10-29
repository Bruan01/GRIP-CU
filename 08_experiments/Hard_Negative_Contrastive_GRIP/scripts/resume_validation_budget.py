#!/usr/bin/env python3
"""Resume the frozen validation schedule after an infrastructure interruption.

Keep the original configuration and source hashes intact. Only the wall-clock
cutoff changes, in a separate immutable recovery record shared by restarts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import time

import search_validation_budget as budget


def recovery_record(args, hours):
    config_path = args.search_dir / "search_config.json"
    original = budget.hpo.read_json(config_path)
    if budget.frozen_config(args) != original:
        raise ValueError("Original configuration/source hashes changed; recovery refused")
    if not math.isfinite(hours) or not 0 < hours <= 24:
        raise ValueError("Recovery hours must be positive and <= 24")
    path = args.search_dir / "recovery_config.json"
    if path.exists():
        recovery = budget.hpo.read_json(path)
        if (recovery["original_config_sha256"] != budget.hpo.sha256(config_path)
                or recovery["hours"] != hours):
            raise ValueError("Recovery configuration differs; reuse its original hours")
    else:
        started = time.time()
        recovery = {
            "version": 1,
            "reason": "User authorized continuation after disk-full checkpoint failure",
            "original_config_sha256": budget.hpo.sha256(config_path),
            "original_deadline_utc": original["deadline_utc"],
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "deadline_utc": datetime.fromtimestamp(started + hours * 3600, timezone.utc).isoformat(),
            "hours": hours,
            "schedule_changed": False,
            "test_evaluation": False,
        }
    return original, recovery


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recovery_hours", type=float, default=18)
    extra, rest = parser.parse_known_args(argv)
    args = budget.parse_args(rest)
    budget.verify_source(args)
    if args.dry_run:
        _, recovery = recovery_record(args, extra.recovery_hours)
        print(json.dumps(recovery, indent=2))
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Use configs/run_validation_budget.sh with RECOVERY_HOURS inside tmux")
    # Serialize recovery creation and hold this lock until every child exits.
    with (args.search_dir / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original, recovery = recovery_record(args, extra.recovery_hours)
        budget.hpo.freeze_text(args.search_dir / "recovery_config.json",
                               json.dumps(recovery, sort_keys=True, indent=2) + "\n")
        cutoff = datetime.fromisoformat(recovery["deadline_utc"]).timestamp()
        # Reuse the original scheduler unchanged, with a separately recorded cutoff.
        # Source/config verification above happens before these two overrides.
        budget.frozen_config = lambda unused: original
        budget.deadline = lambda unused: cutoff
        print(f"[recovery] original configuration retained; deadline={recovery['deadline_utc']}", flush=True)
        budget.run_search(args)


if __name__ == "__main__":
    main()
