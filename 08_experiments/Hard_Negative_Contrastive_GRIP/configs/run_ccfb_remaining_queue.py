#!/usr/bin/env python3
"""Run the frozen CCFB follow-up seeds serially after WN18RR seed2027.

Start this controller inside a detached tmux session. It waits for the active
WN batch, then uses the existing per-batch launcher and all of its input,
source, GPU and disk checks. Low disk space pauses the queue; a failed or
interrupted experiment stops it without retrying or deleting evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HNG = Path(__file__).resolve().parents[1]
RUNS = HNG / "results/runs/ccfb"
QUEUE = RUNS / "queue_20261108"
STATE = QUEUE / "status.json"
CURRENT = ("wn18rr", 2027)
MIN_FREE_GIB = 45
GIB = 1024**3

# Finish the NELL three-seed comparison first, then the WN high-variance study.
# The remaining two three-seed datasets follow. All nine jobs are the fixed
# matrix from configs/ccfb_sprint_matrix.json; no mechanism search is queued.
JOBS = (
    ("nell23k", 2027),
    ("nell23k", 2028),
    ("wn18rr", 2028),
    ("wn18rr", 2029),
    ("wn18rr", 2030),
    ("fb15k237", 2027),
    ("fb15k237", 2028),
    ("codexm", 2027),
    ("codexm", 2028),
)


def name(batch: tuple[str, int]) -> str:
    return f"{batch[0]}:seed{batch[1]}"


def run_root(batch: tuple[str, int]) -> Path:
    return RUNS / batch[0] / f"seed{batch[1]}"


def run_status(batch: tuple[str, int]) -> str | None:
    path = run_root(batch) / "run_status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def free_gib() -> float:
    return shutil.disk_usage(RUNS).free / GIB


def note(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def save_state(state: str, current: tuple[str, int] | None, completed: list[str],
               detail: str = "") -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "current": None if current is None else name(current),
        "completed": completed,
        "pending": [name(job) for job in JOBS if name(job) not in completed],
        "free_gib": round(free_gib(), 3),
        "minimum_free_gib": MIN_FREE_GIB,
        "detail": detail,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = STATE.with_name(f"status.json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, STATE)


def stop(reason: str, batch: tuple[str, int] | None, completed: list[str]) -> int:
    note("STOPPED: " + reason)
    save_state("STOPPED", batch, completed, reason)
    return 1


def wait_for_current(completed: list[str]) -> bool:
    session = "ccfb-wn18rr-2027"
    while True:
        status = run_status(CURRENT)
        if status == "COMPLETE":
            if not (run_root(CURRENT) / "comparison.json").is_file():
                stop("WN18RR seed2027 says COMPLETE but comparison.json is missing",
                     CURRENT, completed)
                return False
            if not tmux_session_exists(session):
                note("WN18RR seed2027 completed; starting fixed follow-ups")
                return True
        elif status != "RUNNING" or not tmux_session_exists(session):
            stop(f"WN18RR seed2027 is {status!r}; inspect its run before continuing",
                 CURRENT, completed)
            return False
        save_state("WAITING_CURRENT", CURRENT, completed,
                   "Waiting for WN18RR seed2027 COMPLETE and its tmux session to exit")
        time.sleep(60)


def execute() -> int:
    if not os.environ.get("TMUX"):
        print("Start this queue in tmux; it must survive the local SSH session.",
              file=sys.stderr)
        return 2
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another CCFB follow-up queue owns queue.lock.", file=sys.stderr)
            return 2
        completed: list[str] = []
        note("Queue started: " + ", ".join(map(name, JOBS)))
        if not wait_for_current(completed):
            return 1
        for batch in JOBS:
            status = run_status(batch)
            if status == "COMPLETE":
                if not (run_root(batch) / "comparison.json").is_file():
                    return stop(f"{name(batch)} has no comparison.json", batch, completed)
                completed.append(name(batch))
                note(f"Reusing completed {name(batch)}")
                continue
            if status is not None:
                return stop(f"{name(batch)} is {status!r}; inspect it before retrying",
                            batch, completed)
            while free_gib() < MIN_FREE_GIB:
                detail = (f"{name(batch)} needs at least {MIN_FREE_GIB} GiB free; "
                          "waiting for separately reviewed storage action")
                save_state("WAITING_SPACE", batch, completed, detail)
                note(f"WAITING_SPACE: {detail}; free={free_gib():.2f} GiB")
                time.sleep(300)
            save_state("RUNNING", batch, completed,
                       "Batch launcher will recheck hashes, GPU and measured disk budget")
            note(f"Starting {name(batch)}; free={free_gib():.2f} GiB")
            env = os.environ.copy()
            for override in ("ROOT", "SHARED_ROOT", "PYTHON", "PYTHONPATH"):
                env.pop(override, None)
            env.update(DATASET=batch[0], SEED=str(batch[1]),
                       TMUX_SESSION=f"ccfb-{batch[0]}-{batch[1]}",
                       DRY_RUN="0", CCFB_NELL_COMPACT="0")
            result = subprocess.run(["bash", str(HNG / "configs/run_ccfb_followup.sh")],
                                    cwd=HNG, env=env, check=False)
            if result.returncode != 0 or run_status(batch) != "COMPLETE":
                return stop(f"{name(batch)} launcher exited {result.returncode}; "
                            f"run_status={run_status(batch)!r}", batch, completed)
            if not (run_root(batch) / "comparison.json").is_file():
                return stop(f"{name(batch)} completed without comparison.json",
                            batch, completed)
            completed.append(name(batch))
            save_state("BETWEEN_BATCHES", None, completed)
            note(f"Completed {name(batch)}; free={free_gib():.2f} GiB")
        save_state("COMPLETE", None, completed, "All nine fixed follow-ups completed")
        note("All nine fixed follow-ups completed")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-plan", action="store_true",
                        help="Print the fixed order without starting or waiting")
    args = parser.parse_args()
    if args.show_plan:
        print(json.dumps({"wait_for": name(CURRENT), "jobs": list(map(name, JOBS)),
                          "minimum_free_gib": MIN_FREE_GIB}, indent=2))
        return 0
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
