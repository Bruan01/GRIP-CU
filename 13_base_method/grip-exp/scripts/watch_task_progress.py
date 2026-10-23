#!/usr/bin/env python3
"""Watch durable GRIP task-generation progress snapshots.

Example:
    python scripts/watch_task_progress.py \
      --cache-dir /mnt/e/GRIP-CU/artifacts/grip_paper_tasks_siliconflow_qwen7b/clegr_reasoning
"""

import argparse
import json
import time
from pathlib import Path


def _bar(percent: float, width: int = 32) -> str:
    bounded = min(100.0, max(0.0, percent))
    filled = int(width * bounded / 100.0)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _load_progress(path: Path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def render(cache_dir: Path) -> str:
    rows = []
    for path in sorted(cache_dir.glob("*.progress.json")):
        payload = _load_progress(path)
        if payload is not None:
            rows.append(payload)

    lines = [
        f"GRIP task construction: {cache_dir}",
        time.strftime("updated %Y-%m-%d %H:%M:%S"),
        "",
    ]
    if not rows:
        lines.append("Waiting for the first task-generation progress snapshot...")
        return "\n".join(lines)

    for row in rows:
        completed = int(row.get("completed", 0))
        total = int(row.get("total", 0))
        percent = float(row.get("percent", 0.0))
        rate = float(row.get("rate_prompt_per_second", 0.0))
        status = row.get("status", "unknown")
        eta = _duration(row.get("eta_seconds"))
        lines.append(
            f"{row.get('stage', 'unknown'):22s} {_bar(percent)} "
            f"{completed:>6d}/{total:<6d} {percent:6.2f}% "
            f"{rate:6.2f} prompt/s ETA {eta:>8s} [{status}]"
        )
        if row.get("error"):
            lines.append(f"  error: {row['error']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = parser.parse_args()

    while True:
        output = render(args.cache_dir)
        if not args.once:
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if args.once:
            return
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
