#!/usr/bin/env python3
"""Render a compact curve table from diagnostic_metrics.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from priority_distill.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    rows = [json.loads(line) for line in (run_dir / "diagnostic_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    table = []
    for row in rows:
        for metric_name, metric in row["metrics"].items():
            table.append({
                "checkpoint": row["checkpoint"],
                "stage": row["stage"],
                "epoch": row["epoch"],
                "metric": metric_name,
                "accuracy": metric["accuracy"],
                "deep_3_4_accuracy": metric["deep_3_4_accuracy"],
                "correct": metric["correct"],
                "count": metric["count"],
            })
    write_json(run_dir / "diagnostic_curve.json", table)
    lines = [
        "# Oracle Stage-2 diagnostic curve",
        "",
        "`graph_free` is the actual intended evaluation. `oracle_evidence` is diagnostic-only and exposes the row's gold path.",
        "",
        "| checkpoint | condition/split | accuracy | deep 3/4 |",
        "|---|---|---:|---:|",
    ]
    for row in table:
        lines.append(f"| {row['checkpoint']} | {row['metric']} | {row['accuracy']:.4f} | {row['deep_3_4_accuracy']:.4f} |")
    (run_dir / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'DIAGNOSTIC_REPORT.md'}")


if __name__ == "__main__":
    main()
