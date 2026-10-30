#!/usr/bin/env python3
"""Create a reproducible paired comparison for candidate-rerank runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def atomic_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--docs-baseline", type=float, default=0.8774)
    args = parser.parse_args()

    ours = load_jsonl(args.ours)
    baseline = load_jsonl(args.baseline)
    count = min(len(ours), len(baseline)) if args.count <= 0 else args.count
    if len(ours) < count or len(baseline) < count:
        raise ValueError(f"not enough rows for paired comparison: ours={len(ours)}, baseline={len(baseline)}, count={count}")

    ours = ours[:count]
    baseline = baseline[:count]
    for index, (ours_row, baseline_row) in enumerate(zip(ours, baseline)):
        if ours_row.get("dataset_index") != baseline_row.get("dataset_index"):
            raise ValueError(f"dataset index mismatch at paired row {index}")
        if ours_row.get("target") != baseline_row.get("target"):
            raise ValueError(f"target mismatch at paired row {index}")

    ours_correct = [bool(row.get("correct")) for row in ours]
    baseline_correct = [bool(row.get("correct")) for row in baseline]
    ours_em = sum(ours_correct) / count if count else None
    baseline_em = sum(baseline_correct) / count if count else None
    payload = {
        "format_version": 1,
        "status": "completed",
        "protocol": "same official candidate list, same NELL23K test prefix, teacher-forced candidate scoring",
        "ours": {"path": str(args.ours), "count": count, "em": ours_em},
        "baseline": {"path": str(args.baseline), "count": count, "em": baseline_em},
        "docs_baseline_em": args.docs_baseline,
        "ours_minus_baseline_em": ours_em - baseline_em if count else None,
        "ours_minus_docs_baseline_em": ours_em - args.docs_baseline if count else None,
        "paired": {
            "gains": sum(a and not b for a, b in zip(ours_correct, baseline_correct)),
            "regressions": sum((not a) and b for a, b in zip(ours_correct, baseline_correct)),
            "both_correct": sum(a and b for a, b in zip(ours_correct, baseline_correct)),
            "both_wrong": sum((not a) and (not b) for a, b in zip(ours_correct, baseline_correct)),
        },
    }
    atomic_dump(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
