#!/usr/bin/env python3
"""Build the original CLEGR graph-level 20/80 split from the local 500 graphs.

The released GRIP preprocessing keeps 20% of CLEGR graphs as training
references and evaluates on the remaining 80% (400 graphs).  The benchmark
packaging used by this workspace additionally creates a 300/100/100 split for
development, so the paper runner must not silently treat its 100-graph test
file as the paper test set.  This utility reconstructs a deterministic
20/80 split from the union of the packaged records without touching the
existing benchmark files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = args.data_dir / f"processed_{split}.json"
        for row in load_jsonl(path):
            graph_id = str(row.get("id") or "")
            if not graph_id:
                raise ValueError(f"{path}: graph record has no id")
            if graph_id in rows:
                raise ValueError(f"duplicate graph id {graph_id!r}")
            rows[graph_id] = row

    if len(rows) != 500:
        raise ValueError(
            f"the original CLEGR protocol expects the local source to contain "
            f"500 graphs, found {len(rows)}"
        )

    graph_ids = sorted(rows)
    random.Random(args.seed).shuffle(graph_ids)
    train_ids = graph_ids[:100]
    test_ids = graph_ids[100:]
    train_rows = [rows[graph_id] for graph_id in sorted(train_ids)]
    test_rows = [rows[graph_id] for graph_id in sorted(test_ids)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "processed_train.json"
    test_path = args.output_dir / "processed_test.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)
    manifest = {
        "format_version": 1,
        "dataset": "clegr_reasoning",
        "protocol": "original_grip_20_percent_train_80_percent_test",
        "seed": args.seed,
        "source_dir": str(args.data_dir.resolve()),
        "source_graphs": len(rows),
        "train_graphs": len(train_rows),
        "test_graphs": len(test_rows),
        "train_file": str(train_path.resolve()),
        "test_file": str(test_path.resolve()),
        "sha256": {
            "train_file": sha256(train_path),
            "test_file": sha256(test_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
