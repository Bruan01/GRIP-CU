#!/usr/bin/env python3
"""Build deterministic train-only PriorityDistill supervision artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from priority_distill.candidates import build_candidate_pools, relation_overlap
from priority_distill.config import load_config
from priority_distill.io_utils import sha256_file, write_json, write_jsonl
from priority_distill.records import load_splits
from priority_distill.supervision import METHODS, build_method_supervision, supervision_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_priority_smoke.json")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "artifacts/supervision")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = load_config(config_path)
    splits, paths, split_audit = load_splits(REPO_ROOT, config["data"])
    train_rows = splits["train"]
    seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(train_rows, int(config["candidates"]["distractor_count"]), seed)
    candidate_rows = []
    overlaps = []
    for task_id in sorted(pools):
        gold = next(candidate for candidate in pools[task_id] if candidate["is_gold"])
        for candidate in pools[task_id]:
            overlap = relation_overlap(gold["path_relations"], candidate["path_relations"])
            candidate_rows.append({"query_task_id": task_id, "relation_overlap": overlap, **candidate})
            if not candidate["is_gold"]:
                overlaps.append(overlap)
    candidate_path = output_dir / "candidate_pools.jsonl"
    write_jsonl(candidate_path, candidate_rows)
    method_rows = {method: build_method_supervision(train_rows, pools, method, seed) for method in METHODS}
    files = {"candidate_pools": candidate_path}
    for method, rows in method_rows.items():
        path = output_dir / f"{method}.jsonl"
        write_jsonl(path, rows)
        files[method] = path
    audit = {
        "status": "SUPERVISION_READY",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "source_data": {
            "paths": {split: str(path.relative_to(REPO_ROOT)) for split, path in paths.items()},
            "sha256": {split: sha256_file(path) for split, path in paths.items()},
            **split_audit,
        },
        "candidate_policy": config["candidates"],
        "candidate_pool_count": len(pools),
        "candidate_rows": len(candidate_rows),
        "distractor_relation_overlap_mean": sum(overlaps) / len(overlaps),
        "methods": supervision_audit(method_rows),
        "artifacts": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "prompt_boundary": {
            "training_may_use_paths": True,
            "validation_and_test_use_paths": False,
            "inference_graph_access": False,
        },
    }
    write_json(output_dir / "audit.json", audit)
    print(json.dumps({"status": audit["status"], "train_queries": len(train_rows), "candidate_rows": len(candidate_rows), "methods": len(method_rows), "output": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
