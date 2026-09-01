#!/usr/bin/env python3
"""Validate the registered design without importing torch or downloading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from priority_distill.candidates import build_candidate_pools
from priority_distill.config import load_config
from priority_distill.io_utils import sha256_file, write_json
from priority_distill.records import build_evaluation_prompt, load_splits
from priority_distill.supervision import METHODS, build_method_supervision, supervision_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/oracle_priority_smoke.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "artifacts/setup_audit.json")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    splits, paths, split_audit = load_splits(REPO_ROOT, config["data"])
    seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(splits["train"], int(config["candidates"]["distractor_count"]), seed)
    method_rows = {method: build_method_supervision(splits["train"], pools, method, seed) for method in METHODS}
    all_paths_positions = {row["gold_position"] for row in method_rows["all_paths_equal_token"]}
    evaluation_prompts = [build_evaluation_prompt(row["text"]) for row in splits["test"]]
    invariants = {
        "methods_match_registration": tuple(config["methods"]) == METHODS,
        "candidate_pools_cover_train": len(pools) == len(splits["train"]),
        "four_candidates_per_pool": all(len(pool) == 4 for pool in pools.values()),
        "one_gold_per_pool": all(sum(candidate["is_gold"] for candidate in pool) == 1 for pool in pools.values()),
        "distractors_train_only": all(candidate["source_split"] == "train" for pool in pools.values() for candidate in pool),
        "distractors_same_depth": all(len({candidate["depth_label"] for candidate in pool}) == 1 for pool in pools.values()),
        "all_paths_gold_position_not_constant": len(all_paths_positions) > 1,
        "random_path_never_gold": all(row["gold_position"] == -1 and row["selected_source_task_ids"] != [row["task_id"]] for row in method_rows["random_path_equal_token"]),
        "oracle_has_one_gold_path": all(row["candidate_count"] == 1 and row["gold_position"] == 0 for row in method_rows["oracle_priority_equal_token"]),
        "evaluation_has_no_candidate_evidence": all("Candidate evidence:" not in prompt for prompt in evaluation_prompts),
        "same_target_modules_as_grip": config["lora"]["target_modules"] == ["down_proj", "up_proj", "gate_proj"],
    }
    if not all(invariants.values()):
        raise AssertionError(invariants)
    payload = {
        "status": "READY_FOR_WSL_GPU",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "data": {**split_audit, "paths": {split: str(path.relative_to(REPO_ROOT)) for split, path in paths.items()}, "sha256": {split: sha256_file(path) for split, path in paths.items()}},
        "supervision": supervision_audit(method_rows),
        "all_paths_gold_positions": sorted(all_paths_positions),
        "invariants": invariants,
    }
    write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"status": payload["status"], "train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]), "methods": len(METHODS), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
