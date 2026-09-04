#!/usr/bin/env python3
"""Validate v0.1.5 data/config without loading a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from priority_distill.candidates import build_candidate_pool_for_query, build_candidate_pools
from priority_distill.config import load_config
from priority_distill.io_utils import sha256_file, write_json
from priority_distill.records import build_evaluation_prompt, load_splits
from priority_distill.supervision import build_method_supervision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/direct_answer_only.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/setup_audit.json")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    splits, paths, split_audit = load_splits(REPO, config["data"])
    seed = int(config["candidates"]["seed"])
    pools = build_candidate_pools(splits["train"], int(config["candidates"]["distractor_count"]), seed)
    oracle = build_method_supervision(splits["train"], pools, "oracle_priority_equal_token", seed)
    all_paths = build_method_supervision(splits["train"], pools, "all_paths_equal_token", seed)
    anti_copy = build_method_supervision(splits["train"], pools, "all_paths_terminal_masked_equal_token", seed)
    explicit = build_method_supervision(splits["train"], pools, "explicit_path_selection", seed)
    explicit_anti_copy = build_method_supervision(splits["train"], pools, "explicit_path_selection_terminal_masked", seed)
    trace = build_method_supervision(splits["train"], pools, "graph_free_trace_terminal_masked", seed)
    query_pool = build_candidate_pool_for_query(splits["validation"][0], splits["train"], int(config["candidates"]["distractor_count"]), seed)
    prompts = [build_evaluation_prompt(row["text"]) for row in splits["test"]]
    invariants = {
        "candidate_pools_cover_train": len(pools) == len(splits["train"]),
        "four_candidates_per_pool": all(len(pool) == 4 for pool in pools.values()),
        "one_gold_per_pool": all(sum(candidate["is_gold"] for candidate in pool) == 1 for pool in pools.values()),
        "distractors_train_only": all(candidate["source_split"] == "train" for pool in pools.values() for candidate in pool),
        "oracle_has_one_gold_path": all(row["candidate_count"] == 1 and row["gold_position"] == 0 for row in oracle),
        "all_paths_has_four_unlabeled_paths": all(row["candidate_count"] == 4 and row["gold_position"] >= 0 for row in all_paths),
        "anti_copy_has_four_paths": all(row["candidate_count"] == 4 for row in anti_copy),
        "anti_copy_hides_terminal_marker": all(
            "<MASKED_TERMINAL>" in row["prompt"] and "Candidate paths (terminal entity hidden):" in row["prompt"]
            for row in anti_copy
        ),
        "anti_copy_lists_four_unassociated_endpoints": all(row["prompt"].count("- ") >= 4 for row in anti_copy),
        "explicit_has_two_line_targets": all(row["target_text"].startswith("Selected path: ") and "\nAnswer: " in row["target_text"] for row in explicit),
        "explicit_selection_has_four_paths": all(row["candidate_count"] == 4 and 1 <= row["selection_target"] <= 4 for row in explicit),
        "explicit_anti_copy_masks_terminals": all("<MASKED_TERMINAL>" in row["prompt"] for row in explicit_anti_copy),
        "trace_has_graph_free_prompts": all(
            "Candidate evidence:" not in row["prompt"] and "Candidate paths" not in row["prompt"]
            for row in trace
        ),
        "trace_targets_have_trace_and_answer": all(
            row["target_text"].startswith("Trace: ") and "\nAnswer: " in row["target_text"]
            for row in trace
        ),
        "trace_targets_mask_terminal": all(
            row["target_text"].splitlines()[0].split(" -> ")[-1] == "<MASKED_TERMINAL>"
            for row in trace
        ),
        "trace_depth_labels_and_intermediates": all(
            row["intermediate_node_count"] == max(0, int(row["depth_label"]) - 1)
            for row in trace
        ),
        "query_pool_is_gold_plus_train_distractors": len(query_pool) == 4 and all(candidate["source_split"] == "train" for candidate in query_pool[1:]),
        "evaluation_has_no_candidate_evidence": all("Candidate evidence:" not in prompt for prompt in prompts),
        "controlled_protocol_is_valid": config["protocol"] in {
            "direct_answer_only",
            "oracle_two_stage",
            "candidate_selection_two_stage",
            "candidate_selection_anti_copy",
            "candidate_selection_anti_copy_replay",
            "candidate_index_two_stage",
            "candidate_index_anti_copy",
            "candidate_index_anti_copy_replay",
            "graph_free_trace",
            "candidate_index_anti_copy_bridge",
        },
    }
    if not all(invariants.values()):
        raise AssertionError(invariants)
    payload = {
        "status": "READY_FOR_WSL_GPU",
        "protocol": config["protocol"],
        "config": str(config_path.relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "data": {
            **split_audit,
            "paths": {split: str(path.relative_to(REPO)) for split, path in paths.items()},
            "sha256": {split: sha256_file(path) for split, path in paths.items()},
        },
        "oracle_rows": len(oracle),
        "all_paths_rows": len(all_paths),
        "anti_copy_rows": len(anti_copy),
        "explicit_rows": len(explicit),
        "explicit_anti_copy_rows": len(explicit_anti_copy),
        "trace_rows": len(trace),
        "invariants": invariants,
    }
    write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"status": payload["status"], "protocol": payload["protocol"], "train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
