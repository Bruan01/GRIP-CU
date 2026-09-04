#!/usr/bin/env python3
"""Static audit: config, data, vocabulary, graph boundary, and checkpoint registry."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from entity_decoder.artifacts import import_d0_predictions
from entity_decoder.checkpoints import resolve_checkpoint_registry
from entity_decoder.config import load_config
from entity_decoder.io_utils import sha256_file, write_json
from entity_decoder.records import build_prompt, load_split
from entity_decoder.vocabulary import audit_answer_coverage, read_entities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase_a_decoder_smoke.json")
    ap.add_argument("--vocabulary", type=Path, default=ROOT / "artifacts/entities_train_graph.jsonl")
    ap.add_argument("--require-checkpoints", action="store_true")
    ap.add_argument("--checkpoint", action="append", dest="checkpoint_names")
    args = ap.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    entities = read_entities(args.vocabulary.resolve())
    splits = {name: load_split(REPO / config["data"][name], name) for name in ("train", "validation", "test")}
    for name, expected in config["data"]["expected_sizes"].items():
        if len(splits[name]) != int(expected):
            raise ValueError(f"{name} size changed")
    ids = {name: {row["task_id"] for row in rows} for name, rows in splits.items()}
    split_ids_disjoint = not (
        ids["train"] & ids["validation"]
        or ids["train"] & ids["test"]
        or ids["validation"] & ids["test"]
    )
    if not split_ids_disjoint:
        raise ValueError("task-id leakage across splits")
    graph_boundary = True
    for checkpoint in config["checkpoints"]:
        for split in ("validation", "test"):
            for row in splits[split]:
                prompt = build_prompt(row, checkpoint["prompt_protocol"])
                leaked = [
                    node for node in row["path_nodes"][1:]
                    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(node))}(?![A-Za-z0-9_])", prompt)
                ]
                if leaked:
                    graph_boundary = False
                    raise ValueError(f"prompt leaked non-query path nodes: {row['task_id']} {leaked}")
    checkpoint_entries = config["checkpoints"]
    if args.checkpoint_names:
        selected = set(args.checkpoint_names)
        checkpoint_entries = [entry for entry in checkpoint_entries if entry["name"] in selected]
        missing_selected = selected - {entry["name"] for entry in checkpoint_entries}
        if missing_selected:
            raise ValueError(f"unknown selected checkpoints: {sorted(missing_selected)}")
    checkpoints = resolve_checkpoint_registry(checkpoint_entries, REPO, args.require_checkpoints)
    d0_artifacts = {}
    initial_names = set(config["evaluation"]["initial_checkpoints"])
    for entry in config["checkpoints"]:
        if entry["name"] in initial_names:
            _, d0_artifacts[entry["name"]] = import_d0_predictions(
                entry["d0_predictions"]["validation"], REPO, splits["validation"]
            )
    coverages = {name: audit_answer_coverage(entities, rows) for name, rows in splits.items()}
    data_paths = {name: config["data"][name] for name in ("train", "validation", "test")}
    expected_sizes_match = all(
        len(splits[name]) == int(config["data"]["expected_sizes"][name])
        for name in ("train", "validation", "test")
    )
    expected_coverage = float(config["data"]["expected_test_answer_coverage"])
    audit = {
        "status": "READY_FOR_WSL_VALIDATION_D0_D1" if args.require_checkpoints else "STATIC_READY_D0_ARTIFACTS_VERIFIED_REMOTE_CHECKPOINTS_OPTIONAL",
        "config": str(config_path.relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "data": {
            "paths": data_paths,
            "train_graph": config["data"]["train_graph"],
            "vocabulary_source_split": config["data"]["vocabulary_source_split"],
        },
        "entity_count": len(entities),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "answer_coverage": coverages,
        "graph_free_prompt_boundary": graph_boundary,
        "checkpoints": checkpoints,
        "d0_artifacts": d0_artifacts,
        "decoder_registry": config["decoders"],
        "evaluation_splits": config["evaluation"]["splits"],
        "initial_checkpoints": config["evaluation"]["initial_checkpoints"],
        "official_phase_b_anchor_registered": bool(config.get("phase_b_official_anchor")),
        "invariants": {
            "train_only_vocabulary_source": config["data"]["vocabulary_source_split"] == "train",
            "expected_entity_count": len(entities) == int(config["data"]["expected_entity_count"]),
            "expected_split_sizes": expected_sizes_match,
            "split_task_ids_disjoint": split_ids_disjoint,
            "train_answer_coverage_complete": coverages["train"]["coverage"] == 1.0,
            "validation_answer_coverage_complete": coverages["validation"]["coverage"] == 1.0,
            "test_answer_coverage_expected": coverages["test"]["coverage"] == expected_coverage,
            "graph_free_prompt_boundary": graph_boundary,
            "checkpoint_registry_complete": len(checkpoints) == (len(args.checkpoint_names) if args.checkpoint_names else len(config["checkpoints"])),
            "d0_artifacts_verified": len(d0_artifacts) == len(initial_names),
            "validation_only_initial_split": config["evaluation"]["splits"] == ["validation"],
            "decoder_registry_complete": config["decoders"] == ["D0", "D1"],
            "official_phase_b_anchor_registered": bool(config.get("phase_b_official_anchor")),
        },
    }
    write_json(ROOT / "artifacts/setup_audit.json", audit)
    print(audit["status"])


if __name__ == "__main__":
    main()
