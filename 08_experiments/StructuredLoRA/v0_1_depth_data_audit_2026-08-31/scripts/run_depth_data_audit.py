#!/usr/bin/env python3
"""Run the complete CPU-only StructuredLoRA v0.1 depth data audit."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from structured_lora_audit.depth import (  # noqa: E402
    distribution_rows,
    frequency_depth_rows,
    label_split,
    normalized_mutual_information,
    relation_depth_rows,
)
from structured_lora_audit.exact_hop import sample_exact_hop_tasks  # noqa: E402
from structured_lora_audit.graph import KnowledgeGraph  # noqa: E402
from structured_lora_audit.io_utils import (  # noqa: E402
    read_triples,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from structured_lora_audit.predictions import analyze_prediction_files  # noqa: E402
from structured_lora_audit.report import evaluate_gates, render_markdown  # noqa: E402


def resolve_from_experiment(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (EXPERIMENT_ROOT / path).resolve()


def assign_exact_hop_splits(tasks: list[dict], ratios: dict[str, float]) -> list[dict]:
    names = ["train", "validation", "test"]
    if set(ratios) != set(names):
        raise ValueError(f"exact_hop_split must contain exactly {names}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("exact_hop_split ratios must sum to 1")
    by_depth: dict[int, list[dict]] = {}
    for task in tasks:
        by_depth.setdefault(int(task["depth_label"]), []).append(task)
    output: list[dict] = []
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        train_end = int(len(rows) * ratios["train"])
        validation_end = train_end + int(len(rows) * ratios["validation"])
        for index, task in enumerate(rows):
            if index < train_end:
                split = "train"
            elif index < validation_end:
                split = "validation"
            else:
                split = "test"
            output.append({**task, "split": split})
    return output


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["dataset", "data_dir", "output_dir", "max_depth", "exact_hop_per_depth", "seed"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "depth_audit_nell23k.json",
    )
    args = parser.parse_args()
    started = time.time()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_dir = resolve_from_experiment(config["data_dir"])
    output_dir = resolve_from_experiment(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_depth = int(config["max_depth"])
    seed = int(config["seed"])

    source_paths = {split: data_dir / f"{split}.txt" for split in ("train", "valid", "test")}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = {split: sha256_file(path) for split, path in source_paths.items()}
    expected_hashes = config.get("expected_source_sha256") or {}
    mismatches = {
        split: {"expected": expected_hashes.get(split), "actual": actual}
        for split, actual in source_hashes.items()
        if expected_hashes.get(split) not in (None, actual)
    }
    if mismatches:
        raise ValueError(f"NELL23K source hash mismatch: {mismatches}")
    triples = {split: read_triples(path) for split, path in source_paths.items()}
    graph = KnowledgeGraph(triples["train"])
    train_set = set(triples["train"])
    train_relations = set(graph.relation_count)
    split_integrity = {}
    for split, rows in triples.items():
        split_integrity[split] = {
            "rows": len(rows),
            "unique_triples": len(set(rows)),
            "duplicate_rows": len(rows) - len(set(rows)),
            "self_loops": sum(head == tail for head, _relation, tail in rows),
            "exact_train_overlap": (
                None if split == "train" else sum(row in train_set for row in rows)
            ),
            "unseen_relation_rows": sum(relation not in train_relations for _head, relation, _tail in rows),
            "unseen_endpoint_rows": sum(
                head not in graph.nodes or tail not in graph.nodes for head, _relation, tail in rows
            ),
        }

    labels: list[dict] = []
    for split in ("train", "valid", "test"):
        labels.extend(
            label_split(
                graph,
                triples[split],
                split=split,
                max_depth=max_depth,
                leave_one_out=(split == "train"),
            )
        )

    depth_dir = output_dir / "depth_labels"
    report_dir = output_dir / "reports"
    exact_dir = output_dir / "exact_hop"
    label_path = depth_dir / "nell23k_support_depth.jsonl.gz"
    write_jsonl(label_path, labels)

    distributions = distribution_rows(labels)
    relation_depth = relation_depth_rows(labels, mode="undirected")
    frequency_depth = frequency_depth_rows(labels, mode="undirected")
    write_csv(report_dir / "depth_distribution.csv", distributions)
    write_csv(report_dir / "relation_depth_distribution.csv", relation_depth)
    write_csv(report_dir / "frequency_depth_distribution.csv", frequency_depth)
    write_json(report_dir / "split_integrity.json", split_integrity)

    exact_tasks, exact_metadata = sample_exact_hop_tasks(
        graph,
        per_depth=int(config["exact_hop_per_depth"]),
        max_depth=max_depth,
        seed=seed,
        max_attempts_per_depth=int(config.get("max_attempts_per_depth", 500_000)),
    )
    exact_tasks = assign_exact_hop_splits(exact_tasks, config["exact_hop_split"])
    exact_metadata["split_counts"] = dict(Counter(task["split"] for task in exact_tasks))
    exact_metadata["depth_split_counts"] = {
        f"d{depth}:{split}": sum(
            task["depth_label"] == depth and task["split"] == split for task in exact_tasks
        )
        for depth in range(1, max_depth + 1)
        for split in ("train", "validation", "test")
    }
    write_jsonl(exact_dir / "nell23k_exact_hop_tasks.jsonl", exact_tasks)
    for split in ("train", "validation", "test"):
        write_jsonl(exact_dir / f"nell23k_exact_hop_{split}.jsonl", (t for t in exact_tasks if t["split"] == split))
    write_json(exact_dir / "generation_metadata.json", exact_metadata)

    prediction_paths = [resolve_from_experiment(value) for value in config.get("prediction_files", [])]
    existing_prediction_paths = [path for path in prediction_paths if path.is_file()]
    prediction_metrics, prediction_audit = analyze_prediction_files(
        existing_prediction_paths,
        labels,
        mode="undirected",
    )
    prediction_audit["files_found"] = [
        value for value, path in zip(config.get("prediction_files", []), prediction_paths) if path.is_file()
    ]
    prediction_audit["files_missing"] = [
        value for value, path in zip(config.get("prediction_files", []), prediction_paths) if not path.is_file()
    ]
    write_csv(report_dir / "baseline_accuracy_by_depth.csv", prediction_metrics)
    write_json(report_dir / "prediction_join_audit.json", prediction_audit)

    nmi = normalized_mutual_information(labels, split="test", mode="undirected")
    expected_exact = int(config["exact_hop_per_depth"]) * max_depth
    decision = evaluate_gates(
        distributions,
        prediction_metrics,
        exact_hop_total=len(exact_tasks),
        expected_exact_hop_total=expected_exact,
        relation_depth_nmi=nmi,
    )
    summary = {
        "format_version": 1,
        "experiment": "StructuredLoRA v0.1 depth data audit",
        "config": config,
        "config_path": str(config_path.relative_to(EXPERIMENT_ROOT)),
        "dataset": {
            "name": config["dataset"],
            "split_sizes": {split: len(rows) for split, rows in triples.items()},
            "train_nodes": len(graph.nodes),
            "train_edges": len(graph.triples),
            "train_relations": len(graph.relation_count),
            "source_sha256": source_hashes,
            "split_integrity": split_integrity,
        },
        "label_artifact": {
            "path": str(label_path.relative_to(EXPERIMENT_ROOT)),
            "rows": len(labels),
            "sha256": sha256_file(label_path),
            "censoring": (
                f"A null bounded distance means no path was found within {max_depth} steps; "
                "it does not prove global unreachability."
            ),
        },
        "exact_hop": {**exact_metadata, "sha256": sha256_file(exact_dir / "nell23k_exact_hop_tasks.jsonl")},
        "predictions": prediction_audit,
        "decision": decision,
    }
    write_json(report_dir / "summary.json", summary)
    (report_dir / "REPORT.md").write_text(render_markdown(summary), encoding="utf-8")
    elapsed_seconds = round(time.time() - started, 3)

    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "decision": decision["decision"],
        "label_rows": len(labels),
        "exact_hop_tasks": len(exact_tasks),
        "prediction_join_rate": prediction_audit["join_rate"],
        "elapsed_seconds": elapsed_seconds,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
