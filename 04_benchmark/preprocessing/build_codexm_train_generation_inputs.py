#!/usr/bin/env python3
"""Build CoDEx-M GRIP task-generation inputs from train-only evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from prepare_benchmark_data import (
    ROOT,
    atomic_json_dump,
    atomic_jsonl_dump,
    codexm_entity_surface,
    codexm_train_reference_record,
    file_record,
    parse_codexm_triples,
)


DATASET_DIR = ROOT / "04_benchmark" / "data" / "codexm"


def load_one_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"expected one graph record in {path}, got {len(payload)}")
        return payload[0]
    return payload


def main() -> None:
    raw_dir = DATASET_DIR / "raw"
    processed_dir = DATASET_DIR / "processed"
    tasks_dir = DATASET_DIR / "tasks"
    train_triples = parse_codexm_triples((raw_dir / "train.txt").read_bytes(), "train.txt")
    entity_metadata = json.loads((raw_dir / "entity2text.json").read_text(encoding="utf-8"))
    relation_labels = json.loads((raw_dir / "rel2text.json").read_text(encoding="utf-8"))
    relation_order = list(dict.fromkeys(relation for _, relation, _ in train_triples))
    entity_text = {
        entity: codexm_entity_surface(entity, metadata)
        for entity, metadata in entity_metadata.items()
    }
    missing = sorted(
        {entity for h, _, t in train_triples for entity in (h, t)}.difference(entity_text)
    )
    if missing:
        raise ValueError(f"train entities missing English metadata: {len(missing)}")
    train_entity_order = list(
        dict.fromkeys(entity for head, _, tail in train_triples for entity in (head, tail))
    )
    entity_text_tsv = raw_dir / "entity2text.txt"
    if any("\t" in entity_text[entity] or "\n" in entity_text[entity] for entity in train_entity_order):
        raise ValueError("train entity surfaces must fit the tab-separated mapping format")
    temporary_entity_text_tsv = entity_text_tsv.with_name(entity_text_tsv.name + ".tmp")
    temporary_entity_text_tsv.write_text(
        "".join(f"{entity}\t{entity_text[entity]}\n" for entity in train_entity_order),
        encoding="utf-8",
    )
    temporary_entity_text_tsv.replace(entity_text_tsv)
    if len(relation_order) != 51 or len(train_triples) != 185584:
        raise ValueError(
            f"unexpected CoDEx-M train inventory: triples={len(train_triples)}, "
            f"relations={len(relation_order)}"
        )

    train_reference = codexm_train_reference_record(
        train_triples,
        entity_text=entity_text,
        relation_labels=relation_labels,
        relation_order=relation_order,
        seed=2026,
    )
    train_reference_path = processed_dir / "processed_train_reference.json"
    atomic_jsonl_dump(train_reference_path, [train_reference])

    # The evaluation records carry the same train-only graph. Keep only that
    # graph here so task generation never loads held-out QA rows as input.
    graph_source = load_one_record(processed_dir / "processed_val.json")
    graph = graph_source["graph"]
    if len(graph.get("edge_list", [])) != len(train_triples):
        raise ValueError("processed graph edge count does not match raw/train.txt")
    train_graph_path = processed_dir / "processed_train_graph.json"
    atomic_jsonl_dump(
        train_graph_path,
        [{"title": "codexm", "graph": graph, "id": "0", "source_split": "train_graph"}],
    )

    plan_path = tasks_dir / "generation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "status": "pending_inference",
            "input_split": "train",
            "reference_split": "train",
            "input_file": "processed/processed_train_graph.json",
            "reference_file": "processed/processed_train_reference.json",
            "task_generator_backend": "local_vllm",
            "task_generator_batch_size": 8,
            "task_generator_gpu_memory_utilization": 0.82,
            "task_generator_model_name": "qwen-7b",
            "task_generator_model_cache_dir": "/mnt/e/GRIP-CU/model_cache",
            "task_generator_local_files_only": True,
            "task_generator_tokenize_max_length": 10000,
            "tokenize_max_length": 4096,
            "task_gen_max_length": 1000,
            "num_context_qa": 8000,
            "num_reason_qa": 2000,
            "num_summarization": 6000,
            "involve_qa_epochs": 5,
            "sample_node_attribute_task": False,
            "repharse_context_qa": False,
            "context_upsampling": False,
            "format_as_instruction": True,
            "seed": 2026,
            "cache_directory": "tasks/cache_vllm_qwen25_7b_b8_gpu82_20261026",
            "run_state_file": "tasks/cache_vllm_qwen25_7b_b8_gpu82_20261026/run_state.json",
            "note": (
                "Input is a graph-only record containing the raw/train.txt graph. K-shot examples "
                "come only from raw/train.txt. Valid/test QA rows are not passed to task generation."
            ),
        }
    )
    atomic_json_dump(plan_path, plan)

    manifest_path = DATASET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("generated_files", {})
    manifest["generated_files"]["processed/processed_train_graph.json"] = {
        **file_record(train_graph_path),
        "record_count": 1,
        "edge_count": len(train_triples),
        "source_split": "raw/train.txt",
    }
    manifest["generated_files"]["processed/processed_train_reference.json"] = {
        **file_record(train_reference_path),
        "record_count": 1,
        "question_count": len(train_reference["questions"]),
        "source_split": "raw/train.txt",
    }
    manifest["generated_files"]["raw/entity2text.txt"] = {
        **file_record(entity_text_tsv, line_count=len(train_entity_order)),
        "source_split": "raw/train.txt",
    }
    manifest["generated_files"]["tasks/generation_plan.json"] = file_record(plan_path)
    manifest.setdefault("protocol", {})["task_generation_reference"] = {
        "split": "train",
        "source": "raw/train.txt",
        "question_count": len(train_reference["questions"]),
        "purpose": "eight-shot examples for synthetic reasoning QA generation",
        "valid_test_examples_included": False,
    }
    atomic_json_dump(manifest_path, manifest)

    print(
        json.dumps(
            {
                "train_triples": len(train_triples),
                "relations": len(relation_order),
                "reference_questions": len(train_reference["questions"]),
                "train_entity_text_rows": len(train_entity_order),
                "graph_edges": len(graph.get("edge_list", [])),
                "valid_test_rows_used_as_examples": 0,
                "train_graph_sha256": file_record(train_graph_path)["sha256"],
                "train_reference_sha256": file_record(train_reference_path)["sha256"],
                "generation_plan": plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
