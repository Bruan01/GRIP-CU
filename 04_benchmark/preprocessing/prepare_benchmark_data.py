#!/usr/bin/env python3
"""Prepare the benchmark datasets used by the GRIP-CU experiments.

The script is deliberately data-first and idempotent:

* NELL23K and FB15K237-2 are validated in their existing benchmark layout.
* WN18RR is copied from the already reproduced GRIP asset and gets a
  deterministic task package.
* CoDEx-M is downloaded from a pinned upstream commit, converted to the
  relation-prediction schema, and keeps its verified triple-classification
  negatives separate from relation candidates.
* CLEGR-Facts and CLEGR-Reasoning are converted from the local PyTorch
  artifacts with memory mapping.  The source tensors are not copied into the
  benchmark directory; the processed JSONL records are self-contained and a
  source manifest preserves provenance.

Run from the repository root with the mandated guardenv interpreter:

    /home/kieran/miniconda3/envs/guardenv/bin/python \
        04_benchmark/preprocessing/prepare_benchmark_data.py

The generated dataset files are intentionally ignored by Git.  This script
and the accompanying documentation are the reproducible part of the data
preparation step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import types
from urllib.request import Request, urlopen
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "04_benchmark" / "data"
WN_SOURCE = ROOT / "13_base_method" / "grip-exp" / "outputs" / "data" / "wn18rr"
CLEGR_SOURCE = ROOT / "13_base_method" / "grip-exp" / "outputs" / "data" / "clegr" / "datasets_for_hf"
CODEXM_COMMIT = "3132e426c2a6b643b70bad679905a3a6270be440"
CODEXM_UPSTREAM_FILES = {
    "train.txt": ("data/triples/codex-m/train.txt", "3aed27810ca6fbdee5597e1142603b277fad3fd2"),
    "valid.txt": ("data/triples/codex-m/valid.txt", "b5648a7daadf276f2da3552c0a91265e26091753"),
    "test.txt": ("data/triples/codex-m/test.txt", "cf9416e164aef8aa55eb3f645a9d9d5d32093ad5"),
    "valid_negatives.txt": (
        "data/triples/codex-m/valid_negatives.txt",
        "1e77f1b1e90cc487544c6a9f299e483054e12fd6",
    ),
    "test_negatives.txt": (
        "data/triples/codex-m/test_negatives.txt",
        "ae599bf76e3cb0863803c7e708f90d379c5bf1a2",
    ),
    "entities_en.json": (
        "data/entities/en/entities.json",
        "4d16d7251779deae3b264da1a43a73ed944f4932",
    ),
    "relations_en.json": (
        "data/relations/en/relations.json",
        "d305a54cfb79916d15615dfa95af9506970a4ad0",
    ),
}
SEED = 2026


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def file_record(path: Path, *, line_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if line_count is not None:
        record["line_count"] = line_count
    return record


def atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def atomic_jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    os.replace(tmp_path, path)
    return count


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value: Any) -> Any:
    """Convert tensor/scalar values appearing in CLEGR records to JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def import_torch_without_torch_geometric() -> Any:
    """Import torch and provide the class name needed by CLEGR pickles.

    The saved CLEGR tuple contains a torch-geometric Data class reference, but
    the actual experiment environment does not need torch-geometric to read
    the serialized tensors.  A minimal placeholder is enough for unpickling;
    the stored data object is then consumed as a plain mapping.
    """
    import torch  # type: ignore

    # The missing dependency is only referenced by the pickle's class name.
    # Install a local placeholder even when torch itself imports correctly;
    # torch.load resolves the class later, during unpickling.
    class Data:
        pass

    package = types.ModuleType("torch_geometric")
    package.__path__ = []  # type: ignore[attr-defined]
    data_package = types.ModuleType("torch_geometric.data")
    data_package.__path__ = []  # type: ignore[attr-defined]
    data_module = types.ModuleType("torch_geometric.data.data")
    data_package.Data = Data  # type: ignore[attr-defined]
    data_module.Data = Data  # type: ignore[attr-defined]
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.data"] = data_package
    sys.modules["torch_geometric.data.data"] = data_module
    return torch


def load_clegr_tuple(source_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = import_torch_without_torch_geometric()
    loaded = torch.load(
        source_file,
        weights_only=False,
        mmap=True,
        map_location="cpu",
    )
    data, slices, _data_class = loaded
    if not isinstance(data, dict):
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif hasattr(data, "__dict__"):
            data = vars(data)
        else:
            raise TypeError(f"Unsupported CLEGR data object: {type(data)!r}")
    return data, slices


def clegr_graph_records(source_file: Path, dataset_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data, slices = load_clegr_tuple(source_file)
    required = {
        "edge_index",
        "graph_id",
        "node_ids",
        "node_texts",
        "edge_texts",
        "question",
        "question_group",
        "question_subgroup",
        "label",
        "question_type",
    }
    missing = required.difference(data)
    if missing:
        raise KeyError(f"{source_file} is missing CLEGR fields: {sorted(missing)}")

    graph_order: list[str] = []
    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    question_count = len(data["graph_id"])
    edge_index = data["edge_index"]
    edge_slices = slices["edge_index"]

    for question_index in range(question_count):
        graph_id = str(data["graph_id"][question_index])
        nodes = [str(item) for item in data["node_texts"][question_index]]
        node_ids = [str(item) for item in data["node_ids"][question_index]]
        edge_texts = [str(item) for item in data["edge_texts"][question_index]]
        edge_start = int(edge_slices[question_index])
        edge_end = int(edge_slices[question_index + 1])
        local_edge_index = edge_index[:, edge_start:edge_end].transpose(0, 1).tolist()
        if any(len(pair) != 2 for pair in local_edge_index):
            raise ValueError(f"Malformed edge index for {dataset_name} question {question_index}")
        edges: list[list[str]] = []
        for edge_no, (src, dst) in enumerate(local_edge_index):
            src_index = int(src)
            dst_index = int(dst)
            if not (0 <= src_index < len(nodes) and 0 <= dst_index < len(nodes)):
                raise ValueError(f"Out-of-range edge in {dataset_name}/{graph_id}")
            edges.append([nodes[src_index], edge_texts[edge_no], nodes[dst_index]])

        if graph_id not in records:
            graph_order.append(graph_id)
            records[graph_id] = {
                "title": f"{dataset_name} graph {graph_id}",
                "id": graph_id,
                "source_dataset": dataset_name,
                "graph": {
                    "node_ids": node_ids,
                    "node_list": nodes,
                    "edge_list": edges,
                    "edge_index": [[int(src), int(dst)] for src, dst in local_edge_index],
                },
                "questions": [],
                "answers": [],
                "full_answers": [],
                "question_types": [],
                "question_groups": [],
                "question_subgroups": [],
            }
        else:
            current_graph = records[graph_id]["graph"]
            if current_graph["node_ids"] != node_ids or current_graph["edge_list"] != edges:
                raise ValueError(f"Graph structure changed across questions for {dataset_name}/{graph_id}")

        record = records[graph_id]
        record["questions"].append(json_safe(data["question"][question_index]))
        answer = json_safe(data["label"][question_index])
        record["answers"].append(answer)
        record["full_answers"].append(answer)
        record["question_types"].append(json_safe(data["question_type"][question_index]))
        record["question_groups"].append(json_safe(data["question_group"][question_index]))
        record["question_subgroups"].append(json_safe(data["question_subgroup"][question_index]))

    records_list = [records[graph_id] for graph_id in graph_order]
    summary = {
        "source_question_count": question_count,
        "source_graph_count": len(records_list),
        "source_edge_count": sum(len(item["graph"]["edge_list"]) for item in records_list),
        "source_node_count": sum(len(item["graph"]["node_list"]) for item in records_list),
    }
    return records_list, summary


def split_graph_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    graph_ids = sorted(str(item["id"]) for item in records)
    if len(graph_ids) != 500:
        raise ValueError(f"Expected 500 CLEGR graphs, found {len(graph_ids)}")
    rng = random.Random(seed)
    rng.shuffle(graph_ids)
    split_ids = {
        "train": set(graph_ids[:300]),
        "val": set(graph_ids[300:400]),
        "test": set(graph_ids[400:]),
    }
    by_id = {str(item["id"]): item for item in records}
    return {
        split: [by_id[graph_id] for graph_id in sorted(ids)]
        for split, ids in split_ids.items()
    }


def clegr_split_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "graph_count": len(rows),
        "question_count": sum(len(row["questions"]) for row in rows),
        "edge_count": sum(len(row["graph"]["edge_list"]) for row in rows),
        "node_count": sum(len(row["graph"]["node_list"]) for row in rows),
    }


def prepare_clegr_variant(variant: str, *, seed: int, hash_source: bool) -> dict[str, Any]:
    source_file = CLEGR_SOURCE / variant / "processed" / "data_list.pt"
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    dataset_name = variant.replace("-", "_")
    target_dir = TARGET / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[clegr] reading {source_file} with mmap", flush=True)
    records, source_summary = clegr_graph_records(source_file, dataset_name)
    splits = split_graph_records(records, seed)

    files: dict[str, Any] = {}
    for split, rows in splits.items():
        output = target_dir / "processed" / f"processed_{split}.json"
        row_count = atomic_jsonl_dump(output, rows)
        files[str(output.relative_to(TARGET))] = {
            **file_record(output),
            "record_count": row_count,
            "summary": clegr_split_summary(rows),
        }

    source_record: dict[str, Any] = {
        "path": str(source_file),
        "size_bytes": source_file.stat().st_size,
        **source_summary,
        "sha256": sha256_file(source_file) if hash_source else None,
        "hash_status": "computed" if hash_source else "not_computed",
    }
    atomic_json_dump(
        target_dir / "raw" / "source_manifest.json",
        {
            "format_version": 1,
            "dataset": dataset_name,
            "source_file": source_record,
            "note": "The large PyTorch source is kept at its existing machine-local path; processed JSONL is the benchmark input.",
        },
    )
    files[str((target_dir / "raw" / "source_manifest.json").relative_to(TARGET))] = file_record(
        target_dir / "raw" / "source_manifest.json"
    )

    manifest = {
        "format_version": 2,
        "dataset": dataset_name,
        "task_family": "compositional_graph_question_answering",
        "created_at_utc": utc_now(),
        "seed": seed,
        "source_variant": variant,
        "source": source_record,
        "split_protocol": {
            "unit": "graph",
            "counts": {"train": 300, "val": 100, "test": 100},
            "graph_ids_disjoint": True,
            "ordering": "sorted graph ids after seeded shuffle",
            "questions_kept_per_graph": "all",
        },
        "split_summaries": {split: clegr_split_summary(rows) for split, rows in splits.items()},
        "files": files,
    }
    atomic_json_dump(target_dir / "dataset_manifest.json", manifest)
    files[str((target_dir / "dataset_manifest.json").relative_to(TARGET))] = file_record(
        target_dir / "dataset_manifest.json"
    )
    atomic_json_dump(target_dir / "tasks" / "manifest.json", {
        "format_version": 1,
        "dataset": dataset_name,
        "task_source": "processed JSONL records",
        "splits": {
            split: str((target_dir / "processed" / f"processed_{split}.json").relative_to(TARGET))
            for split in ("train", "val", "test")
        },
    })
    atomic_json_dump(target_dir / "tasks" / "generation_plan.json", {
        "format_version": 1,
        "dataset": dataset_name,
        "status": "pending_inference",
        "requires_model_inference": True,
        "model_name": "qwen-7b",
        "model_source": "local",
        "task_types": ["context", "summarization", "context_qa", "reasoning_qa"],
        "input_split": "test",
        "reference_split": "train",
        "input_file": f"processed/processed_test.json",
        "reference_file": f"processed/processed_train.json",
        "cache_directory": "tasks/cache/",
        "note": "Stage1/Stage2 QA and summary samples must be generated with the GRIP task generator; native CLEGR question-answer pairs are not substitutes.",
    })
    return manifest


def relation_manifest(dataset: str, *, raw_names: list[str], extra_names: list[str], task_name: str | None) -> dict[str, Any]:
    dataset_dir = TARGET / dataset
    raw_files: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for name in raw_names:
        path = dataset_dir / "raw" / name
        if not path.exists():
            raise FileNotFoundError(path)
        line_count = count_nonempty_lines(path) if name.endswith(".txt") else None
        raw_files[name] = file_record(path, line_count=line_count)
        if name in {"train.txt", "valid.txt", "test.txt"}:
            counts[name.removesuffix(".txt")] = line_count or 0

    generated_files: dict[str, Any] = {}
    for name in extra_names:
        path = dataset_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        generated_files[name] = file_record(path)
    if task_name is not None:
        task_path = dataset_dir / "tasks" / task_name
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        generated_files[str(task_path.relative_to(dataset_dir))] = file_record(task_path)

    manifest = {
        "format_version": 2,
        "dataset": dataset,
        "task_family": "relation_prediction",
        "created_at_utc": utc_now(),
        "seed": SEED,
        "counts": {"raw_triples": counts},
        "raw_files": raw_files,
        "generated_files": generated_files,
        "protocol": {
            "graph_source": "train.txt",
            "candidate_source": "train.txt",
            "validation_and_test_are_evaluation_only": True,
        },
    }
    atomic_json_dump(dataset_dir / "manifest.json", manifest)
    return manifest


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def fetch_codexm_source() -> tuple[dict[str, bytes], dict[str, Any]]:
    """Download the official CoDEx-M files and verify their pinned Git blobs."""
    contents: dict[str, bytes] = {}
    provenance: dict[str, Any] = {}
    for local_name, (upstream_path, expected_blob) in CODEXM_UPSTREAM_FILES.items():
        url = f"https://raw.githubusercontent.com/tsafavi/codex/{CODEXM_COMMIT}/{upstream_path}"
        request = Request(url, headers={"User-Agent": "GRIP-CU benchmark preparation"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        actual_blob = git_blob_sha1(content)
        if actual_blob != expected_blob:
            raise ValueError(
                f"Pinned CoDEx source changed for {upstream_path}: "
                f"expected Git blob {expected_blob}, received {actual_blob}"
            )
        contents[local_name] = content
        provenance[local_name] = {
            "upstream_path": upstream_path,
            "url": url,
            "git_blob_sha1": actual_blob,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    return contents, provenance


def parse_codexm_triples(content: bytes, source_name: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) != 3 or any(not field for field in fields):
            raise ValueError(
                f"{source_name}:{line_number}: expected three non-empty tab-separated fields"
            )
        triples.append((fields[0], fields[1], fields[2]))
    return triples


def codexm_entity_surface(entity_id: str, metadata: dict[str, Any]) -> str:
    label = str(metadata.get("label") or "").strip()
    description = str(metadata.get("description") or "").strip()
    if label and description:
        text = f"{label}, {description}"
    else:
        text = label or description or entity_id
    return text.strip().strip(".")


def codexm_processed_record(
    split: str,
    triples: list[tuple[str, str, str]],
    *,
    train_triples: list[tuple[str, str, str]],
    entity_text: dict[str, str],
    entity_ids: dict[str, int],
    relation_labels: dict[str, str],
    relation_order: list[str],
    rng: Any,
) -> dict[str, Any]:
    question_format = (
        "What is the relation between word node {src} and word node {tgt}? "
        "Selected from the following candidate answers: {candidates}."
    )
    train_known: dict[tuple[str, str], set[str]] = {}
    for head, relation, tail in train_triples:
        train_known.setdefault((head, tail), set()).add(relation)

    graph_edge_index = [[entity_ids[head], entity_ids[tail]] for head, _, tail in train_triples]
    graph_edge_list = [
        [entity_text[head], relation_labels[relation], entity_text[tail]]
        for head, relation, tail in train_triples
    ]
    questions: list[list[Any]] = []
    answers: list[str] = []
    candidate_relations: list[list[str]] = []
    question_metadata: list[dict[str, Any]] = []
    minimum_negative_pool = len(relation_order)

    for triple_index, (head, gold_relation, tail) in enumerate(triples):
        blocked = train_known.get((head, tail), set()) | {gold_relation}
        negative_ids = [relation for relation in relation_order if relation not in blocked]
        minimum_negative_pool = min(minimum_negative_pool, len(negative_ids))
        if len(negative_ids) < 9:
            raise ValueError(
                f"{split} row {triple_index} has only {len(negative_ids)} legal negatives; need 9"
            )
        selected_ids = rng.sample(negative_ids, 9)
        candidate_ids = selected_ids + [gold_relation]
        rng.shuffle(candidate_ids)
        candidate_texts = [relation_labels[relation] for relation in candidate_ids]
        question = question_format.format(
            src=entity_text[head],
            tgt=entity_text[tail],
            candidates="; ".join(candidate_texts),
        )
        questions.append([question, [entity_ids[head], entity_ids[tail]]])
        answers.append(relation_labels[gold_relation])
        candidate_relations.append(candidate_texts)
        question_metadata.append(
            {
                "split": split,
                "source_triple_index": triple_index,
                "triple_ids": [head, gold_relation, tail],
            }
        )

    return {
        "title": "codexm",
        "graph": {
            "edge_index": graph_edge_index,
            "edge_list": graph_edge_list,
            "node_list": list(entity_text.values()),
        },
        "questions": questions,
        "answers": answers,
        "candidate_relations": candidate_relations,
        "question_metadata": question_metadata,
        "id": "0",
        "_candidate_audit": {
            "rows": len(triples),
            "minimum_legal_negative_pool": minimum_negative_pool,
        },
    }


def codexm_train_reference_record(
    train_triples: list[tuple[str, str, str]],
    *,
    entity_text: dict[str, str],
    relation_labels: dict[str, str],
    relation_order: list[str],
    seed: int,
) -> dict[str, Any]:
    """Build train-only relation QA examples for reasoning-task few-shot prompts."""
    import random

    question_format = (
        "What is the relation between word node {src} and word node {tgt}? "
        "Selected from the following candidate answers: {candidates}."
    )
    train_known: dict[tuple[str, str], set[str]] = {}
    for head, relation, tail in train_triples:
        train_known.setdefault((head, tail), set()).add(relation)

    rng = random.Random(seed)
    questions: list[str] = []
    answers: list[str] = []
    for index, (head, gold_relation, tail) in enumerate(train_triples):
        negative_pool = [
            relation
            for relation in relation_order
            if relation not in train_known[(head, tail)]
        ]
        if len(negative_pool) < 9:
            raise ValueError(
                f"train reference row {index} has only {len(negative_pool)} legal negatives"
            )
        candidates = rng.sample(negative_pool, 9) + [gold_relation]
        rng.shuffle(candidates)
        questions.append(
            question_format.format(
                src=entity_text[head],
                tgt=entity_text[tail],
                candidates="; ".join(relation_labels[relation] for relation in candidates),
            )
        )
        answers.append(relation_labels[gold_relation])

    return {
        "title": "codexm_train_reference",
        "id": "train_reference",
        "split": "train",
        "questions": questions,
        "answers": answers,
        "reference_metadata": {
            "source_file": "raw/train.txt",
            "source_triple_count": len(train_triples),
            "seed": seed,
            "candidate_count": 10,
            "validation_or_test_examples_included": False,
        },
    }


def prepare_codexm(seed: int = SEED) -> dict[str, Any]:
    """Package official CoDEx-M triples and deterministic relation QA records."""
    dataset_dir = TARGET / "codexm"
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    tasks_dir = dataset_dir / "tasks"
    raw_dir.mkdir(parents=True, exist_ok=True)

    upstream, upstream_provenance = fetch_codexm_source()
    upstream_names = {
        "train.txt": "train.txt",
        "valid.txt": "valid.txt",
        "test.txt": "test.txt",
        "valid_negatives.txt": "valid_negatives.txt",
        "test_negatives.txt": "test_negatives.txt",
    }
    for local_name, upstream_name in upstream_names.items():
        target_path = raw_dir / local_name
        target_path.write_bytes(upstream[upstream_name])

    train_triples = parse_codexm_triples(upstream["train.txt"], "train.txt")
    valid_triples = parse_codexm_triples(upstream["valid.txt"], "valid.txt")
    test_triples = parse_codexm_triples(upstream["test.txt"], "test.txt")
    valid_negatives = parse_codexm_triples(upstream["valid_negatives.txt"], "valid_negatives.txt")
    test_negatives = parse_codexm_triples(upstream["test_negatives.txt"], "test_negatives.txt")

    positive_triples = train_triples + valid_triples + test_triples
    all_entities = {entity for head, _, tail in positive_triples for entity in (head, tail)}
    all_entities.update(entity for rows in (valid_negatives, test_negatives) for h, _, t in rows for entity in (h, t))
    relation_order: list[str] = []
    seen_relations: set[str] = set()
    for _, relation, _ in train_triples:
        if relation not in seen_relations:
            seen_relations.add(relation)
            relation_order.append(relation)
    all_relations = {relation for _, relation, _ in positive_triples}
    if all_relations != set(relation_order):
        raise ValueError("CoDEx-M validation/test contains a relation absent from train")

    official_entities = json.loads(upstream["entities_en.json"].decode("utf-8"))
    official_relations = json.loads(upstream["relations_en.json"].decode("utf-8"))
    missing_entities = sorted(all_entities.difference(official_entities))
    missing_relations = sorted(all_relations.difference(official_relations))
    if missing_entities or missing_relations:
        raise ValueError(
            f"CoDEx metadata missing entities={len(missing_entities)} relations={len(missing_relations)}"
        )

    entity2text = {entity: official_entities[entity] for entity in official_entities if entity in all_entities}
    relation2text = {relation: official_relations[relation] for relation in relation_order}
    relation_labels = {
        relation: str(metadata.get("label") or "").strip()
        for relation, metadata in relation2text.items()
    }
    if any(not label for label in relation_labels.values()):
        raise ValueError("Every CoDEx-M train relation must have a non-empty English label")
    if len(set(relation_labels.values())) != len(relation_labels):
        raise ValueError("CoDEx-M relation labels are not unique; candidate answers would be ambiguous")

    entity_text = {
        entity: codexm_entity_surface(entity, metadata)
        for entity, metadata in entity2text.items()
    }
    expected_counts = {
        "entities": 17050,
        "relations": 51,
        "train": 185584,
        "valid": 10310,
        "test": 10311,
        "valid_negatives": 10310,
        "test_negatives": 10311,
    }
    actual_counts = {
        "entities": len(entity_text),
        "relations": len(relation_order),
        "train": len(train_triples),
        "valid": len(valid_triples),
        "test": len(test_triples),
        "valid_negatives": len(valid_negatives),
        "test_negatives": len(test_negatives),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected pinned CoDEx-M statistics: {actual_counts}")
    entity_ids = {entity: index for index, entity in enumerate(entity_text)}
    raw_text_files = {
        "entity2text.json": entity2text,
        "codexm.json": relation2text,
        "rel2text.json": relation_labels,
    }
    for filename, payload in raw_text_files.items():
        atomic_json_dump(raw_dir / filename, payload)

    import random

    rng = random.Random(seed)
    processed_records: dict[str, dict[str, Any]] = {}
    candidate_audit: dict[str, Any] = {}
    for split, triples in (("val", valid_triples), ("test", test_triples)):
        record = codexm_processed_record(
            split,
            triples,
            train_triples=train_triples,
            entity_text=entity_text,
            entity_ids=entity_ids,
            relation_labels=relation_labels,
            relation_order=relation_order,
            rng=rng,
        )
        candidate_audit[split] = record.pop("_candidate_audit")
        output_path = processed_dir / f"processed_{split}.json"
        atomic_jsonl_dump(output_path, [record])
        processed_records[split] = record

    train_reference = codexm_train_reference_record(
        train_triples,
        entity_text=entity_text,
        relation_labels=relation_labels,
        relation_order=relation_order,
        seed=seed,
    )
    atomic_jsonl_dump(processed_dir / "processed_train_reference.json", [train_reference])
    train_graph_record = {
        "title": "codexm",
        "graph": processed_records["val"]["graph"],
        "id": "0",
        "source_split": "train_graph",
    }
    atomic_jsonl_dump(processed_dir / "processed_train_graph.json", [train_graph_record])

    label_to_relation = {label: relation for relation, label in relation_labels.items()}
    heldout_relations_by_pair: dict[tuple[str, str], set[str]] = {}
    for head, relation, tail in valid_triples + test_triples:
        heldout_relations_by_pair.setdefault((head, tail), set()).add(relation)
    train_relations_by_pair: dict[tuple[str, str], set[str]] = {}
    for head, relation, tail in train_triples:
        train_relations_by_pair.setdefault((head, tail), set()).add(relation)
    for split, triples in (("val", valid_triples), ("test", test_triples)):
        record = processed_records[split]
        total_negative_candidates = 0
        known_train_collisions = 0
        heldout_positive_collisions = 0
        collision_queries = 0
        for index, (head, gold_relation, tail) in enumerate(triples):
            candidates = [label_to_relation[label] for label in record["candidate_relations"][index]]
            negative_candidates = [relation for relation in candidates if relation != gold_relation]
            total_negative_candidates += len(negative_candidates)
            known_train_collisions += len(
                set(negative_candidates) & train_relations_by_pair.get((head, tail), set())
            )
            heldout_gold_relations = heldout_relations_by_pair.get((head, tail), set()) - {gold_relation}
            overlaps = set(negative_candidates) & heldout_gold_relations
            heldout_positive_collisions += len(overlaps)
            collision_queries += bool(overlaps)
        if known_train_collisions:
            raise ValueError(f"{split} candidate list contains a train-known relation")
        candidate_audit[split].update(
            {
                "candidate_count_per_query": 10,
                "negative_candidate_count": total_negative_candidates,
                "train_known_collision_candidates": known_train_collisions,
                "heldout_positive_overlap_candidates": heldout_positive_collisions,
                "queries_with_heldout_positive_overlap": collision_queries,
                "open_world_unresolved_candidates": total_negative_candidates - heldout_positive_collisions,
                "holdout_overlap_note": "Measured after candidate lists were frozen; valid/test facts do not filter or change candidate sampling.",
            }
        )

    plan = {
        "format_version": 1,
        "dataset": "codexm",
        "status": "pending_inference",
        "requires_model_inference": True,
        "model_name": "qwen-7b",
        "model_source": "local",
        "task_types": ["context", "summarization", "context_qa", "reasoning_qa"],
        "input_split": "train",
        "reference_split": "train",
        "input_file": "processed/processed_train_graph.json",
        "reference_file": "processed/processed_train_reference.json",
        "graph_nodes": len(entity_text),
        "graph_edges": len(train_triples),
        "cache_directory": "tasks/cache/",
        "output_file": "tasks/grip_codexm_tasks.json",
        "note": (
            "Relation-ranking data conversion is complete. Stage1/Stage2 GRIP task generation "
            "uses a graph-only train input and train-only few-shot references; valid/test rows are "
            "reserved for evaluation. CoDEx-M verified negatives are reserved for triple-classification "
            "audits and are not relation candidates."
        ),
    }
    atomic_json_dump(tasks_dir / "generation_plan.json", plan)

    raw_files: dict[str, Any] = {}
    for filename in (
        "train.txt",
        "valid.txt",
        "test.txt",
        "valid_negatives.txt",
        "test_negatives.txt",
        "entity2text.json",
        "codexm.json",
        "rel2text.json",
    ):
        file_path = raw_dir / filename
        line_count = count_nonempty_lines(file_path) if filename.endswith(".txt") else None
        raw_files[filename] = file_record(file_path, line_count=line_count)

    generated_files = {
        f"processed/processed_{split}.json": {
            **file_record(processed_dir / f"processed_{split}.json"),
            "record_count": 1,
            "question_count": len(processed_records[split]["questions"]),
        }
        for split in ("val", "test")
    }
    generated_files["processed/processed_train_graph.json"] = {
        **file_record(processed_dir / "processed_train_graph.json"),
        "record_count": 1,
        "edge_count": len(train_triples),
        "source_split": "raw/train.txt",
    }
    generated_files["processed/processed_train_reference.json"] = {
        **file_record(processed_dir / "processed_train_reference.json"),
        "record_count": 1,
        "question_count": len(train_reference["questions"]),
        "source_split": "raw/train.txt",
    }
    generated_files["tasks/generation_plan.json"] = file_record(tasks_dir / "generation_plan.json")
    source_manifest = {
        "format_version": 1,
        "dataset": "codexm",
        "upstream_repository": "https://github.com/tsafavi/codex",
        "upstream_commit": CODEXM_COMMIT,
        "upstream_commit_url": f"https://github.com/tsafavi/codex/tree/{CODEXM_COMMIT}",
        "language": "en",
        "downloaded_source_files": upstream_provenance,
        "derived_text_files": {
            "entity2text.json": "English entity metadata subset for IDs appearing in positive or verified-negative CoDEx-M triples.",
            "codexm.json": "English relation labels and descriptions for the 51 train relations.",
            "rel2text.json": "Raw relation ID to English label map consumed by the SC-CL tooling.",
        },
        "note": "All source files are pinned and verified against upstream Git blob SHA-1 before conversion.",
    }
    atomic_json_dump(raw_dir / "source_manifest.json", source_manifest)
    generated_files["raw/source_manifest.json"] = file_record(raw_dir / "source_manifest.json")

    manifest = {
        "format_version": 2,
        "dataset": "codexm",
        "dataset_name": "CoDEx-M",
        "task_family": "relation_prediction",
        "created_at_utc": utc_now(),
        "seed": seed,
        "source": source_manifest,
        "counts": {
            "entities": len(entity_text),
            "relations": len(relation_order),
            "raw_triples": {
                "train": len(train_triples),
                "valid": len(valid_triples),
                "test": len(test_triples),
            },
            "verified_triple_classification_negatives": {
                "valid": len(valid_negatives),
                "test": len(test_negatives),
            },
        },
        "raw_files": raw_files,
        "generated_files": generated_files,
        "protocol": {
            "task": "given fixed (head, tail), rank the gold relation among 10 candidates",
            "graph_source": "raw/train.txt",
            "candidate_source": "raw/train.txt relation vocabulary",
            "candidate_count": 10,
            "negative_count": 9,
            "candidate_sampling": "uniform without replacement; one frozen RNG stream seeded from manifest seed",
            "known_train_relation_filter": "remove all train-known relations for the same (head, tail), as well as the gold relation",
            "validation_and_test_are_evaluation_only": True,
            "verified_negative_files": "retained for triple-classification audit only; never used as fixed-(head,tail) relation candidates or truth labels",
            "candidate_coverage": candidate_audit,
            "task_generation_reference": {
                "split": "train",
                "source": "raw/train.txt",
                "purpose": "eight-shot examples for synthetic reasoning QA generation",
                "valid_test_examples_included": False,
            },
            "stage1_stage2_task_generation": "pending_inference",
        },
    }
    atomic_json_dump(dataset_dir / "manifest.json", manifest)
    return manifest


def prepare_wn18rr() -> dict[str, Any]:
    dataset_dir = TARGET / "wn18rr"
    for name in ("entity2text.txt", "train.txt", "valid.txt", "test.txt", "wn18rr.json"):
        copy_file(WN_SOURCE / name, dataset_dir / "raw" / name)
    # The legacy GRIP files are a single JSON graph record, while the current
    # runner consumes JSONL.  Normalize the container without changing the
    # graph, questions, or answers.
    for name in ("processed_val.json", "processed_test.json"):
        legacy_record = json.loads((WN_SOURCE / name).read_text(encoding="utf-8"))
        atomic_jsonl_dump(dataset_dir / "processed" / name, [legacy_record])

    test_record = json.loads((WN_SOURCE / "processed_test.json").read_text(encoding="utf-8"))
    graph = test_record["graph"]
    atomic_json_dump(dataset_dir / "tasks" / "generation_plan.json", {
        "format_version": 1,
        "dataset": "wn18rr",
        "status": "pending_inference",
        "requires_model_inference": True,
        "model_name": "qwen-7b",
        "model_source": "local",
        "task_types": ["context", "summarization", "context_qa", "reasoning_qa"],
        "input_split": "test",
        "reference_split": "val",
        "input_file": "processed/processed_test.json",
        "reference_file": "processed/processed_val.json",
        "graph_nodes": len(graph["node_list"]),
        "graph_edges": len(graph["edge_list"]),
        "cache_directory": "tasks/cache/",
        "note": "Stage1/Stage2 QA and summary samples must be generated with the GRIP task generator; static relation QA is not a substitute.",
    })
    return relation_manifest(
        "wn18rr",
        raw_names=["entity2text.txt", "train.txt", "valid.txt", "test.txt", "wn18rr.json"],
        extra_names=["processed/processed_val.json", "processed/processed_test.json"],
        task_name="generation_plan.json",
    )


def validate_existing_relation_dataset(dataset: str, *, raw_names: list[str], extra_names: list[str], task_name: str) -> dict[str, Any]:
    return relation_manifest(dataset, raw_names=raw_names, extra_names=extra_names, task_name=task_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-codexm",
        action="store_true",
        help="Prepare CoDEx-M and update the root manifest without reprocessing other datasets.",
    )
    parser.add_argument("--skip-clegr-facts", action="store_true")
    parser.add_argument("--skip-clegr-reasoning", action="store_true")
    parser.add_argument(
        "--hash-source",
        action="store_true",
        help="Compute SHA256 for the large CLEGR source tensors; this reads about 14GB.",
    )
    args = parser.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    if args.only_codexm:
        codexm_manifest = prepare_codexm()
        root_manifest_path = TARGET / "manifest.json"
        if root_manifest_path.is_file():
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        else:
            root_manifest = {"format_version": 2, "seed": SEED, "datasets": {}, "notes": {}}
        root_manifest["created_at_utc"] = utc_now()
        root_manifest["repository_commit"] = git_commit()
        root_manifest["python_executable"] = sys.executable
        root_manifest.setdefault("datasets", {})["codexm"] = {
            "manifest": "codexm/manifest.json",
            "task_family": codexm_manifest["task_family"],
            "counts": codexm_manifest["counts"],
        }
        root_manifest.setdefault("notes", {})["codexm"] = (
            "CoDEx-M is pinned to the official upstream commit; verified triple-classification "
            "negatives are retained separately from relation-ranking candidates."
        )
        atomic_json_dump(root_manifest_path, root_manifest)
        print(json.dumps(root_manifest, ensure_ascii=False, indent=2))
        return

    manifests: dict[str, Any] = {}
    manifests["nell23k"] = validate_existing_relation_dataset(
        "nell23k",
        raw_names=["entity2text.json", "train.txt", "valid.txt", "test.txt"],
        extra_names=["processed/processed_val.json", "processed/processed_test.json"],
        task_name="grip_nell23k_tasks.json",
    )
    manifests["fb15k237_2"] = validate_existing_relation_dataset(
        "fb15k237_2",
        raw_names=["entity2wikidata.json", "rel2text.json", "train.txt", "valid.txt", "test.txt"],
        extra_names=["processed/processed_val.json", "processed/processed_test.json"],
        task_name="grip_fb15k237_2_tasks.json",
    )
    manifests["wn18rr"] = prepare_wn18rr()
    manifests["codexm"] = prepare_codexm()

    if not args.skip_clegr_facts:
        manifests["clegr_facts"] = prepare_clegr_variant("clegr-facts", seed=SEED, hash_source=args.hash_source)
    if not args.skip_clegr_reasoning:
        manifests["clegr_reasoning"] = prepare_clegr_variant("clegr-reasoning", seed=SEED, hash_source=args.hash_source)

    root_manifest = {
        "format_version": 2,
        "created_at_utc": utc_now(),
        "repository_commit": git_commit(),
        "python_executable": sys.executable,
        "seed": SEED,
        "datasets": {
            name: {
                "manifest": str((TARGET / name / ("dataset_manifest.json" if name.startswith("clegr_") else "manifest.json")).relative_to(TARGET)),
                "task_family": payload.get("task_family"),
                "counts": payload.get("counts", payload.get("split_summaries")),
            }
            for name, payload in manifests.items()
        },
        "notes": {
            "relation_datasets": "Existing NELL23K/FB15K237-2 assets were validated, not regenerated.",
            "wn18rr": "Copied from the reproduced GRIP asset and packaged with deterministic provenance.",
            "codexm": "Downloaded from a pinned official commit; triple-classification verified negatives remain separate from relation candidates.",
            "clegr": "Both standard CLEGR variants are converted from local PyTorch tensors using memory mapping; source tensors are not duplicated.",
        },
    }
    atomic_json_dump(TARGET / "manifest.json", root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
