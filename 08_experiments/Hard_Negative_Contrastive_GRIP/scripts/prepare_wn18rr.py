#!/usr/bin/env python3
"""Prepare CPU-only WN18RR all-11 relation prediction with source provenance.

Validation then test consume one seeded RandomState, with no subsampling.
Sorted training relation texts freeze the official processor's unordered set.
Training reference selection and candidates have independent random streams.
All source rows, including a final row without a newline, are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from prepare_fb15k237 import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
WAY = 11
TITLE = "wn18rr"
REQUIRED_FILES = ("entity2text.txt", "train.txt", "valid.txt", "test.txt")
SPLIT_FILES = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
EXPECTED_COUNTS = {"train": 86835, "validation": 3034, "test": 3134}
CANDIDATE_SOURCE = "official_grip_wn18rr_sorted_train_relation_texts_all11"
QUESTION_TEMPLATE = (
    "What is the relation between word node {src} and word node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)


def source_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if line.lstrip().startswith("version https://git-lfs.github.com/spec/"):
                raise ValueError(f"{path.name}:{index}: Git LFS pointer is not raw data")
            if not line.strip():
                raise ValueError(f"{path.name}:{index}: blank source line")
            yield index, line


def load_entities(path: Path) -> dict[str, str]:
    entities = {}
    for index, line in source_lines(path):
        fields = line.strip().split("\t")
        if len(fields) != 2 or not all(fields):
            raise ValueError(f"{path.name}:{index}: expected entity ID and text separated by tab")
        key, text = fields
        if key in entities:
            raise ValueError(f"{path.name}:{index}: duplicate entity ID {key}")
        entities[key] = text
    if not entities:
        raise ValueError("empty entity mapping")
    return entities


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for index, line in source_lines(path):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path.name}:{index}: expected exactly three fields")
        rows.append(tuple(fields))
    if not rows:
        raise ValueError(f"{path.name}: empty triple file")
    return rows


def official_relation_text(relation: str) -> str:
    text = " ".join(relation.split("_")[1:])
    if not text:
        raise ValueError(f"invalid WN18RR relation ID: {relation!r}")
    return text


def load_raw(raw_dir: Path, *, validate_counts: bool = True):
    entities = load_entities(raw_dir / "entity2text.txt")
    triples = {split: load_triples(raw_dir / name) for split, name in SPLIT_FILES.items()}
    if validate_counts:
        for split, expected in EXPECTED_COUNTS.items():
            if len(triples[split]) != expected:
                raise ValueError(f"{split}: expected {expected} triples, got {len(triples[split])}")
    relation_ids = {rel for _, rel, _ in triples["train"]}
    relation_text = {rel: official_relation_text(rel) for rel in sorted(relation_ids)}
    if len(relation_ids) != WAY or len(set(relation_text.values())) != WAY:
        raise ValueError(f"expected {WAY} distinct training relation IDs and texts")
    for split in ("validation", "test"):
        missing = {rel for _, rel, _ in triples[split]} - relation_ids
        if missing:
            raise ValueError(f"{split}: relations absent from training pool: {sorted(missing)}")
    # Match the official fallback only for training nodes; evaluation cannot
    # append out-of-range IDs after the graph has frozen its node list.
    train_unknown = 0
    for src, _, tgt in triples["train"]:
        for entity in (src, tgt):
            if entity not in entities:
                entities[entity] = "Unknown"
                train_unknown += 1
    missing = sorted({entity for split in ("validation", "test")
                      for src, _, tgt in triples[split] for entity in (src, tgt)
                      if entity not in entities})
    if missing:
        raise ValueError(f"held-out entities absent from official train graph nodes: {missing[:10]}")
    return entities, triples, relation_text, train_unknown


def official_record(graph: dict, samples: list[dict]) -> dict:
    return {"title": TITLE, "graph": graph,
            "questions": [[q["question"], q["node_indices"]] for q in samples],
            "answers": [q["answer"] for q in samples], "id": "0"}


def prepare_wn18rr(raw_dir: Path, *, seed: int = SEED, train_questions: int = 2000,
                   smoke_questions: int = 96, validate_counts: bool = True):
    if min(train_questions, smoke_questions) < 1:
        raise ValueError("question counts must be positive")
    source_hashes = {name: sha256_file(raw_dir / name) for name in REQUIRED_FILES}
    entities, triples, relation_text, train_unknown = load_raw(raw_dir, validate_counts=validate_counts)
    if train_questions > len(triples["train"]):
        raise ValueError("train_questions exceeds training source count")
    if smoke_questions > min(train_questions, len(triples["validation"]), len(triples["test"])):
        raise ValueError("smoke_questions must fit in every full split")
    entity_ids = {key: index for index, key in enumerate(entities)}
    relation_order = sorted(relation_text.values())
    graph = {
        "edge_index": [[entity_ids[src], entity_ids[tgt]] for src, _, tgt in triples["train"]],
        "edge_list": [[entities[src], relation_text[rel], entities[tgt]] for src, rel, tgt in triples["train"]],
        "node_list": list(entities.values()),
    }

    def question(split, index, rng):
        src, rel, tgt = triples[split][index]
        candidates = rng.permutation(relation_order).tolist()
        return {
            "question_id": f"wn18rr:{split}:{index}",
            "question": QUESTION_TEMPLATE.format(src=entities[src], tgt=entities[tgt], candidates="; ".join(candidates)),
            "answer": relation_text[rel], "split": split, "true_hop": None,
            "question_type": "WN18RRRelationPrediction",
            "source_node": entities[src], "target_node": entities[tgt],
            "source_entity_id": src, "target_entity_id": tgt, "source_relation_id": rel,
            "node_indices": [entity_ids[src], entity_ids[tgt]],
            "candidate_relations": candidates, "candidate_source": CANDIDATE_SOURCE,
            "source_file": SPLIT_FILES[split], "source_triple_index": index,
        }

    selection_rng = np.random.RandomState(seed)
    train_rng = np.random.RandomState(seed)
    selected = {"train": [question("train", int(index), train_rng)
                           for index in selection_rng.permutation(len(triples["train"]))[:train_questions]]}
    eval_rng = np.random.RandomState(seed)
    for split in ("validation", "test"):
        selected[split] = [question(split, index, eval_rng) for index in range(len(triples[split]))]
    metadata = {
        "task": "WN18RRRelationPrediction", "graph_source": "train.txt", "train_qa_source": "train.txt",
        "validation_source": "valid.txt", "test_source": "test.txt", "seed": seed,
        "num_candidates": WAY, "candidate_source": CANDIDATE_SOURCE,
        "relation_order": "sorted unique train relation TEXTS",
        "structural_distance_status": "not computed; true_hop is null",
        "eval_protocol": "all validation then all test source triples; all 11 candidates; no subsampling",
        "train_rng": "independent RandomState(seed) instances for selection and candidates",
    }
    base = {"title": TITLE, "id": "0", "dataset": "WN18RR", "graph": graph}
    outputs = {
        "processed_train.json": official_record(graph, selected["train"]),
        "processed_val.json": official_record(graph, selected["validation"]),
        "processed_test.json": official_record(graph, selected["test"]),
        "recurrent_relation_prediction_full.json": {
            **base, "recurrent_questions": [q for rows in selected.values() for q in rows],
            "dataset_metadata": {**metadata, "subset": "full"}},
        "recurrent_relation_prediction_smoke.json": {
            **base, "recurrent_questions": [q for rows in selected.values() for q in rows[:smoke_questions]],
            "dataset_metadata": {**metadata, "subset": "smoke", "smoke_selection": "first N per split from full",
                                 "smoke_questions_per_split": smoke_questions}},
    }
    verify_records(outputs, triples, entities, entity_ids, relation_text, train_questions)
    for name, expected in source_hashes.items():
        if sha256_file(raw_dir / name) != expected:
            raise RuntimeError(f"source changed during preparation: {name}")
    manifest = {
        "format_version": 1, "dataset": "WN18RR", "source": "WFRaain/TAG_datasets/WN18RR",
        "source_sha256": source_hashes, "seed": seed, "rng": "numpy.random.RandomState (MT19937)",
        "numpy_version": np.__version__, "num_candidates": WAY, "relation_order": relation_order,
        "official_processor": "data/raw_datasets/wn18rr/process.py",
        "deterministic_deviation": "Official list(set(train relation texts)) becomes sorted unique training relation texts; exact permutations may differ from an arbitrary original process run.",
        "other_deviations": [
            "Retain the final source line even without a trailing newline; reject blank or malformed lines.",
            "Add independently sampled training references and deterministic smoke prefixes.",
            "Reject unmapped held-out entities instead of emitting indices beyond the frozen graph.",
        ],
        "eval_rng_schedule": ["valid: permutation(all 11 texts) for every source triple",
                              "test: permutation(all 11 texts) for every source triple; continue valid RNG"],
        "eval_subsampling": False, "official_counts_validated": validate_counts,
        "train_rng": metadata["train_rng"],
        "train_selection": "RandomState(seed).permutation(train count)[:train_questions]",
        "smoke_selection": "first N per split from full; no candidate resampling",
        "question_id_definition": "wn18rr:{train|validation|test}:{zero-based source triple index}",
        "entity_text_policy": "official tab-separated text; unknown train entities append as Unknown; reject unmapped held-out entities",
        "relation_text_policy": "' '.join(relation_id.split('_')[1:])",
        "structural_distance_status": metadata["structural_distance_status"],
        "counts": {
            "source_triples": {split: len(rows) for split, rows in triples.items()},
            "graph_nodes": len(entities), "graph_edges": len(graph["edge_list"]),
            "train_relation_ids": len(relation_text), "train_relation_texts": len(relation_order),
            "train_unknown_entities": train_unknown,
            "full_questions": {split: len(rows) for split, rows in selected.items()},
            "smoke_questions": {split: smoke_questions for split in selected}},
        "verification": {
            "train_only_graph_and_references": True, "source_triple_ids_and_entity_text": True,
            "eleven_unique_train_text_candidates_including_gold": True,
            "all_validation_and_test_source_lines_retained": True,
            "official_and_recurrent_questions_match": True, "smoke_identical_subset_of_full": True,
            "source_hashes_unchanged_during_preparation": True},
    }
    return outputs, manifest


def verify_records(outputs, triples, entities, entity_ids, relation_text, train_questions):
    full = outputs["recurrent_relation_prediction_full.json"]
    graph = full["graph"]
    if graph != {
        "edge_index": [[entity_ids[src], entity_ids[tgt]] for src, _, tgt in triples["train"]],
        "edge_list": [[entities[src], relation_text[rel], entities[tgt]] for src, rel, tgt in triples["train"]],
        "node_list": list(entities.values()),
    }:
        raise ValueError("graph does not exactly match training triples and node mapping")
    by_split = {split: [] for split in triples}
    ids = set()
    for q in full["recurrent_questions"]:
        split, index = q["split"], q["source_triple_index"]
        if index < 0 or index >= len(triples[split]):
            raise ValueError("source triple index out of range")
        src, rel, tgt = triples[split][index]
        if q["question_id"] != f"wn18rr:{split}:{index}" or q["question_id"] in ids:
            raise ValueError("invalid or duplicate source question ID")
        ids.add(q["question_id"])
        if (q["source_file"], q["source_entity_id"], q["source_relation_id"], q["target_entity_id"]) != (SPLIT_FILES[split], src, rel, tgt):
            raise ValueError("source provenance mismatch")
        if (q["source_node"], q["answer"], q["target_node"]) != (entities[src], relation_text[rel], entities[tgt]):
            raise ValueError("source text mismatch")
        candidates = q["candidate_relations"]
        if len(candidates) != WAY or set(candidates) != set(relation_text.values()):
            raise ValueError("invalid all-11 candidates")
        if q["node_indices"] != [entity_ids[src], entity_ids[tgt]]:
            raise ValueError("node indices mismatch")
        if q["question"] != QUESTION_TEMPLATE.format(src=entities[src], tgt=entities[tgt], candidates="; ".join(candidates)):
            raise ValueError("question prompt mismatch")
        by_split[split].append(q)
    if len(by_split["train"]) != train_questions:
        raise ValueError("incorrect train reference count")
    for split in ("validation", "test"):
        if [q["source_triple_index"] for q in by_split[split]] != list(range(len(triples[split]))):
            raise ValueError("evaluation must retain every source line in order")
    for split, filename in (("train", "processed_train.json"), ("validation", "processed_val.json"), ("test", "processed_test.json")):
        if outputs[filename] != official_record(graph, by_split[split]):
            raise ValueError("official/recurrent records mismatch")
    smoke = outputs["recurrent_relation_prediction_smoke.json"]
    n = smoke["dataset_metadata"]["smoke_questions_per_split"]
    if smoke["graph"] != graph or smoke["recurrent_questions"] != [q for rows in by_split.values() for q in rows[:n]]:
        raise ValueError("smoke must preserve full per-split prefixes")
    if Counter(q["split"] for q in smoke["recurrent_questions"]) != Counter({split: n for split in triples}):
        raise ValueError("incorrect smoke split counts")


def write_preparation(raw_dir: Path, out_dir: Path, **kwargs):
    outputs, manifest = prepare_wn18rr(raw_dir, **kwargs)
    # Derive the training helper mapping from the same hashed source snapshot.
    relations = {q[1]: official_relation_text(q[1]) for q in load_triples(raw_dir / "train.txt")}
    if any(sha256_file(raw_dir / name) != digest for name, digest in manifest["source_sha256"].items()):
        raise RuntimeError("source changed before writing preparation")
    manifest["raw_outputs"] = {}
    manifest["outputs"] = {}
    statuses = {}
    manifest["raw_outputs"]["rel2text.json"], statuses["raw/rel2text.json"] = atomic_write_json(raw_dir / "rel2text.json", dict(sorted(relations.items())), jsonl=False)
    for name, payload in outputs.items():
        manifest["outputs"][name], statuses[name] = atomic_write_json(out_dir / name, payload, jsonl=True)
    _, statuses["preparation_manifest.json"] = atomic_write_json(out_dir / "preparation_manifest.json", manifest, jsonl=False)
    return manifest, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=ROOT / "data/raw_datasets/wn18rr")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data/wn18rr")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_questions", type=int, default=2000)
    parser.add_argument("--smoke_questions", type=int, default=96)
    parser.add_argument("--allow_small_fixture", action="store_true", help="Disable official split-count checks for CPU fixtures only; still require 11 relations")
    args = parser.parse_args()
    manifest, statuses = write_preparation(args.raw_dir, args.out_dir, seed=args.seed,
        train_questions=args.train_questions, smoke_questions=args.smoke_questions,
        validate_counts=not args.allow_small_fixture)
    print(json.dumps({"output_dir": str(args.out_dir.resolve()), "counts": manifest["counts"], "files": statuses}, indent=2))


if __name__ == "__main__":
    main()
