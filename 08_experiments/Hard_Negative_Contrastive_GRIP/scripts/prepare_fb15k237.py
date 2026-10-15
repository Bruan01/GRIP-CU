#!/usr/bin/env python3
"""Prepare deterministic, CPU-only FB15K-237 official and recurrent JSONL.

Run with RecurrentGRIP's .venv/bin/python. Defaults write data/fb15k237 next
 to this experiment. Validation then test follow fb15k237_2/process.py:
 generate candidates for EVERY source triple, then choose 10,000 without
 replacement for that split. The intentional reproducibility deviation is
 sorting unique TRAIN RELATION TEXTS instead of converting a set to a list.
 Train-reference selection/candidates use independent RandomState instances;
 neither advances the shared validation/test RandomState.

All five dataset files contain one JSONL record. Smoke takes the first 96
 questions of each full split, with unchanged source IDs and candidates.
 Existing identical files are verified and retained; different files are
 atomically replaced. The manifest is written last and records output hashes.
 Structural paths are not computed: null true_hop is not a distance claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
WAY = 10
TITLE = "fb15k237_2"
QUESTION_TEMPLATE = (
    "What is the relation between node {src} and node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)
REQUIRED_FILES = (
    "entity2wikidata.json", "rel2text.json", "train.txt", "valid.txt", "test.txt"
)
CANDIDATE_SOURCE = "official_grip_fb15k237_sorted_train_relation_texts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected exactly three fields")
            triples.append(tuple(fields))
    return triples


def official_entity_text(value: dict) -> str:
    """Keep the original processor's punctuation, whitespace, and fallback."""
    entity = ""
    if value["label"] is not None:
        entity += value["label"] + ", "
    if value["description"] is not None:
        entity += value["description"]
    if not entity:
        entity = "missing"
    return entity.strip().strip(".")


def sample_candidates(answer: str, relation_order: list[str], rng) -> list[str]:
    pool = relation_order.copy()
    pool.remove(answer)
    negatives = rng.permutation(pool)[:WAY - 1]
    return [str(item) for item in rng.permutation(negatives.tolist() + [answer])]


def build_question(
    split: str, source_index: int, triple: tuple[str, str, str],
    entity_text: dict[str, str], entity_ids: dict[str, int],
    relation_text: dict[str, str], candidates: list[str],
) -> dict:
    source, relation, target = triple
    return {
        "question_id": f"fb15k237:{split}:{source_index}",
        "question": QUESTION_TEMPLATE.format(
            src=entity_text[source], tgt=entity_text[target],
            candidates="; ".join(candidates),
        ),
        "answer": relation_text[relation],
        "split": split,
        "true_hop": None,
        "question_type": "FB15K237RelationPrediction",
        "source_node": entity_text[source],
        "target_node": entity_text[target],
        "source_entity_id": source,
        "target_entity_id": target,
        "source_relation_id": relation,
        "node_indices": [entity_ids[source], entity_ids[target]],
        "candidate_relations": candidates,
        "candidate_source": CANDIDATE_SOURCE,
        "source_file": {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}[split],
        "source_triple_index": int(source_index),
    }


def official_record(graph: dict, samples: list[dict]) -> dict:
    return {
        "title": TITLE,
        "graph": graph,
        "questions": [[q["question"], q["node_indices"]] for q in samples],
        "answers": [q["answer"] for q in samples],
        "id": "0",
    }


def prepare_fb15k237(
    raw_dir: Path, *, seed: int = SEED, train_questions: int = 2000,
    eval_questions: int = 10000, smoke_questions: int = 96,
) -> tuple[dict[str, dict], dict]:
    if min(train_questions, eval_questions, smoke_questions) < 1:
        raise ValueError("all question counts must be positive")
    if smoke_questions > min(train_questions, eval_questions):
        raise ValueError("smoke_questions must fit in every full split")
    required = {name: raw_dir / name for name in REQUIRED_FILES}
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # Hash before reading and recheck afterwards to reject a changing source.
    source_hashes = {name: sha256_file(path) for name, path in required.items()}
    optional = raw_dir / "fb15k237.json"
    if optional.is_file():
        source_hashes[optional.name] = sha256_file(optional)
    entity_data = json.loads(required["entity2wikidata.json"].read_text(encoding="utf-8"))
    relation_text = json.loads(required["rel2text.json"].read_text(encoding="utf-8"))
    triples = {
        split: load_triples(required[filename])
        for split, filename in (("train", "train.txt"), ("validation", "valid.txt"), ("test", "test.txt"))
    }
    for split, count in (("train", train_questions), ("validation", eval_questions), ("test", eval_questions)):
        if count > len(triples[split]):
            raise ValueError(f"{split}: requested {count}, only {len(triples[split])} triples")
    entity_text = {key: official_entity_text(value) for key, value in entity_data.items()}
    entity_ids = {key: index for index, key in enumerate(entity_text)}
    for source, _, target in triples["train"]:
        for entity in (source, target):
            if entity not in entity_ids:
                entity_ids[entity] = len(entity_ids)
                entity_text[entity] = "Unknown"
    # The official graph freezes node_list before evaluation. Reject unknown
    # held-out entities rather than silently emitting out-of-range graph IDs.
    missing_eval = sorted({
        entity for split in ("validation", "test")
        for source, _, target in triples[split] for entity in (source, target)
        if entity not in entity_ids
    })
    if missing_eval:
        raise ValueError(f"held-out entities absent from official train graph nodes: {missing_eval[:10]}")
    relation_order = sorted({relation_text[rel] for _, rel, _ in triples["train"]})
    if len(relation_order) < WAY:
        raise ValueError(f"need at least {WAY} distinct training relation texts")
    for split in ("validation", "test"):
        unknown = {relation_text[rel] for _, rel, _ in triples[split]} - set(relation_order)
        if unknown:
            raise ValueError(f"{split}: relations absent from training text pool: {sorted(unknown)}")
    graph = {
        "edge_index": [[entity_ids[src], entity_ids[tgt]] for src, _, tgt in triples["train"]],
        "edge_list": [[entity_text[src], relation_text[rel], entity_text[tgt]] for src, rel, tgt in triples["train"]],
        "node_list": list(entity_text.values()),
    }

    # Separate train selection and candidate streams keep prefixes stable even
    # when the number of train references changes. Neither touches eval_rng.
    train_selection_rng = np.random.RandomState(seed)
    train_candidate_rng = np.random.RandomState(seed)
    train_indices = train_selection_rng.permutation(len(triples["train"]))[:train_questions]
    selected = {"train": []}
    for index in train_indices:
        triple = triples["train"][int(index)]
        candidates = sample_candidates(relation_text[triple[1]], relation_order, train_candidate_rng)
        selected["train"].append(build_question(
            "train", int(index), triple, entity_text, entity_ids, relation_text, candidates,
        ))

    eval_rng = np.random.RandomState(seed)
    for split in ("validation", "test"):
        all_questions = []
        for index, triple in enumerate(triples[split]):
            candidates = sample_candidates(relation_text[triple[1]], relation_order, eval_rng)
            all_questions.append(build_question(
                split, index, triple, entity_text, entity_ids, relation_text, candidates,
            ))
        # Preserve official np.random.choice ordering: do not sort this array.
        indices = eval_rng.choice(len(all_questions), eval_questions, replace=False)
        selected[split] = [all_questions[int(index)] for index in indices]

    metadata = {
        "task": "FB15K237RelationPrediction",
        "graph_source": "train.txt",
        "train_qa_source": "train.txt",
        "validation_source": "valid.txt",
        "test_source": "test.txt",
        "seed": seed,
        "num_candidates": WAY,
        "candidate_source": CANDIDATE_SOURCE,
        "relation_order": "sorted unique train relation TEXTS",
        "structural_distance_status": "not computed; true_hop is null",
        "eval_protocol": "all valid candidates, valid subsample, all test candidates, test subsample",
        "train_rng": "independent RandomState(seed) instances for selection and candidates",
    }
    recurrent_base = {"title": TITLE, "id": "0", "dataset": "FB15K-237", "graph": graph}
    full = {
        **recurrent_base,
        "recurrent_questions": [q for split in selected for q in selected[split]],
        "dataset_metadata": {**metadata, "subset": "full"},
    }
    smoke = {
        **recurrent_base,
        "recurrent_questions": [q for split in selected for q in selected[split][:smoke_questions]],
        "dataset_metadata": {**metadata, "subset": "smoke", "smoke_selection": "first N per split from full", "smoke_questions_per_split": smoke_questions},
    }
    outputs = {
        "processed_train.json": official_record(graph, selected["train"]),
        "processed_val.json": official_record(graph, selected["validation"]),
        "processed_test.json": official_record(graph, selected["test"]),
        "recurrent_relation_prediction_full.json": full,
        "recurrent_relation_prediction_smoke.json": smoke,
    }
    verify_records(outputs, triples, entity_text, entity_ids, relation_text, relation_order)
    for name, expected in source_hashes.items():
        if sha256_file(raw_dir / name) != expected:
            raise RuntimeError(f"source changed during preparation: {name}")
    manifest = {
        "format_version": 1,
        "dataset": "FB15K-237",
        "source": "WFRaain/TAG_datasets/FB15K237",
        "source_sha256": source_hashes,
        "unused_source_files": ["fb15k237.json"] if optional.is_file() else [],
        "seed": seed,
        "rng": "numpy.random.RandomState (MT19937)",
        "numpy_version": np.__version__,
        "num_candidates": WAY,
        "relation_order": relation_order,
        "official_processor": "data/raw_datasets/fb15k237_2/process.py",
        "deterministic_deviation": "The official processor uses list(set(train relation texts)), whose order is unspecified. We freeze sorted(unique train relation texts). Exact candidates need not match an arbitrary original process run.",
        "eval_rng_schedule": ["valid: all source triples candidates", f"valid: choice({eval_questions}, replace=False)", "test: all source triples candidates", f"test: choice({eval_questions}, replace=False)"],
        "eval_questions_per_split": eval_questions,
        "train_rng": metadata["train_rng"],
        "train_selection": "RandomState(seed).permutation(train count)[:train_questions]",
        "smoke_selection": "first N per split from full; no candidate resampling",
        "question_id_definition": "fb15k237:{train|validation|test}:{zero-based source triple index}; never a subsample index",
        "entity_text_policy": "official label + ', ' + description; fallback missing; strip().strip('.'); unknown TRAIN entities append as Unknown; reject unknown eval entities",
        "structural_distance_status": metadata["structural_distance_status"],
        "counts": {
            "source_triples": {split: len(rows) for split, rows in triples.items()},
            "graph_nodes": len(entity_ids),
            "graph_edges": len(graph["edge_list"]),
            "train_relation_ids": len({rel for _, rel, _ in triples["train"]}),
            "train_relation_texts": len(relation_order),
            "train_unknown_entities": len(entity_ids) - len(entity_data),
            "full_questions": {split: len(rows) for split, rows in selected.items()},
            "smoke_questions": {split: smoke_questions for split in selected},
        },
        "verification": {
            "train_only_graph_and_references": True,
            "source_triple_ids_and_entity_text": True,
            "ten_unique_train_text_candidates_including_gold": True,
            "official_and_recurrent_questions_match": True,
            "smoke_identical_subset_of_full": True,
            "source_hashes_unchanged_during_preparation": True,
        },
    }
    return outputs, manifest


def verify_records(outputs, triples, entity_text, entity_ids, relation_text, relation_order) -> None:
    """Fail before writing if graph, provenance, candidates, or subsets drift."""
    full = outputs["recurrent_relation_prediction_full.json"]
    graph = full["graph"]
    expected_edges = [[entity_text[src], relation_text[rel], entity_text[tgt]] for src, rel, tgt in triples["train"]]
    if graph["edge_list"] != expected_edges:
        raise ValueError("graph is not exactly the training triples")
    if graph["edge_index"] != [[entity_ids[src], entity_ids[tgt]] for src, _, tgt in triples["train"]]:
        raise ValueError("graph edge indices do not match training triples")
    allowed = set(relation_order)
    by_split = {split: [] for split in triples}
    by_id = {}
    for sample in full["recurrent_questions"]:
        split = sample["split"]
        index = sample["source_triple_index"]
        src, rel, tgt = triples[split][index]
        if sample["question_id"] != f"fb15k237:{split}:{index}":
            raise ValueError("unstable source question ID")
        if sample["question_id"] in by_id:
            raise ValueError("duplicate source question ID")
        by_id[sample["question_id"]] = sample
        if (sample["source_node"], sample["answer"], sample["target_node"]) != (entity_text[src], relation_text[rel], entity_text[tgt]):
            raise ValueError("question does not match source triple text")
        candidates = sample["candidate_relations"]
        if len(candidates) != WAY or len(set(candidates)) != WAY or not set(candidates) <= allowed or candidates.count(sample["answer"]) != 1:
            raise ValueError("invalid gold + nine uniform candidates")
        if sample["question"] != QUESTION_TEMPLATE.format(src=entity_text[src], tgt=entity_text[tgt], candidates="; ".join(candidates)):
            raise ValueError("question text and candidate metadata disagree")
        if sample["node_indices"] != [entity_ids[src], entity_ids[tgt]]:
            raise ValueError("question node indices disagree")
        by_split[split].append(sample)
    for split, filename in (("train", "processed_train.json"), ("validation", "processed_val.json"), ("test", "processed_test.json")):
        if outputs[filename] != official_record(graph, by_split[split]):
            raise ValueError(f"official/recurrent mismatch: {filename}")
    smoke = outputs["recurrent_relation_prediction_smoke.json"]
    for sample in smoke["recurrent_questions"]:
        if by_id.get(sample["question_id"]) != sample:
            raise ValueError("smoke resampled a full question")
    n = smoke["dataset_metadata"]["smoke_questions_per_split"]
    if smoke["recurrent_questions"] != [q for split in by_split for q in by_split[split][:n]]:
        raise ValueError("smoke is not the per-split full prefix")
    if Counter(q["split"] for q in smoke["recurrent_questions"]) != Counter({split: n for split in by_split}):
        raise ValueError("incorrect smoke split counts")


def atomic_write_json(path: Path, payload: dict, *, jsonl: bool) -> tuple[dict, str]:
    """Stage in destination filesystem; preserve bytes/mtime if already equal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            staged = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=None if jsonl else 2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(staged)
        size = staged.stat().st_size
        if path.is_file() and sha256_file(path) == digest:
            status = "verified_identical"
        else:
            os.replace(staged, path)
            status = "written"
        return {"sha256": digest, "bytes": size}, status
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=ROOT / "data/raw_datasets/fb15k237")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data/fb15k237")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_questions", type=int, default=2000)
    parser.add_argument("--eval_questions", type=int, default=10000, help="Official value is 10000; smaller values support CPU fixtures")
    parser.add_argument("--smoke_questions", type=int, default=96)
    args = parser.parse_args()
    outputs, manifest = prepare_fb15k237(
        args.raw_dir, seed=args.seed, train_questions=args.train_questions,
        eval_questions=args.eval_questions, smoke_questions=args.smoke_questions,
    )
    manifest["outputs"] = {}
    statuses = {}
    for name, record in outputs.items():
        manifest["outputs"][name], statuses[name] = atomic_write_json(args.out_dir / name, record, jsonl=True)
    manifest_path = args.out_dir / "preparation_manifest.json"
    _, statuses[manifest_path.name] = atomic_write_json(manifest_path, manifest, jsonl=False)
    print(json.dumps({"output_dir": str(args.out_dir.resolve()), "counts": manifest["counts"], "files": statuses, "manifest": str(manifest_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
