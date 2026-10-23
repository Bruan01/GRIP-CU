"""Build a train-only NELL23K graph record for strict-paper GRIP task generation.

Official ``process.py`` writes only ``processed_val.json`` / ``processed_test.json``,
so the old NELL task file used validation questions as k-shot exemplars. GRIP
(Sec. 4.3) takes k-shot exemplars "from the training set (if have)". This writes
``processed_train.json`` with:

- ``graph``: built exactly like ``process.py`` (entity2text keys as node_list,
  every train triple as an edge), but reading every non-empty train line;
- ``questions`` / ``answers``: one relation question per train triple with the
  official question template and 10-way rule (train relation vocabulary only).

No validation or test triple is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    sample_official_candidates,
    train_relation_insertion_order,
)

Q_FORMAT = (
    "What is the relation between word node {src} and word node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)


def read_train_lines(path: Path) -> tuple[list[list[str]], bool]:
    """All non-empty train triples, plus whether ``process.py`` would drop the last one."""
    text = path.read_text(encoding="utf-8")
    rows = [line.split() for line in text.split("\n") if line.strip()]
    official_drops_last = not text.endswith("\n")
    for number, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise ValueError(f"{path}:{number}: expected 3 fields, got {len(row)}")
    return rows, official_drops_last


def build_record(raw_dir: Path, *, seed: int) -> tuple[dict, dict]:
    with (raw_dir / "entity2text.json").open(encoding="utf-8") as stream:
        entity_data = json.load(stream)
    entity2text = {key: (key.strip().strip(".") or "missing") for key in entity_data}
    entity2id = {entity: index for index, entity in enumerate(entity2text)}
    triples, official_drops_last = read_train_lines(raw_dir / "train.txt")
    edge_list, edge_index = [], []
    unknown = 0
    for src, rel, tgt in triples:
        for entity in (src, tgt):
            if entity not in entity2id:
                entity2text[entity] = "Unknown"
                entity2id[entity] = len(entity2id)
                unknown += 1
        edge_list.append([entity2text[src], rel, entity2text[tgt]])
        edge_index.append([entity2id[src], entity2id[tgt]])
    relation_order = train_relation_insertion_order([tuple(t) for t in triples])
    rng = np.random.RandomState(seed)
    questions, answers = [], []
    for src, rel, tgt in triples:
        candidates = sample_official_candidates(rel, relation_order, rng=rng)
        questions.append([
            Q_FORMAT.format(src=entity2text[src], tgt=entity2text[tgt], candidates="; ".join(candidates)),
            [entity2id[src], entity2id[tgt]],
        ])
        answers.append(rel)
    graph = {"edge_index": edge_index, "edge_list": edge_list, "node_list": list(entity2text.values())}
    record = {"title": "nell23k", "graph": graph, "questions": questions, "answers": answers, "id": "0"}
    endpoint_ids = {index for pair in edge_index for index in pair}
    manifest = {
        "raw_dir": str(raw_dir),
        "seed": seed,
        "train_sha256": hashlib.sha256((raw_dir / "train.txt").read_bytes()).hexdigest(),
        "train_triples": len(triples),
        "official_process_py_drops_last_train_line": official_drops_last,
        "nodes": len(graph["node_list"]),
        "nodes_in_train_edges": len(endpoint_ids),
        "unknown_entities": unknown,
        "relations": len(relation_order),
        "note": "Train split only; k-shot reference and task-generation graph for strict-paper B1.",
    }
    return record, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_NELL23K)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    record, manifest = build_record(args.raw_dir, seed=args.seed)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    # JSONL, one record per line: grip-exp ``load_list_json`` format.
    args.output_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = args.output_file.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
