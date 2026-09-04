#!/usr/bin/env python3
"""Build and audit the immutable train-graph entity vocabulary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from entity_decoder.config import load_config
from entity_decoder.io_utils import write_json
from entity_decoder.records import load_split
from entity_decoder.vocabulary import audit_answer_coverage, build_train_entity_vocabulary, read_entities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase_a_decoder_smoke.json")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/entities_train_graph.jsonl")
    ap.add_argument("--audit", type=Path, default=ROOT / "artifacts/entity_vocabulary_audit.json")
    args = ap.parse_args()
    config = load_config(args.config.resolve())
    train_graph = REPO / config["data"]["train_graph"]
    output = args.output.resolve()
    audit = build_train_entity_vocabulary(train_graph, output)
    entities = read_entities(output)
    for split in ("train", "validation", "test"):
        rows = load_split(REPO / config["data"][split], split)
        audit[f"{split}_answer_coverage"] = audit_answer_coverage(entities, rows)
    expected = int(config["data"]["expected_entity_count"])
    if audit["entity_count"] != expected:
        raise ValueError(f"entity count changed: expected {expected}, found {audit['entity_count']}")
    expected_coverage = float(config["data"]["expected_test_answer_coverage"])
    if audit["test_answer_coverage"]["coverage"] != expected_coverage:
        raise ValueError("test answer coverage changed")
    write_json(args.audit.resolve(), audit)
    print(f"ENTITY_VOCABULARY_READY entities={audit['entity_count']} test_coverage={audit['test_answer_coverage']['coverage']:.6f}")


if __name__ == "__main__":
    main()
