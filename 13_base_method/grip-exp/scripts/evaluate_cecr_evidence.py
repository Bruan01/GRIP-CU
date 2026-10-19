"""Audit train-only CECR evidence coverage on NELL23K candidate splits.

This diagnostic does not load a language model.  It reports how often exact
training-pair evidence or learned two-hop composition evidence is available,
and the accuracy of an evidence-only candidate selector.  It is intended to
separate graph-coverage limits from LM-decoding limits before expensive 7B
runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grip.relation_cecr import (  # noqa: E402
    build_relation_memory_from_triples,
    parse_question_entities,
    parse_relation_candidates,
)


def load_record(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected one graph record in {path}, found {len(payload)}")
        return payload[0]
    return payload


def audit_split(memory, path: str | Path) -> dict:
    record = load_record(path)
    total = len(record["questions"])
    valid = target_in_candidates = target_supported = nonzero = evidence_only_hit = 0
    target_counts = Counter()
    target_supported_counts = Counter()
    evidence_hit_counts = Counter()
    margins = []
    for raw_question, target in zip(record["questions"], record["answers"]):
        question = raw_question[0] if isinstance(raw_question, list) else raw_question
        candidates = parse_relation_candidates(question)
        source, destination = parse_question_entities(question)
        scores = [memory.evidence_score(candidate, source, destination) for candidate in candidates]
        target_counts[target] += 1
        valid += int(bool(candidates))
        target_in_candidates += int(target in candidates)
        target_score = memory.evidence_score(target, source, destination)
        target_supported += int(target_score > 0)
        target_supported_counts[target] += int(target_score > 0)
        nonzero += int(max(scores, default=0.0) > 0)
        if candidates:
            best = max(range(len(candidates)), key=scores.__getitem__)
            hit = candidates[best] == target
            evidence_only_hit += int(hit)
            evidence_hit_counts[target] += int(hit)
            ordered = sorted(scores, reverse=True)
            margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])

    def ratio(n: int, d: int = total) -> float:
        return float(n) / max(1, d)

    return {
        "file": str(path),
        "samples": total,
        "candidate_valid_rate": ratio(valid),
        "target_in_candidate_rate": ratio(target_in_candidates),
        "target_evidence_coverage": ratio(target_supported),
        "any_evidence_rate": ratio(nonzero),
        "evidence_only_accuracy": ratio(evidence_only_hit),
        "mean_evidence_margin": sum(margins) / max(1, len(margins)),
        "per_relation": {
            relation: {
                "samples": count,
                "target_evidence_coverage": ratio(target_supported_counts[relation], count),
                "evidence_only_accuracy": ratio(evidence_hit_counts[relation], count),
            }
            for relation, count in target_counts.most_common()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-triples-file", default="outputs/data/nell23k/train.txt")
    parser.add_argument(
        "--split", nargs="+", default=["direct", "novel2hop", "patternnovel"],
        help="Split names resolved as outputs/data/nell23k/processed_test_<name>.json.",
    )
    parser.add_argument("--output-file", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    memory = build_relation_memory_from_triples(args.train_triples_file)
    report = {
        "train_triples_file": str(Path(args.train_triples_file).resolve()),
        "relation_cards": len(memory.cards),
        "train_entities": len(memory.entity_roles),
        "train_pairs": len(memory.pair_relations),
        "composition_patterns": len(memory.composition_patterns),
        "splits": {},
    }
    for split in args.split:
        path = Path("outputs/data/nell23k") / f"processed_test_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report["splits"][split] = audit_split(memory, path)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
