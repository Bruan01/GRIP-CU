"""Analyze original GRIP NELL23K failures and emit an auditable JSON report.

The script is intentionally model-agnostic: it consumes a prediction JSONL
and optionally the evaluation JSON plus train graph.  It reports relation-level
failure rates, target->prediction confusions, candidate validity, and train
pair overlap so CECR ablations can be compared against the same baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grip.relation_cecr import parse_relation_candidates, parse_question_entities  # noqa: E402


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_target(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def load_train_pairs(path: str | Path | None) -> dict[tuple[str, str], set[str]]:
    pairs = defaultdict(set)
    if not path:
        return pairs
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                pairs[(fields[0], fields[2])].add(fields[1])
    return pairs


def relation_bucket(relation: str) -> str:
    relation = relation.lower()
    if relation in {"concept:subpartof", "concept:locationlocatedwithinlocation", "concept:proxyfor", "concept:mutualproxyfor"}:
        return "hierarchy_or_proxy"
    if relation in {"concept:animalsuchasinsect", "concept:animalsuchasinvertebrate", "concept:arthropodandotherarthropod", "concept:animalthatfeedoninsect", "concept:animalpreyson", "concept:animalistypeofanimal"}:
        return "animal_taxonomy_or_diet"
    if relation in {"concept:worksfor", "concept:personleadsorganization", "concept:organizationhiredperson", "concept:agentcontrols", "concept:agentcollaborateswithagent"}:
        return "employment_or_social"
    if "citylocated" in relation or "locatedwithin" in relation or "atlocation" in relation or "headquartered" in relation:
        return "location"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", default="outputs/grip_inf/nell23k/nell23k_qwen7b_direct_seed2026.jsonl")
    parser.add_argument("--train-triples-file", default="outputs/data/nell23k/train.txt")
    parser.add_argument("--output-file", default="artifacts/cecr/grip7b_failure_report.json")
    args = parser.parse_args()

    rows = load_jsonl(args.prediction_file)
    train_pairs = load_train_pairs(args.train_triples_file)
    relation_stats = defaultdict(lambda: Counter(samples=0, errors=0, target_pair_overlap=0))
    bucket_stats = defaultdict(lambda: Counter(samples=0, errors=0))
    confusions = Counter()
    candidate_lengths = Counter()
    failures = []
    out_of_candidate = 0
    candidate_total = 0

    for row in rows:
        question = row.get("question", "")
        target = normalize_target(row.get("target", ""))
        prediction = str(row.get("response", ""))
        source, destination = parse_question_entities(question)
        candidates = parse_relation_candidates(question)
        correct = prediction == target
        stats = relation_stats[target]
        stats["samples"] += 1
        stats["errors"] += int(not correct)
        stats["target_pair_overlap"] += int(target in train_pairs.get((source, destination), set()))
        bucket = bucket_stats[relation_bucket(target)]
        bucket["samples"] += 1
        bucket["errors"] += int(not correct)
        if candidates:
            candidate_total += 1
            candidate_lengths[len(candidates)] += 1
            out_of_candidate += int(prediction not in candidates)
        if not correct:
            confusions[(target, prediction)] += 1
            if len(failures) < 200:
                failures.append({
                    "id": row.get("id"),
                    "question": question,
                    "target": target,
                    "prediction": prediction,
                    "candidates": candidates,
                    "source": source,
                    "destination": destination,
                    "train_pair_relations": sorted(train_pairs.get((source, destination), set())),
                })

    def rate(value: int, total: int) -> float:
        return value / max(1, total)

    report = {
        "prediction_file": str(Path(args.prediction_file).resolve()),
        "train_triples_file": str(Path(args.train_triples_file).resolve()),
        "samples": len(rows),
        "errors": sum(1 for row in rows if str(row.get("response", "")) != normalize_target(row.get("target", ""))),
        "accuracy": rate(len(rows) - sum(1 for row in rows if str(row.get("response", "")) != normalize_target(row.get("target", ""))), len(rows)),
        "candidate_valid_rate": rate(candidate_total - out_of_candidate, candidate_total),
        "candidate_lengths": dict(sorted(candidate_lengths.items())),
        "relation_level": {
            relation: {
                "samples": values["samples"],
                "errors": values["errors"],
                "error_rate": rate(values["errors"], values["samples"]),
                "target_pair_overlap": values["target_pair_overlap"],
            }
            for relation, values in sorted(relation_stats.items(), key=lambda item: (-item[1]["errors"], item[0]))
        },
        "failure_buckets": {
            bucket: {
                "samples": values["samples"],
                "errors": values["errors"],
                "error_rate": rate(values["errors"], values["samples"]),
            }
            for bucket, values in sorted(bucket_stats.items())
        },
        "top_confusions": [
            {"target": target, "prediction": prediction, "count": count}
            for (target, prediction), count in confusions.most_common(100)
        ],
        "failure_examples": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
