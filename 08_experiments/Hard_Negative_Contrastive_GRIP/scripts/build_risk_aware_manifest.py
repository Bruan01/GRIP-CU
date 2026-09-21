#!/usr/bin/env python3
"""Build a risk-aware hard-negative manifest for A1.

The input is the immutable JSONL manifest produced by
``mine_score_hard_negatives.py``.  For each relation-like QA item, this script
keeps the same train-graph relation vocabulary and known-pair exclusions, but
samples negatives with a ProGCL/RotatE-inspired score:

    sampling_weight = exp(model_score / temperature) * (1 - risk)

Here ``risk`` is the Jaccard overlap between the candidate relation's observed
train tails and the positive relation's observed train tails.  High overlap is
a conservative proxy for type/semantic equivalence and therefore receives a
lower negative weight.  The script never turns a known train/valid/test fact
into a negative; the input manifest already records those pair-level facts and
this script rechecks them.

The output preserves the score-hard manifest schema so the existing
``--listed_negative_source score_hard`` training path can consume it without
changing the baseline snapshot.  It also records risk and weight provenance
for auditing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="B1 score-hard JSONL")
    parser.add_argument("--raw-dir", type=Path, required=True, help="NELL23K raw split directory")
    parser.add_argument("--output", type=Path, required=True, help="Risk-aware JSONL output")
    parser.add_argument("--total-k", type=int, default=9)
    parser.add_argument("--hard-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--risk-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def load_train_triples(raw_dir: Path) -> list[tuple[str, str, str]]:
    path = raw_dir / "train.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    triples: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            continue
        triples.append((fields[0], fields[1], fields[2]))
    if not triples:
        raise ValueError(f"{path} contains no triples")
    return triples


def tail_sets(triples: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for _, relation, tail in triples:
        result.setdefault(relation, set()).add(tail)
    return result


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def stable_weights(
    candidates: list[str],
    scores: dict[str, float],
    risks: dict[str, float],
    *,
    temperature: float,
    risk_floor: float,
) -> dict[str, float]:
    logits = [
        float(scores[relation]) / temperature
        + math.log(max(1.0 - risks.get(relation, 0.0), risk_floor))
        for relation in candidates
    ]
    maximum = max(logits)
    exponentials = {relation: math.exp(logit - maximum) for relation, logit in zip(candidates, logits)}
    total = sum(exponentials.values())
    if not math.isfinite(total) or total <= 0:
        return {relation: 1.0 / len(candidates) for relation in candidates}
    return {relation: value / total for relation, value in exponentials.items()}


def weighted_sample_without_replacement(
    candidates: list[str], weights: dict[str, float], k: int, rng: random.Random
) -> list[str]:
    remaining = list(candidates)
    chosen: list[str] = []
    for _ in range(min(k, len(remaining))):
        total = sum(max(float(weights.get(item, 0.0)), 0.0) for item in remaining)
        if total <= 0 or not math.isfinite(total):
            index = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            index = 0
            for index, item in enumerate(remaining):
                threshold -= max(float(weights.get(item, 0.0)), 0.0)
                if threshold <= 0:
                    break
        chosen.append(remaining.pop(index))
    return chosen


def main() -> None:
    args = parse_args()
    if args.total_k < 0 or args.hard_k < 0 or args.hard_k > args.total_k:
        raise ValueError("require 0 <= hard_k <= total_k")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= args.risk_floor < 1:
        raise ValueError("risk-floor must be in [0, 1)")

    rows = load_jsonl(args.input)
    train_tails = tail_sets(load_train_triples(args.raw_dir))
    rng = random.Random(args.seed)
    output: list[dict] = []
    risk_values: list[float] = []

    for row in rows:
        gold = str(row.get("matched_train_relation") or row.get("positive_relation") or "")
        scores = {
            str(item["relation"]): float(item["score"])
            for item in row.get("all_candidate_scores", [])
            if isinstance(item, dict) and "relation" in item and "score" in item
        }
        if not gold or gold not in train_tails or not scores:
            raise ValueError(f"row {row.get('question_id')!r} lacks gold or candidate scores")
        known = {str(rel) for rel in row.get("known_pair_relations", [])}
        candidates = [
            relation for relation in scores
            if relation != gold and relation not in known
        ]
        if len(candidates) < args.total_k:
            raise ValueError(
                f"row {row.get('question_id')!r} has only {len(candidates)} safe candidates"
            )
        risks = {
            relation: jaccard(train_tails.get(relation, set()), train_tails[gold])
            for relation in candidates
        }
        weights = stable_weights(
            candidates,
            scores,
            risks,
            temperature=args.temperature,
            risk_floor=args.risk_floor,
        )
        selected = weighted_sample_without_replacement(candidates, weights, args.total_k, rng)
        hard = sorted(selected, key=lambda relation: (-scores[relation], relation))[:args.hard_k]
        uniform = [relation for relation in selected if relation not in set(hard)]
        enriched = dict(row)
        enriched.update(
            {
                "negative_relations": [*hard, *uniform],
                "hard_negative_relations": hard,
                "uniform_negative_relations": uniform,
                "negative_source": "a1_risk_aware_score_plus_sampling",
                "risk_temperature": args.temperature,
                "risk_floor": args.risk_floor,
                "risk_seed": args.seed,
                "candidate_risk": {
                    relation: round(risks[relation], 8) for relation in selected
                },
                "candidate_sampling_weight": {
                    relation: round(weights[relation], 8) for relation in selected
                },
            }
        )
        output.append(enriched)
        risk_values.extend(risks[relation] for relation in selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
    temporary.replace(args.output)

    mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    print(
        json.dumps(
            {
                "rows": len(output),
                "selected_negatives": len(risk_values),
                "mean_selected_risk": mean_risk,
                "temperature": args.temperature,
                "risk_floor": args.risk_floor,
                "seed": args.seed,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
