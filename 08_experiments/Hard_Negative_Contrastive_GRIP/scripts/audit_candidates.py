from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hard_negative_grip import generate_hard_negatives, generate_random_negatives


def _triples(record: dict) -> list[tuple[str, str, str]]:
    graph = record["graph"]
    return [tuple(edge) for edge in graph["edge_list"]]


def _candidate_record(candidate) -> dict:
    data = asdict(candidate)
    data["triple"] = list(candidate.triple)
    return data


def audit_record(record: dict, num_per_kind: int, path_hops: int, seed: int) -> dict:
    graph_triples = _triples(record)
    all_questions = record.get("recurrent_questions", [])
    all_known = list(graph_triples)
    for sample in all_questions:
        source = sample.get("source_node")
        relation = sample.get("answer")
        target = sample.get("target_node")
        if source is not None and relation is not None and target is not None:
            all_known.append((str(source), str(relation), str(target)))
    relations = sorted({triple[1] for triple in graph_triples})
    counts: dict[str, int] = {
        "uniform_relation": 0,
        "tail_range_relation": 0,
        "path_relation": 0,
        "random": 0,
    }
    protected = set(all_known)
    invalid = 0
    rows = []
    for sample in all_questions:
        source = sample.get("source_node")
        relation = sample.get("answer")
        target = sample.get("target_node")
        if None in (source, relation, target):
            continue
        head = str(source)
        positive_relation = str(relation)
        tail = str(target)
        hard = generate_hard_negatives(
            positive_relation=positive_relation,
            head=head,
            tail=tail,
            relations=relations,
            graph_triples=graph_triples,
            all_known_triples=all_known,
            num_per_kind=num_per_kind,
            path_hops=path_hops,
        )
        random_negatives = generate_random_negatives(
            positive_relation=positive_relation,
            head=head,
            tail=tail,
            relations=relations,
            graph_triples=graph_triples,
            all_known_triples=all_known,
            num_negatives=num_per_kind,
            seed=seed + len(rows),
        )
        candidates = hard + random_negatives
        for candidate in candidates:
            counts[candidate.kind] += 1
            if (candidate.head, candidate.relation, candidate.tail) in protected:
                invalid += 1
            if candidate.relation == positive_relation:
                invalid += 1
        rows.append(
            {
                "question_id": sample.get("question_id"),
                "positive_relation": positive_relation,
                "head": head,
                "tail": tail,
                "hard_candidates": [_candidate_record(candidate) for candidate in hard],
                "random_candidates": [_candidate_record(candidate) for candidate in random_negatives],
            }
        )
    return {
        "graph_id": record.get("id", record.get("title", "graph")),
        "question_count": len(rows),
        "candidate_counts": counts,
        "invalid_candidates": invalid,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--num_per_kind", type=int, default=4)
    parser.add_argument("--path_hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = [audit_record(record, args.num_per_kind, args.path_hops, args.seed) for record in records]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(result), "output_file": str(args.output_file)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
