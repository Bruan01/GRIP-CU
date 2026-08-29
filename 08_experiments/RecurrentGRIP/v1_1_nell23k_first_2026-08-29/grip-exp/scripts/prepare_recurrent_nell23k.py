"""Build a deterministic NELL23K-first input for RecurrentGRIP.

The original GRIP NELL23K processor exposes one training graph and relation-
prediction questions.  This adapter keeps the official train/valid/test roles:
train triples define the graph and supervised recurrent QA, validation triples
remain validation questions, and test triples remain final test questions.

For mechanism analysis, the script computes the undirected shortest-path
length between each queried entity pair in the *training graph*.  This is a
structural-distance diagnostic, not a claim that the relation label is fully
determined by that path.  Unreachable and self-pair queries retain
``true_hop=None`` and are still included in aggregate benchmark accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from pathlib import Path
from typing import Iterable


QUESTION_TEMPLATE = (
    "What is the relation between word node {src} and word node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)
DEFAULT_QUESTION_TYPE = "NELL23KRelationPrediction"


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, found {len(fields)}")
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def load_entities(path: Path, triples: Iterable[tuple[str, str, str]]) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    entities = list(payload)
    known = set(entities)
    for source, _, target in triples:
        if source not in known:
            entities.append(source)
            known.add(source)
        if target not in known:
            entities.append(target)
            known.add(target)
    return entities


def build_undirected_adjacency(
    triples: Iterable[tuple[str, str, str]],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for source, _, target in triples:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return adjacency


def shortest_path(
    adjacency: dict[str, set[str]], source: str, target: str
) -> list[str] | None:
    if source == target:
        return [source]
    if source not in adjacency or target not in adjacency:
        return None
    queue = deque([source])
    parent: dict[str, str | None] = {source: None}
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == target:
                path = [target]
                cursor = target
                while parent[cursor] is not None:
                    cursor = parent[cursor]  # type: ignore[assignment]
                    path.append(cursor)
                return list(reversed(path))
            queue.append(neighbor)
    return None


def _selected_indices(size: int, limit: int, rng: random.Random) -> list[int]:
    if limit <= 0 or limit >= size:
        return list(range(size))
    return sorted(rng.sample(range(size), k=limit))


def _candidate_relations(
    answer: str,
    relation_vocabulary: list[str],
    num_candidates: int,
    rng: random.Random,
) -> list[str]:
    if answer not in relation_vocabulary:
        raise ValueError(f"answer relation is absent from the training vocabulary: {answer}")
    negatives = [relation for relation in relation_vocabulary if relation != answer]
    requested_negatives = max(0, min(num_candidates - 1, len(negatives)))
    candidates = rng.sample(negatives, k=requested_negatives) + [answer]
    rng.shuffle(candidates)
    return candidates


def build_questions(
    *,
    triples: list[tuple[str, str, str]],
    split: str,
    relation_vocabulary: list[str],
    adjacency: dict[str, set[str]],
    limit: int,
    num_candidates: int,
    seed: int,
) -> tuple[list[dict], dict]:
    selection_rng = random.Random(f"{seed}:{split}:selection")
    candidate_rng = random.Random(f"{seed}:{split}:candidates")
    selected = _selected_indices(len(triples), limit, selection_rng)
    questions: list[dict] = []
    distance_counts: dict[str, int] = {}
    for source_index in selected:
        source, answer, target = triples[source_index]
        candidates = _candidate_relations(
            answer, relation_vocabulary, num_candidates, candidate_rng
        )
        path = shortest_path(adjacency, source, target)
        distance = len(path) - 1 if path is not None else None
        true_hop = distance if distance is not None and distance >= 1 else None
        distance_key = str(distance) if distance is not None else "unreachable"
        distance_counts[distance_key] = distance_counts.get(distance_key, 0) + 1
        questions.append(
            {
                "question_id": f"nell23k:{split}:{source_index}",
                "question": QUESTION_TEMPLATE.format(
                    src=source,
                    tgt=target,
                    candidates="; ".join(candidates),
                ),
                "answer": answer,
                "split": split,
                "true_hop": true_hop,
                "question_type": DEFAULT_QUESTION_TYPE,
                "source_node": source,
                "target_node": target,
                "shortest_path": path or [],
                "structural_distance": distance,
                "structural_reachable": path is not None,
                "candidate_relations": candidates,
                "source_triple_index": source_index,
            }
        )
    return questions, {
        "available": len(triples),
        "selected": len(questions),
        "distance_counts": distance_counts,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_nell23k(
    *,
    raw_dir: Path,
    max_train_questions: int,
    max_validation_questions: int,
    max_test_questions: int,
    num_candidates: int,
    seed: int,
) -> tuple[list[dict], dict]:
    required = {
        name: raw_dir / name
        for name in ("train.txt", "valid.txt", "test.txt", "entity2text.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing NELL23K files: " + ", ".join(missing))
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")

    train_triples = load_triples(required["train.txt"])
    validation_triples = load_triples(required["valid.txt"])
    test_triples = load_triples(required["test.txt"])
    all_triples = train_triples + validation_triples + test_triples
    entities = load_entities(required["entity2text.json"], all_triples)
    entity_to_index = {entity: index for index, entity in enumerate(entities)}
    relation_vocabulary = sorted({relation for _, relation, _ in train_triples})
    unknown_eval_relations = sorted(
        {
            relation
            for _, relation, _ in validation_triples + test_triples
            if relation not in relation_vocabulary
        }
    )
    if unknown_eval_relations:
        raise ValueError(
            "validation/test relations absent from training vocabulary: "
            + ", ".join(unknown_eval_relations)
        )

    adjacency = build_undirected_adjacency(train_triples)
    edge_list = [[source, relation, target] for source, relation, target in train_triples]
    edge_index = [
        [entity_to_index[source], entity_to_index[target]]
        for source, _, target in train_triples
    ]
    graph = {
        "edge_index": edge_index,
        "edge_list": edge_list,
        "node_list": entities,
    }

    split_specs = (
        ("train", train_triples, max_train_questions),
        ("validation", validation_triples, max_validation_questions),
        ("test", test_triples, max_test_questions),
    )
    recurrent_questions: list[dict] = []
    split_stats: dict[str, dict] = {}
    for split, triples, limit in split_specs:
        questions, stats = build_questions(
            triples=triples,
            split=split,
            relation_vocabulary=relation_vocabulary,
            adjacency=adjacency,
            limit=limit,
            num_candidates=num_candidates,
            seed=seed,
        )
        recurrent_questions.extend(questions)
        split_stats[split] = stats

    record = {
        "title": "nell23k",
        "id": "nell23k",
        "dataset": "NELL23K",
        "graph": graph,
        "recurrent_questions": recurrent_questions,
        "dataset_metadata": {
            "task": DEFAULT_QUESTION_TYPE,
            "graph_source": "train.txt",
            "train_qa_source": "train.txt",
            "validation_source": "valid.txt",
            "test_source": "test.txt",
            "distance_definition": "undirected shortest path in the train graph",
            "distance_interpretation": (
                "structural diagnostic only; it does not prove that the relation label "
                "is determined by the shortest path"
            ),
            "seed": seed,
            "num_candidates": num_candidates,
        },
    }
    stats = {
        "dataset": "NELL23K",
        "graph": {
            "nodes": len(entities),
            "edges": len(train_triples),
            "relations": len(relation_vocabulary),
        },
        "splits": split_stats,
        "source_sha256": {
            name: sha256_file(path) for name, path in required.items()
        },
        "seed": seed,
        "num_candidates": num_candidates,
    }
    return [record], stats


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw_datasets/nell23k")
    parser.add_argument(
        "--output_file",
        default="outputs/data/nell23k/recurrent_relation_prediction.json",
    )
    parser.add_argument("--max_train_questions", type=int, default=64)
    parser.add_argument("--max_validation_questions", type=int, default=32)
    parser.add_argument("--max_test_questions", type=int, default=64)
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    rows, stats = prepare_nell23k(
        raw_dir=Path(args.raw_dir),
        max_train_questions=args.max_train_questions,
        max_validation_questions=args.max_validation_questions,
        max_test_questions=args.max_test_questions,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    save_jsonl(output_path, rows)
    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "output_file": str(output_path),
                "stats_file": str(stats_path),
                "stats": stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
