"""Build a fixed entity vocabulary from the NELL23K training graph."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl, sha256_file, write_jsonl


def parse_train_graph(path: Path) -> tuple[list[tuple[str, str, str]], set[str], set[str]]:
    triples: list[tuple[str, str, str]] = []
    entities: set[str] = set()
    relations: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not all(part.strip() for part in parts):
                raise ValueError(f"malformed triple at {path}:{number}")
            head, relation, tail = (part.strip() for part in parts)
            triples.append((head, relation, tail))
            entities.update((head, tail))
            relations.add(relation)
    if not triples:
        raise ValueError(f"empty training graph: {path}")
    return triples, entities, relations


def build_train_entity_vocabulary(train_graph: Path, output_path: Path) -> dict:
    triples, entity_set, relations = parse_train_graph(train_graph)
    entities = sorted(entity_set)
    write_jsonl(output_path, ({"index": index, "entity": entity} for index, entity in enumerate(entities)))
    return {
        "source_path": str(train_graph),
        "source_split": "train",
        "source_sha256": sha256_file(train_graph),
        "triple_count": len(triples),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def read_entities(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
        entities = [str(row["entity"]) for row in sorted(rows, key=lambda row: int(row["index"]))]
    else:
        entities = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entities or len(entities) != len(set(entities)):
        raise ValueError(f"entity vocabulary must be non-empty and unique: {path}")
    return entities


def audit_answer_coverage(entities: Iterable[str], rows: Iterable[dict]) -> dict:
    vocabulary = set(map(str, entities))
    answers = [str(row["answer"]) for row in rows]
    missing = sorted(set(answers) - vocabulary)
    covered = sum(answer in vocabulary for answer in answers)
    return {
        "total": len(answers),
        "covered": covered,
        "coverage": covered / len(answers) if answers else 0.0,
        "missing_unique_count": len(missing),
        "missing_answers": missing,
    }
