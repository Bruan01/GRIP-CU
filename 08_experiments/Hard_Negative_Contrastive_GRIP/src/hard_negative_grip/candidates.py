from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import inf
import random
from typing import Iterable, Literal


Triple = tuple[str, str, str]
RelationNegativeKind = Literal[
    "uniform_relation",
    "tail_range_relation",
    "path_relation",
    "random",
]


@dataclass(frozen=True)
class Candidate:
    """A filtered wrong-relation answer candidate with provenance for audit.

    The first NELL23K experiment is relation prediction, so the candidate
    *answer* is a relation string (``concept:...``). ``relation`` is the wrong
    relation proposed as an alternative to ``source_relation``; ``head`` and
    ``tail`` record the queried entity pair so the corruption
    ``(head, relation, tail)`` can be reconstructed and re-filtered.
    """

    kind: RelationNegativeKind
    relation: str
    head: str
    source_relation: str
    tail: str
    structural_distance: int | None = None

    @property
    def triple(self) -> Triple:
        """Reconstruct the corrupted triple ``(head, relation, tail)``."""
        return (self.head, self.relation, self.tail)


def known_triples(*triple_sets: Iterable[Triple]) -> set[Triple]:
    """Return all facts that must be protected from negative sampling."""
    return {tuple(triple) for triple_set in triple_sets for triple in triple_set}


def build_adjacency(triples: Iterable[Triple]) -> dict[str, set[str]]:
    """Build an undirected entity adjacency map from KG triples."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for head, _, tail in triples:
        adjacency[head].add(tail)
        adjacency[tail].add(head)
    return dict(adjacency)


def _shortest_distances(adjacency: dict[str, set[str]], source: str) -> dict[str, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _relation_tail_sets(graph_triples: Iterable[Triple]) -> dict[str, set[str]]:
    tail_sets: dict[str, set[str]] = defaultdict(set)
    for _, relation, tail in graph_triples:
        tail_sets[relation].add(tail)
    return dict(tail_sets)


def _valid_negative_relations(
    relations: Iterable[str],
    positive_relation: str,
    head: str,
    tail: str,
    protected: set[Triple],
) -> list[str]:
    """Wrong relations that are not the positive relation and whose corruption
    ``(head, relation, tail)`` is not a known fact."""
    return sorted(
        relation
        for relation in set(relations)
        if relation != positive_relation and (head, relation, tail) not in protected
    )


def generate_random_negatives(
    positive_relation: str,
    *,
    head: str,
    tail: str,
    relations: Iterable[str],
    graph_triples: Iterable[Triple],
    all_known_triples: Iterable[Triple] = (),
    num_negatives: int = 4,
    seed: int = 2026,
) -> list[Candidate]:
    """Generate a deterministic uniform-corruption control set."""
    if num_negatives < 0:
        raise ValueError("num_negatives must be non-negative")
    protected = known_triples(graph_triples, all_known_triples)
    valid = _valid_negative_relations(relations, positive_relation, head, tail, protected)
    rng = random.Random(seed)
    rng.shuffle(valid)
    return [
        Candidate("random", relation, head, positive_relation, tail)
        for relation in valid[:num_negatives]
    ]


def generate_hard_negatives(
    positive_relation: str,
    *,
    head: str,
    tail: str,
    relations: Iterable[str],
    graph_triples: Iterable[Triple],
    all_known_triples: Iterable[Triple] = (),
    num_per_kind: int = 4,
    path_hops: int = 2,
) -> list[Candidate]:
    """Generate deterministic, filtered relation-level hard negatives.

    Families (all relations are wrong alternatives to ``positive_relation``
    and satisfy that ``(head, relation, tail)`` is not a known fact):

    - ``uniform_relation``: deterministic sorted relation, the low-structure
      relation control (original N1).
    - ``tail_range_relation``: a relation whose observed tail set overlaps the
      positive relation's tail set, so the two relations are type-confusable
      (redefined N2).
    - ``path_relation``: a relation observed on an edge incident to an entity
      within ``path_hops`` of ``head``, i.e. locally active near the query
      (redefined N3). ``structural_distance`` is the smallest graph distance
      from ``head`` to an endpoint of an edge carrying that relation.
    """
    if num_per_kind < 0:
        raise ValueError("num_per_kind must be non-negative")
    if path_hops < 1:
        raise ValueError("path_hops must be positive")
    graph_triples = list(graph_triples)
    protected = known_triples(graph_triples, all_known_triples)
    valid = _valid_negative_relations(
        relations, positive_relation, head, tail, protected
    )
    output: list[Candidate] = []
    if num_per_kind == 0:
        return output

    for relation in valid[:num_per_kind]:
        output.append(Candidate("uniform_relation", relation, head, positive_relation, tail))

    tail_sets = _relation_tail_sets(graph_triples)
    positive_tails = tail_sets.get(positive_relation, set())
    tail_range = [
        relation
        for relation in valid
        if tail_sets.get(relation, set()) & positive_tails
    ][:num_per_kind]
    for relation in tail_range:
        output.append(Candidate("tail_range_relation", relation, head, positive_relation, tail))

    adjacency = build_adjacency(graph_triples)
    distances = _shortest_distances(adjacency, head)
    relation_min_distance: dict[str, int] = defaultdict(lambda: inf)
    for edge_head, relation, edge_tail in graph_triples:
        distance = min(distances.get(edge_head, inf), distances.get(edge_tail, inf))
        relation_min_distance[relation] = min(relation_min_distance[relation], distance)
    path_scored = sorted(
        (int(relation_min_distance[relation]), relation)
        for relation in valid
        if relation_min_distance[relation] <= path_hops
    )
    for distance, relation in path_scored[:num_per_kind]:
        output.append(
            Candidate(
                "path_relation",
                relation,
                head,
                positive_relation,
                tail,
                structural_distance=distance,
            )
        )
    return output
