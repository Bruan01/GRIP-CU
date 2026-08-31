"""Graph construction and bounded leave-one-edge-out distance computation."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Iterator

from .io_utils import Triple


@dataclass(frozen=True)
class PathStep:
    head: str
    relation: str
    tail: str


class KnowledgeGraph:
    """A relation-aware directed graph plus an undirected structural view."""

    def __init__(self, triples: Iterable[Triple]) -> None:
        self.triples = list(triples)
        self.directed: dict[str, set[str]] = defaultdict(set)
        self.undirected: dict[str, set[str]] = defaultdict(set)
        self.out_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.relation_out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.directed_pair_count: Counter[tuple[str, str]] = Counter()
        self.undirected_pair_count: Counter[tuple[str, str]] = Counter()
        self.triple_count: Counter[Triple] = Counter()
        self.relation_count: Counter[str] = Counter()
        self.nodes: set[str] = set()
        for head, relation, tail in self.triples:
            self.nodes.update((head, tail))
            self.directed[head].add(tail)
            self.undirected[head].add(tail)
            self.undirected[tail].add(head)
            self.out_edges[head].append((relation, tail))
            self.relation_out[head][relation].add(tail)
            self.directed_pair_count[(head, tail)] += 1
            self.undirected_pair_count[self._undirected_key(head, tail)] += 1
            self.triple_count[(head, relation, tail)] += 1
            self.relation_count[relation] += 1

    @staticmethod
    def _undirected_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left <= right else (right, left)

    def bounded_distance(
        self,
        source: str,
        target: str,
        *,
        max_depth: int,
        directed: bool,
        blocked_pair: tuple[str, str] | None = None,
    ) -> int | None:
        """Return shortest distance up to max_depth, optionally blocking one pair.

        The blocked pair removes adjacency only when the removed triple was the
        sole train edge for that node pair. Parallel relations remain valid
        one-step alternative support and are handled by leave_one_out_distance.
        """
        if source == target:
            return 0
        adjacency = self.directed if directed else self.undirected
        if source not in adjacency:
            return None
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        seen = {source}
        blocked_undirected = self._undirected_key(*blocked_pair) if blocked_pair and not directed else None
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adjacency.get(node, ()):
                if blocked_pair is not None:
                    if directed and (node, neighbor) == blocked_pair:
                        continue
                    if not directed and self._undirected_key(node, neighbor) == blocked_undirected:
                        continue
                next_depth = depth + 1
                if neighbor == target:
                    return next_depth
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, next_depth))
        return None

    def leave_one_out_distance(
        self,
        triple: Triple,
        *,
        max_depth: int,
        directed: bool,
    ) -> int | None:
        head, _relation, tail = triple
        if head == tail:
            return 0
        if directed:
            pair = (head, tail)
            if self.directed_pair_count[pair] > 1:
                return 1
        else:
            pair = self._undirected_key(head, tail)
            if self.undirected_pair_count[pair] > 1:
                return 1
        return self.bounded_distance(
            head,
            tail,
            max_depth=max_depth,
            directed=directed,
            blocked_pair=(head, tail),
        )

    def follow_relations(self, source: str, relations: Iterable[str], *, limit: int = 10_000) -> set[str]:
        current = {source}
        for relation in relations:
            next_nodes: set[str] = set()
            for node in current:
                next_nodes.update(self.relation_out.get(node, {}).get(relation, ()))
                if len(next_nodes) > limit:
                    return next_nodes
            current = next_nodes
            if not current:
                break
        return current

    def iter_simple_random_walk_choices(self, node: str, visited: set[str]) -> Iterator[tuple[str, str]]:
        for relation, tail in self.out_edges.get(node, ()):
            if tail not in visited:
                yield relation, tail
