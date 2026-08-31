"""Deterministic generation of provenance-preserving exact-hop path QA."""
from __future__ import annotations

import random
from collections import Counter

from .graph import KnowledgeGraph


def render_prompt(head: str, relations: list[str]) -> str:
    chain = " -> ".join(relations)
    return (
        f"Starting from entity {head}, follow this ordered relation chain: {chain}. "
        "Which entity is reached? Return only the entity identifier."
    )


def sample_exact_hop_tasks(
    graph: KnowledgeGraph,
    *,
    per_depth: int,
    max_depth: int,
    seed: int,
    max_attempts_per_depth: int = 500_000,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    starts = sorted(graph.out_edges)
    tasks: list[dict] = []
    attempts_by_depth: dict[str, int] = {}
    rejection_counts: Counter[str] = Counter()
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    for depth in range(1, max_depth + 1):
        accepted = 0
        attempts = 0
        while accepted < per_depth and attempts < max_attempts_per_depth:
            attempts += 1
            head = rng.choice(starts)
            node = head
            nodes = [head]
            relations: list[str] = []
            valid_walk = True
            for _ in range(depth):
                choices = list(graph.iter_simple_random_walk_choices(node, set(nodes)))
                if not choices:
                    rejection_counts["dead_end_or_cycle"] += 1
                    valid_walk = False
                    break
                relation, node = rng.choice(choices)
                relations.append(relation)
                nodes.append(node)
            if not valid_walk:
                continue
            tail = nodes[-1]
            shortest = graph.bounded_distance(head, tail, max_depth=depth, directed=True)
            if shortest != depth:
                rejection_counts["shorter_directed_path"] += 1
                continue
            answers = graph.follow_relations(head, relations)
            if answers != {tail}:
                rejection_counts["non_unique_relation_chain"] += 1
                continue
            signature = (head, tuple(relations), tail)
            if signature in seen:
                rejection_counts["duplicate_signature"] += 1
                continue
            seen.add(signature)
            task_id = f"nell23k:exact_path:d{depth}:{accepted:05d}"
            tasks.append(
                {
                    "task_id": task_id,
                    "task_type": "path_qa",
                    "text": render_prompt(head, relations),
                    "response": tail,
                    "answer": tail,
                    "answers": [tail],
                    "depth_label": depth,
                    "depth_source": "exact_directed_shortest_path",
                    "path_nodes": nodes,
                    "path_relations": relations,
                    "path_edges": [
                        {"head": nodes[i], "relation": relations[i], "tail": nodes[i + 1]}
                        for i in range(depth)
                    ],
                    "shortest_path_verified": True,
                    "unique_relation_chain_answer": True,
                    "seed": seed,
                }
            )
            accepted += 1
        attempts_by_depth[str(depth)] = attempts
        if accepted < per_depth:
            raise RuntimeError(
                f"Could only generate {accepted}/{per_depth} exact {depth}-hop tasks "
                f"after {attempts} attempts"
            )

    metadata = {
        "seed": seed,
        "per_depth": per_depth,
        "max_depth": max_depth,
        "total_tasks": len(tasks),
        "attempts_by_depth": attempts_by_depth,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return tasks, metadata
