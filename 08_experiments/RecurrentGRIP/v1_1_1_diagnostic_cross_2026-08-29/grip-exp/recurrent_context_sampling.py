"""Pure-Python graph-context sampling for RecurrentGRIP diagnostics.

The sampler keeps node declarations and edge facts as separate strata.  Edge
facts are selected relation-round-robin so a small context budget cannot be
silently consumed by the node prefix produced by ``GenGraphContextTask``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Iterable

Edge = tuple[str, str, str]


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_nodes(graph: dict, edges: Iterable[Edge]) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for value in graph.get("node_list", []):
        node = str(value)
        if node not in seen:
            nodes.append(node)
            seen.add(node)
    for source, _, target in edges:
        for node in (source, target):
            if node not in seen:
                nodes.append(node)
                seen.add(node)
    return nodes


def _clean_edges(graph: dict) -> tuple[list[Edge], int, int]:
    edges: list[Edge] = []
    seen: set[Edge] = set()
    duplicate_count = 0
    self_loop_count = 0
    for index, raw_edge in enumerate(graph.get("edge_list", [])):
        if len(raw_edge) != 3:
            raise ValueError(f"edge_list[{index}] must contain source, relation, target")
        edge = (str(raw_edge[0]), str(raw_edge[1]), str(raw_edge[2]))
        if edge[0] == edge[2]:
            self_loop_count += 1
            continue
        if edge in seen:
            duplicate_count += 1
            continue
        seen.add(edge)
        edges.append(edge)
    return edges, duplicate_count, self_loop_count


def _round_robin_edges(
    edges: list[Edge],
    limit: int,
    rng: random.Random,
    covered_relations: set[str] | None = None,
) -> list[Edge]:
    if limit <= 0 or not edges:
        return []
    buckets: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        buckets[edge[1]].append(edge)

    covered_relations = covered_relations or set()
    uncovered = sorted(relation for relation in buckets if relation not in covered_relations)
    covered = sorted(relation for relation in buckets if relation in covered_relations)
    rng.shuffle(uncovered)
    rng.shuffle(covered)
    relations = uncovered + covered
    for relation in relations:
        rng.shuffle(buckets[relation])

    selected: list[Edge] = []
    offsets = {relation: 0 for relation in relations}
    while len(selected) < min(limit, len(edges)):
        progressed = False
        for relation in relations:
            offset = offsets[relation]
            bucket = buckets[relation]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[relation] = offset + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _answers(sample: dict) -> set[str]:
    answer = sample.get("answer")
    if answer is None:
        return set()
    if isinstance(answer, (list, tuple, set)):
        return {str(value) for value in answer}
    return {str(answer)}


def _train_qa_facts(samples: list[dict], available_edges: set[Edge]) -> list[Edge]:
    facts: list[Edge] = []
    seen: set[Edge] = set()
    for sample in samples:
        if sample.get("split") != "train":
            continue
        source = sample.get("source_node")
        target = sample.get("target_node")
        if source is None or target is None:
            continue
        for relation in sorted(_answers(sample)):
            fact = (str(source), relation, str(target))
            if fact in available_edges and fact not in seen:
                facts.append(fact)
                seen.add(fact)
    return facts


def select_stratified_graph_context(
    graph: dict,
    node_quota: int,
    edge_quota: int,
    seed: int,
    train_samples: list[dict] | None = None,
) -> tuple[list[str], list[Edge], dict]:
    """Select deterministic node declarations and relation-balanced edge facts.

    ``node_quota`` and ``edge_quota`` define a total context budget.  If one
    stratum has fewer unique items than requested, the unused budget is filled
    from the other stratum, preferring edge facts when node declarations are
    exhausted.
    """

    if node_quota < 0 or edge_quota < 0:
        raise ValueError("node_quota and edge_quota cannot be negative")

    rng = random.Random(seed)
    edges, duplicate_count, self_loop_count = _clean_edges(graph)
    nodes = _unique_nodes(graph, edges)

    node_target = min(len(nodes), node_quota + max(0, edge_quota - len(edges)))
    edge_target = min(len(edges), edge_quota + max(0, node_quota - len(nodes)))

    shuffled_nodes = list(nodes)
    rng.shuffle(shuffled_nodes)
    selected_nodes = shuffled_nodes[:node_target]

    qa_samples = [
        sample for sample in (train_samples or []) if sample.get("split") == "train"
    ]
    priority_edges = _train_qa_facts(qa_samples, set(edges))
    if len(priority_edges) > edge_target:
        selected_edges = _round_robin_edges(priority_edges, edge_target, rng)
    else:
        selected_edges = list(priority_edges)
        selected_edge_set = set(selected_edges)
        remaining_edges = [edge for edge in edges if edge not in selected_edge_set]
        selected_edges.extend(
            _round_robin_edges(
                remaining_edges,
                edge_target - len(selected_edges),
                rng,
                covered_relations={relation for _, relation, _ in selected_edges},
            )
        )

    selected_relations = {relation for _, relation, _ in selected_edges}
    all_relations = {relation for _, relation, _ in edges}
    selected_context_entities = set(selected_nodes)
    for source, _, target in selected_edges:
        selected_context_entities.add(source)
        selected_context_entities.add(target)

    relation_covered = sum(
        bool(_answers(sample) & selected_relations) for sample in qa_samples
    )
    qa_endpoints = [
        str(sample[key])
        for sample in qa_samples
        for key in ("source_node", "target_node")
        if sample.get(key) is not None
    ]
    entity_covered = sum(node in selected_context_entities for node in qa_endpoints)
    selected_edge_set = set(selected_edges)
    qa_fact_covered = sum(fact in selected_edge_set for fact in priority_edges)

    selection_payload = {
        "seed": int(seed),
        "nodes": selected_nodes,
        "edges": [list(edge) for edge in selected_edges],
    }
    manifest = {
        "sampling_strategy": "node_random_train_qa_anchor_edge_relation_round_robin_v2",
        "seed": int(seed),
        "requested_node_count": int(node_quota),
        "requested_edge_count": int(edge_quota),
        "requested_total_count": int(node_quota + edge_quota),
        "unique_node_count": len(nodes),
        "unique_edge_count": len(edges),
        "duplicate_edge_count": duplicate_count,
        "self_loop_count": self_loop_count,
        "relation_count": len(all_relations),
        "selected_node_count": len(selected_nodes),
        "selected_edge_count": len(selected_edges),
        "selected_total_count": len(selected_nodes) + len(selected_edges),
        "selected_relation_count": len(selected_relations),
        "relation_coverage": (
            len(selected_relations) / len(all_relations) if all_relations else 0.0
        ),
        "selected_context_entity_count": len(selected_context_entities),
        "train_qa_count": len(qa_samples),
        "train_qa_relation_covered_count": relation_covered,
        "train_qa_relation_coverage": (
            relation_covered / len(qa_samples) if qa_samples else 0.0
        ),
        "train_qa_fact_count": len(priority_edges),
        "train_qa_fact_covered_count": qa_fact_covered,
        "train_qa_fact_coverage": (
            qa_fact_covered / len(priority_edges) if priority_edges else 0.0
        ),
        "train_qa_entity_endpoint_count": len(qa_endpoints),
        "train_qa_entity_covered_count": entity_covered,
        "train_qa_entity_coverage": (
            entity_covered / len(qa_endpoints) if qa_endpoints else 0.0
        ),
        "selected_nodes": selected_nodes,
        "selected_edges": [list(edge) for edge in selected_edges],
        "selected_nodes_sha256": _sha256(selected_nodes),
        "selected_edges_sha256": _sha256([list(edge) for edge in selected_edges]),
        "selection_sha256": _sha256(selection_payload),
    }
    return selected_nodes, selected_edges, manifest
