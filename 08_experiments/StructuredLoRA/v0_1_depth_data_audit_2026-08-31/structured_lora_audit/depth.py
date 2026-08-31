"""Support-depth labeling and aggregate audit statistics."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .graph import KnowledgeGraph
from .io_utils import Triple


def depth_bucket(distance: int | None, max_depth: int) -> str:
    if distance is None:
        return f">{max_depth}_or_unreachable"
    if distance == 0:
        return "self_loop"
    return str(distance)


def label_split(
    graph: KnowledgeGraph,
    triples: Iterable[Triple],
    *,
    split: str,
    max_depth: int,
    leave_one_out: bool,
) -> list[dict]:
    rows: list[dict] = []
    for index, triple in enumerate(triples):
        head, relation, tail = triple
        distances: dict[str, int | None] = {}
        for mode, directed in (("undirected", False), ("directed", True)):
            if leave_one_out:
                value = graph.leave_one_out_distance(triple, max_depth=max_depth, directed=directed)
            else:
                value = graph.bounded_distance(head, tail, max_depth=max_depth, directed=directed)
            distances[mode] = value
        rows.append(
            {
                "split": split,
                "index": index,
                "head": head,
                "relation": relation,
                "tail": tail,
                "depth_definition": "leave_one_edge_out" if leave_one_out else "train_graph_support",
                "undirected_distance": distances["undirected"],
                "undirected_bucket": depth_bucket(distances["undirected"], max_depth),
                "directed_distance": distances["directed"],
                "directed_bucket": depth_bucket(distances["directed"], max_depth),
                "relation_train_frequency": graph.relation_count.get(relation, 0),
                "exact_triple_in_train": graph.triple_count.get(triple, 0) > (1 if leave_one_out else 0),
                "directed_pair_train_multiplicity": graph.directed_pair_count.get((head, tail), 0),
                "undirected_pair_train_multiplicity": graph.undirected_pair_count.get(
                    graph._undirected_key(head, tail), 0
                ),
            }
        )
    return rows


def distribution_rows(labels: Iterable[dict]) -> list[dict]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in labels:
        for mode in ("undirected", "directed"):
            key = (row["split"], mode, row[f"{mode}_bucket"])
            counts[key] += 1
            totals[(row["split"], mode)] += 1
    order = {"self_loop": 0, "1": 1, "2": 2, "3": 3, "4+": 4, "unreachable": 99}
    output = []
    for (split, mode, bucket), count in sorted(
        counts.items(), key=lambda item: (item[0][0], item[0][1], order.get(item[0][2], 50))
    ):
        output.append(
            {
                "split": split,
                "mode": mode,
                "depth_bucket": bucket,
                "count": count,
                "fraction": count / totals[(split, mode)],
            }
        )
    return output


def relation_depth_rows(labels: Iterable[dict], *, mode: str = "undirected") -> list[dict]:
    table: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in labels:
        table[(row["split"], row["relation"], row[f"{mode}_bucket"])].append(row)
    relation_totals: Counter[tuple[str, str]] = Counter()
    for row in labels:
        relation_totals[(row["split"], row["relation"])] += 1
    output = []
    for (split, relation, bucket), members in sorted(table.items()):
        frequencies = [int(row["relation_train_frequency"]) for row in members]
        output.append(
            {
                "split": split,
                "relation": relation,
                "depth_mode": mode,
                "depth_bucket": bucket,
                "count": len(members),
                "fraction_within_relation": len(members) / relation_totals[(split, relation)],
                "relation_train_frequency": frequencies[0],
            }
        )
    return output



def frequency_depth_rows(labels: Iterable[dict], *, mode: str = "undirected") -> list[dict]:
    rows = list(labels)
    frequencies = sorted({int(row["relation_train_frequency"]) for row in rows})
    if not frequencies:
        return []

    def percentile(q: float) -> int:
        index = round((len(frequencies) - 1) * q)
        return frequencies[index]

    lower = percentile(1 / 3)
    upper = percentile(2 / 3)

    def tier(value: int) -> str:
        if value <= lower:
            return "tail"
        if value <= upper:
            return "mid"
        return "head"

    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        name = tier(int(row["relation_train_frequency"]))
        counts[(row["split"], name, row[f"{mode}_bucket"])] += 1
        totals[(row["split"], name)] += 1
    output = []
    for (split, name, bucket), count in sorted(counts.items()):
        output.append({
            "split": split,
            "frequency_tier": name,
            "depth_mode": mode,
            "depth_bucket": bucket,
            "count": count,
            "fraction_within_tier": count / totals[(split, name)],
            "tail_max_frequency": lower,
            "mid_max_frequency": upper,
        })
    return output

def normalized_mutual_information(labels: Iterable[dict], *, split: str, mode: str = "undirected") -> float:
    pairs = [(row["relation"], row[f"{mode}_bucket"]) for row in labels if row["split"] == split]
    if not pairs:
        return 0.0
    n = len(pairs)
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    mutual_information = 0.0
    for (a, b), count in joint.items():
        p_ab = count / n
        mutual_information += p_ab * math.log(p_ab / ((left[a] / n) * (right[b] / n)))
    h_left = -sum((count / n) * math.log(count / n) for count in left.values())
    h_right = -sum((count / n) * math.log(count / n) for count in right.values())
    denominator = math.sqrt(h_left * h_right)
    return mutual_information / denominator if denominator else 0.0
