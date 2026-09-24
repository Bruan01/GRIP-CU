"""Relation-global confusion vocabulary built from saved QA × candidate scores.

This is not semantic similarity. Each cell describes how often, and how
strongly, frozen GRIP confuses candidate ``r_j`` with gold ``r_i``.

QA-level dumps stay untouched. ``min_support`` filters display views only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .confusion_analysis import (
    iter_jsonl,
    percentile,
    write_csv,
    write_json,
)

PAIR_FIELDS = (
    "gold_relation",
    "candidate_relation",
    "sample_count",
    "mean_candidate_score",
    "mean_score_gap",
    "median_score_gap",
    "mean_pairwise_confusion",
    "median_pairwise_confusion",
    "p90_pairwise_confusion",
    "beats_gold_count",
    "beats_gold_rate",
)
MATRIX_VERSIONS = (
    ("mean_pairwise_confusion", "relation_confusion_mean.csv"),
    ("median_pairwise_confusion", "relation_confusion_median.csv"),
    ("beats_gold_rate", "relation_confusion_beats_gold.csv"),
)


def gold_key(row: dict) -> str:
    """Canonical gold used by the reusable matrix: the matched train relation."""
    matched = row.get("matched_train_relation")
    if matched:
        return str(matched)
    return str(row["gold_relation"])


def beats_gold(row: dict) -> bool:
    """True iff the candidate continuation scores strictly above gold."""
    if row.get("candidate_score") is not None and row.get("gold_score") is not None:
        return float(row["candidate_score"]) > float(row["gold_score"])
    if row.get("score_gap") is None:
        raise ValueError("row is missing score_gap and gold/candidate scores")
    return float(row["score_gap"]) < 0


def iter_valid_negative_rows(path: Path) -> Iterator[dict]:
    for row in iter_jsonl(path):
        if row.get("is_valid_negative"):
            yield row


def aggregate_relation_pairs(rows: Iterable[dict]) -> dict:
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "count": 0,
            "score_sum": 0.0,
            "gap_sum": 0.0,
            "confusion_sum": 0.0,
            "scores": [],
            "gaps": [],
            "confusions": [],
            "beats": 0,
            "qa_ids": set(),
        }
    )
    gold_qas: dict[str, set[str]] = defaultdict(set)
    n_rows = 0
    n_qa = 0
    seen_qa: set[str] = set()
    for row in rows:
        gold = gold_key(row)
        candidate = str(row["candidate_relation"])
        if gold == candidate:
            continue
        qa_id = str(row.get("qa_id") or "")
        gap = float(row["score_gap"])
        confusion = float(row["pairwise_confusion"])
        score = float(row["candidate_score"])
        bucket = buckets[(gold, candidate)]
        bucket["count"] += 1
        bucket["score_sum"] += score
        bucket["gap_sum"] += gap
        bucket["confusion_sum"] += confusion
        bucket["scores"].append(score)
        bucket["gaps"].append(gap)
        bucket["confusions"].append(confusion)
        if beats_gold(row):
            bucket["beats"] += 1
        if qa_id:
            bucket["qa_ids"].add(qa_id)
            gold_qas[gold].add(qa_id)
            if qa_id not in seen_qa:
                seen_qa.add(qa_id)
                n_qa += 1
        n_rows += 1

    pair_rows = []
    for (gold, candidate), bucket in buckets.items():
        count = int(bucket["count"])
        pair_rows.append(
            {
                "gold_relation": gold,
                "candidate_relation": candidate,
                "sample_count": count,
                "mean_candidate_score": bucket["score_sum"] / count,
                "mean_score_gap": bucket["gap_sum"] / count,
                "median_score_gap": percentile(bucket["gaps"], 50),
                "mean_pairwise_confusion": bucket["confusion_sum"] / count,
                "median_pairwise_confusion": percentile(bucket["confusions"], 50),
                "p90_pairwise_confusion": percentile(bucket["confusions"], 90),
                "beats_gold_count": int(bucket["beats"]),
                "beats_gold_rate": bucket["beats"] / count,
            }
        )
    pair_rows.sort(
        key=lambda row: (
            -int(row["sample_count"]),
            -float(row["mean_pairwise_confusion"]),
            str(row["gold_relation"]),
            str(row["candidate_relation"]),
        )
    )
    gold_support = {gold: len(qas) for gold, qas in gold_qas.items()}
    return {
        "n_valid_negatives": n_rows,
        "n_qa": n_qa,
        "n_pairs": len(pair_rows),
        "pair_rows": pair_rows,
        "gold_qa_count": gold_support,
    }


def filter_pairs_for_display(pair_rows: list[dict], min_support: int) -> list[dict]:
    if min_support <= 1:
        return list(pair_rows)
    return [row for row in pair_rows if int(row["sample_count"]) >= min_support]


def pair_lookup(pair_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["gold_relation"], row["candidate_relation"]): row for row in pair_rows}


def matrix_grid(
    pair_rows: list[dict],
    relation_order: list[str],
    *,
    value_key: str,
    min_support: int = 1,
) -> list[list]:
    index = pair_lookup(pair_rows)
    grid: list[list] = []
    for gold in relation_order:
        row = []
        for candidate in relation_order:
            if gold == candidate:
                row.append(None)
                continue
            payload = index.get((gold, candidate))
            if payload is None or int(payload["sample_count"]) < min_support:
                row.append(None)
                continue
            row.append(payload[value_key])
        grid.append(row)
    return grid


def write_matrix_csv(
    path: Path,
    pair_rows: list[dict],
    relation_order: list[str],
    *,
    value_key: str,
    min_support: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = matrix_grid(
        pair_rows,
        relation_order,
        value_key=value_key,
        min_support=min_support,
    )
    fieldnames = ["gold_relation", *relation_order]
    rows = []
    for gold, values in zip(relation_order, grid):
        encoded = {candidate: ("" if value is None else value) for candidate, value in zip(relation_order, values)}
        rows.append({"gold_relation": gold, **encoded})
    write_csv(path, rows, fieldnames)


def confusion_neighbors(
    pair_rows: list[dict],
    *,
    top_k: int = 20,
    min_support: int = 1,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        if int(row["sample_count"]) < min_support:
            continue
        grouped[str(row["gold_relation"])].append(row)
    neighbors: dict[str, list[dict]] = {}
    for gold, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda item: (
                -float(item["mean_pairwise_confusion"]),
                -int(item["sample_count"]),
                str(item["candidate_relation"]),
            ),
        )
        neighbors[gold] = [
            {
                "relation": item["candidate_relation"],
                "mean_confusion": item["mean_pairwise_confusion"],
                "median_confusion": item["median_pairwise_confusion"],
                "p90_confusion": item["p90_pairwise_confusion"],
                "beats_gold_rate": item["beats_gold_rate"],
                "support": item["sample_count"],
            }
            for item in ranked[:top_k]
        ]
    return dict(sorted(neighbors.items()))


def frequent_golds(gold_qa_count: dict[str, int], top_n: int) -> list[str]:
    ranked = sorted(gold_qa_count.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ranked[:top_n]]


def _short_label(relation: str) -> str:
    return relation.removeprefix("concept:")


def _save_heatmap(
    path: Path,
    *,
    labels: list[str],
    grid: list[list],
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.array(
        [[math.nan if value is None else float(value) for value in row] for row in grid],
        dtype=float,
    )
    width = max(8.0, 0.28 * len(labels) + 2.5)
    height = max(7.0, 0.28 * len(labels) + 2.0)
    figure, axes = plt.subplots(figsize=(width, height))
    image = axes.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes.set_title(title)
    ticks = list(range(len(labels)))
    names = [_short_label(name) for name in labels]
    axes.set_xticks(ticks, labels=names, rotation=90, fontsize=7)
    axes.set_yticks(ticks, labels=names, fontsize=7)
    axes.set_xlabel("candidate relation")
    axes.set_ylabel("gold relation")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _save_neighborhood_bar(
    path: Path,
    *,
    gold: str,
    neighbors: list[dict],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    names = [_short_label(item["relation"]) for item in reversed(neighbors)]
    values = [float(item["mean_confusion"]) for item in reversed(neighbors)]
    supports = [int(item["support"]) for item in reversed(neighbors)]
    figure, axes = plt.subplots(figsize=(9, max(4.0, 0.32 * len(names) + 1.5)))
    axes.barh(names, values, color="#3b6ea5")
    axes.set_title(f"Confusion neighborhood: {_short_label(gold)}")
    axes.set_xlabel("mean pairwise_confusion")
    for index, (value, support) in enumerate(zip(values, supports)):
        axes.text(value, index, f"  n={support}", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def write_confusion_vocab_outputs(
    *,
    pair_payload: dict,
    relation_order: list[str],
    output_dir: Path,
    metadata: dict,
    min_support: int,
    neighbor_k: int,
    heatmap_top_n: int,
    neighborhood_plot_n: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    pair_rows: list[dict] = pair_payload["pair_rows"]
    display_pairs = filter_pairs_for_display(pair_rows, min_support)
    write_csv(output_dir / "relation_pair_statistics.csv", pair_rows, list(PAIR_FIELDS))
    for value_key, filename in MATRIX_VERSIONS:
        write_matrix_csv(
            output_dir / filename,
            pair_rows,
            relation_order,
            value_key=value_key,
            min_support=1,
        )
    neighbors = confusion_neighbors(pair_rows, top_k=neighbor_k, min_support=1)
    display_neighbors = confusion_neighbors(
        pair_rows, top_k=neighbor_k, min_support=min_support
    )
    write_json(output_dir / "relation_confusion_neighbors.json", neighbors)

    gold_qa_count = pair_payload["gold_qa_count"]
    top_golds = frequent_golds(gold_qa_count, heatmap_top_n)
    display_index = pair_lookup(display_pairs)
    mean_grid = []
    support_grid = []
    for gold in top_golds:
        mean_row = []
        support_row = []
        for candidate in top_golds:
            if gold == candidate:
                mean_row.append(None)
                support_row.append(None)
                continue
            payload = display_index.get((gold, candidate))
            if payload is None:
                mean_row.append(None)
                support_row.append(None)
            else:
                mean_row.append(payload["mean_pairwise_confusion"])
                support_row.append(payload["sample_count"])
        mean_grid.append(mean_row)
        support_grid.append(support_row)
    figures_written: list[str] = []
    figures_skipped = False
    if top_golds:
        if _save_heatmap(
            figures / "confusion_mean_heatmap_topn.png",
            labels=top_golds,
            grid=mean_grid,
            title=f"Mean pairwise confusion (top {len(top_golds)} golds, min_support={min_support})",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=None,
        ):
            figures_written.append("figures/confusion_mean_heatmap_topn.png")
        else:
            figures_skipped = True
        if _save_heatmap(
            figures / "support_heatmap_topn.png",
            labels=top_golds,
            grid=support_grid,
            title=f"Pair support (top {len(top_golds)} golds, min_support={min_support})",
            cmap="Blues",
            vmin=0.0,
            vmax=None,
        ):
            figures_written.append("figures/support_heatmap_topn.png")
        else:
            figures_skipped = True
    for gold in frequent_golds(gold_qa_count, neighborhood_plot_n):
        items = display_neighbors.get(gold) or []
        if not items:
            continue
        safe = gold.replace(":", "_")
        relative = f"figures/neighborhood_{safe}.png"
        if _save_neighborhood_bar(
            figures / f"neighborhood_{safe}.png",
            gold=gold,
            neighbors=items,
        ):
            figures_written.append(relative)
        else:
            figures_skipped = True

    payload = {
        **metadata,
        "n_qa": pair_payload["n_qa"],
        "n_valid_negatives": pair_payload["n_valid_negatives"],
        "n_pairs": pair_payload["n_pairs"],
        "n_gold_relations_observed": len(gold_qa_count),
        "vocab_size": len(relation_order),
        "min_support": min_support,
        "min_support_applies_to": ["figures", "display filtering"],
        "raw_table_unfiltered": True,
        "neighbor_k": neighbor_k,
        "heatmap_top_n": heatmap_top_n,
        "neighborhood_plot_n": neighborhood_plot_n,
        "gold_key": "matched_train_relation",
        "level": "relation-global",
        "neighbors_are_analysis_only": True,
        "figures": figures_written,
        "figures_skipped": figures_skipped,
        "does_not_replace": "QA-specific candidate_scores.jsonl / analysis/",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "metadata.json", payload)
    return payload
