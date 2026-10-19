"""Describe relation confusion from saved candidate scores only.

This script never loads a language model and never changes the saved scores.
"""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np


GAP_BUCKETS = (
    "gap < 0",
    "0 <= gap < 0.1",
    "0.1 <= gap < 0.5",
    "0.5 <= gap < 1",
    "1 <= gap < 2",
    "gap >= 2",
)
CONFUSION_BUCKETS = (
    "0 <= confusion < 0.05",
    "0.05 <= confusion < 0.10",
    "0.10 <= confusion < 0.20",
    "0.20 <= confusion < 0.30",
    "0.30 <= confusion < 0.40",
    "0.40 <= confusion < 0.50",
    "confusion >= 0.50",
)
TOP_N = (1, 2, 3, 5, 9, 10, 20)
COVERAGE_THRESHOLDS = (0.70, 0.80, 0.90, 0.95, 0.99)
METRICS = (
    "candidate_score",
    "score_gap",
    "pairwise_confusion",
    "negative_mass",
    "negative_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--min-support", "--min_support", dest="min_support", type=int, default=20)
    parser.add_argument("--neighbor-top-k", "--neighbor_top_k", dest="neighbor_top_k", type=int, default=20)
    return parser.parse_args()


def _stats(values) -> dict:
    array_values = np.asarray(values, dtype=np.float64)
    if array_values.size == 0:
        return {"count": 0}
    percentiles = np.percentile(array_values, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(array_values.size),
        "mean": float(np.mean(array_values)),
        "std": float(np.std(array_values)),
        "median": float(percentiles[2]),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "min": float(np.min(array_values)),
        "max": float(np.max(array_values)),
    }


def _bucket_gap(value: float) -> str:
    if value < 0:
        return GAP_BUCKETS[0]
    if value < 0.1:
        return GAP_BUCKETS[1]
    if value < 0.5:
        return GAP_BUCKETS[2]
    if value < 1:
        return GAP_BUCKETS[3]
    if value < 2:
        return GAP_BUCKETS[4]
    return GAP_BUCKETS[5]


def _bucket_confusion(value: float) -> str:
    if value < 0.05:
        return CONFUSION_BUCKETS[0]
    if value < 0.10:
        return CONFUSION_BUCKETS[1]
    if value < 0.20:
        return CONFUSION_BUCKETS[2]
    if value < 0.30:
        return CONFUSION_BUCKETS[3]
    if value < 0.40:
        return CONFUSION_BUCKETS[4]
    if value < 0.50:
        return CONFUSION_BUCKETS[5]
    return CONFUSION_BUCKETS[6]


def _new_qa_state(row: dict) -> dict:
    return {
        "qa_id": row["qa_id"],
        "global_index": row.get("global_index"),
        "gold_relation": row["gold_relation"],
        "gold_score": None,
        "row_count": 0,
        "gold_count": 0,
        "valid_negative_count": 0,
        "invalid_negative_count": 0,
        "scores": [],
        "gaps": [],
        "confusions": [],
        "masses": [],
        "candidates": [],
        "all_true_relations": tuple(row.get("all_true_relations", [])),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_scoring_metadata(input_path: Path, requested_path: str | None) -> Path | None:
    if requested_path is not None:
        metadata_path = Path(requested_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Scoring metadata does not exist: {metadata_path}")
        return metadata_path
    for metadata_path in (input_path.parent / "metadata.json", input_path.parent.parent / "metadata.json"):
        if metadata_path.is_file():
            return metadata_path
    return None


def _infer_dataset(metadata: dict, input_path: Path) -> str | None:
    values = [
        metadata.get("dataset"),
        metadata.get("candidate_file"),
        metadata.get("processed_file"),
        str(input_path),
    ]
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("dataset")
        if not value:
            continue
        match = re.search(r"(fb15k237|nell23k|wikidata5m)", str(value), flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _load_vocab_metadata(
    input_path: Path,
    metadata_path: Path | None,
    context: dict,
    min_support: int,
    neighbor_top_k: int,
) -> dict:
    source = {}
    if metadata_path is not None:
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = source.get("identity", source)
    dataset_metadata = dict(source)
    if isinstance(identity, dict):
        dataset_metadata.update(identity)
    return {
        "artifact_type": "relation_confusion_vocabulary",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_candidate_scores": str(input_path.resolve()),
        "scoring_metadata": str(metadata_path.resolve()) if metadata_path else None,
        "checkpoint": identity.get("model_checkpoint") or source.get("checkpoint"),
        "dataset": _infer_dataset(dataset_metadata, input_path),
        "split": identity.get("split") or source.get("split"),
        "vocab_size": len(context["candidate_vocabulary"]),
        "relation_id_count": identity.get("relation_id_count") or source.get("relation_id_count"),
        "T": identity.get("temperature") or source.get("temperature"),
        "scoring_version": identity.get("scoring_version") or source.get("scoring_version"),
        "min_support": min_support,
        "neighbor_top_k": neighbor_top_k,
        "candidate_vocab_size": len(context["candidate_vocabulary"]),
        "gold_vocab_size": len(context["gold_vocabulary"]),
        "raw_pair_count": 0,
        "display_pair_count": 0,
    }


def _to_vocab_pair_rows(pair_rows: list[dict]) -> list[dict]:
    rows = []
    for row in pair_rows:
        rows.append({
            "gold_relation": row["gold_relation"],
            "candidate_relation": row["candidate_relation"],
            "sample_count": row["count"],
            "mean_candidate_score": row["mean_candidate_score"],
            "mean_score_gap": row["mean_score_gap"],
            "median_score_gap": row["median_score_gap"],
            "mean_pairwise_confusion": row["mean_pairwise_confusion"],
            "median_pairwise_confusion": row["median_pairwise_confusion"],
            "p90_pairwise_confusion": row["p90_pairwise_confusion"],
            "beats_gold_count": row["candidate_beats_gold_count"],
            "beats_gold_rate": row["candidate_beats_gold_fraction"],
        })
    return rows


def _write_relation_matrix(
    path: Path,
    gold_relations: list[str],
    candidate_relations: list[str],
    pair_map: dict[tuple[str, str], dict],
    metric: str,
    missing_value,
) -> None:
    fieldnames = ["gold_relation", *candidate_relations]
    rows = []
    for gold_relation in gold_relations:
        row = {"gold_relation": gold_relation}
        for candidate_relation in candidate_relations:
            pair = pair_map.get((gold_relation, candidate_relation))
            row[candidate_relation] = missing_value if pair is None else pair[metric]
        rows.append(row)
    _write_csv(path, fieldnames, rows)


def _short_label(value: str, limit: int = 24) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


def _select_heatmap_axes(
    pair_rows: list[dict],
    min_support: int,
    gold_limit: int = 20,
    candidate_limit: int = 20,
) -> tuple[list[str], list[str]]:
    support_by_gold = defaultdict(int)
    candidate_score: dict[str, tuple[float, int]] = {}
    for row in pair_rows:
        if row["sample_count"] < min_support:
            continue
        support_by_gold[row["gold_relation"]] += row["sample_count"]
        current = candidate_score.get(row["candidate_relation"], (-1.0, 0))
        candidate_score[row["candidate_relation"]] = (
            max(current[0], row["mean_pairwise_confusion"]),
            current[1] + row["sample_count"],
        )
    gold_relations = [
        relation for relation, _ in sorted(
            support_by_gold.items(), key=lambda item: (-item[1], item[0])
        )[:gold_limit]
    ]
    candidate_relations = [
        relation for relation, _ in sorted(
            candidate_score.items(),
            key=lambda item: (-item[1][0], -item[1][1], item[0]),
        )[:candidate_limit]
    ]
    return gold_relations, candidate_relations


def _make_relation_vocab_figures(
    output_dir: Path,
    pair_rows: list[dict],
    neighbors: dict[str, list[dict]],
    min_support: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    style_candidates = [
        Path("/mnt/c/Users/Administrator/AppData/Roaming/com.ai4s.workbench/runtime/xdg-config/opencode/skills/publication-figures/openscience.mplstyle"),
        Path(r"C:\Users\Administrator\AppData\Roaming\com.ai4s.workbench\runtime\xdg-config\opencode\skills\publication-figures\openscience.mplstyle"),
    ]
    for style_path in style_candidates:
        if style_path.is_file():
            plt.style.use(str(style_path))
            break
    else:
        plt.style.use("default")

    gold_relations, candidate_relations = _select_heatmap_axes(pair_rows, min_support)
    pair_map = {
        (row["gold_relation"], row["candidate_relation"]): row
        for row in pair_rows
        if row["sample_count"] >= min_support
    }

    def heatmap(metric: str, title: str, path: Path, cmap: str) -> None:
        values = np.full((len(gold_relations), len(candidate_relations)), np.nan, dtype=np.float64)
        for row_index, gold_relation in enumerate(gold_relations):
            for column_index, candidate_relation in enumerate(candidate_relations):
                pair = pair_map.get((gold_relation, candidate_relation))
                if pair is not None:
                    values[row_index, column_index] = pair[metric]
        figure, axis = plt.subplots(
            figsize=(max(8.0, 0.45 * len(candidate_relations)), max(5.5, 0.35 * len(gold_relations))),
            constrained_layout=True,
        )
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("Candidate relation")
        axis.set_ylabel("Gold relation")
        axis.set_xticks(range(len(candidate_relations)))
        axis.set_xticklabels([_short_label(value) for value in candidate_relations], rotation=60, ha="right", fontsize=7)
        axis.set_yticks(range(len(gold_relations)))
        axis.set_yticklabels([_short_label(value) for value in gold_relations], fontsize=7)
        figure.colorbar(image, ax=axis, shrink=0.82)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    heatmap(
        "mean_pairwise_confusion",
        f"Mean relation confusion (support >= {min_support})",
        output_dir / "relation_confusion_mean_heatmap.png",
        "Blues",
    )
    heatmap(
        "sample_count",
        f"Relation-pair support (support >= {min_support})",
        output_dir / "relation_confusion_support_heatmap.png",
        "Greens",
    )

    neighborhood_gold = gold_relations[:6]
    columns = 2
    rows = max(1, math.ceil(len(neighborhood_gold) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.1 * rows), squeeze=False, constrained_layout=True)
    for index, gold_relation in enumerate(neighborhood_gold):
        axis = axes[index // columns][index % columns]
        neighborhood = neighbors.get(gold_relation, [])[:5]
        if neighborhood:
            labels = [_short_label(item["relation"], 30) for item in neighborhood][::-1]
            values = [item["mean_confusion"] for item in neighborhood][::-1]
            axis.barh(labels, values, color="#2a78d6")
            axis.set_xlabel("Mean pairwise confusion")
        else:
            axis.text(0.5, 0.5, "No pair meets support threshold", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_title(_short_label(gold_relation, 34))
    for index in range(len(neighborhood_gold), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.savefig(output_dir / "relation_confusion_neighborhoods.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _build_relation_confusion_vocab(
    output_dir: Path,
    pair_rows: list[dict],
    context: dict,
    metadata_path: Path | None,
    min_support: int,
    neighbor_top_k: int,
) -> dict:
    if min_support < 1:
        raise ValueError("min_support must be positive")
    if neighbor_top_k < 1:
        raise ValueError("neighbor_top_k must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_pair_rows = _to_vocab_pair_rows(pair_rows)
    vocab_pair_rows.sort(key=lambda row: (row["gold_relation"], row["candidate_relation"]))
    pair_fields = list(vocab_pair_rows[0]) if vocab_pair_rows else [
        "gold_relation", "candidate_relation", "sample_count", "mean_candidate_score",
        "mean_score_gap", "median_score_gap", "mean_pairwise_confusion",
        "median_pairwise_confusion", "p90_pairwise_confusion", "beats_gold_count", "beats_gold_rate",
    ]
    _write_csv(output_dir / "relation_pair_statistics.csv", pair_fields, vocab_pair_rows)

    pair_map = {(row["gold_relation"], row["candidate_relation"]): row for row in vocab_pair_rows}
    gold_relations = sorted(context["gold_vocabulary"])
    candidate_relations = sorted(context["candidate_vocabulary"])
    _write_relation_matrix(
        output_dir / "relation_confusion_mean.csv",
        gold_relations,
        candidate_relations,
        pair_map,
        "mean_pairwise_confusion",
        "",
    )
    _write_relation_matrix(
        output_dir / "relation_confusion_median.csv",
        gold_relations,
        candidate_relations,
        pair_map,
        "median_pairwise_confusion",
        "",
    )
    _write_relation_matrix(
        output_dir / "relation_confusion_beats_gold.csv",
        gold_relations,
        candidate_relations,
        pair_map,
        "beats_gold_rate",
        "",
    )
    _write_relation_matrix(
        output_dir / "relation_confusion_support.csv",
        gold_relations,
        candidate_relations,
        pair_map,
        "sample_count",
        0,
    )

    neighbors = {}
    for gold_relation in gold_relations:
        candidates = [
            row for row in vocab_pair_rows
            if row["gold_relation"] == gold_relation and row["sample_count"] >= min_support
        ]
        candidates.sort(
            key=lambda row: (
                -row["mean_pairwise_confusion"],
                -row["beats_gold_rate"],
                -row["sample_count"],
                row["candidate_relation"],
            )
        )
        neighbors[gold_relation] = [
            {
                "relation": row["candidate_relation"],
                "mean_confusion": row["mean_pairwise_confusion"],
                "median_confusion": row["median_pairwise_confusion"],
                "beats_gold_rate": row["beats_gold_rate"],
                "support": row["sample_count"],
            }
            for row in candidates[:neighbor_top_k]
        ]
    _write_json(output_dir / "relation_confusion_neighbors.json", neighbors)

    metadata = _load_vocab_metadata(
        Path(context["input_file"]), metadata_path, context, min_support, neighbor_top_k
    )
    metadata["raw_pair_count"] = len(vocab_pair_rows)
    metadata["display_pair_count"] = sum(row["sample_count"] >= min_support for row in vocab_pair_rows)
    metadata["matrix_shape"] = {
        "rows_gold_relations": len(gold_relations),
        "columns_candidate_relations": len(candidate_relations),
    }
    _write_json(output_dir / "metadata.json", metadata)
    _make_relation_vocab_figures(output_dir, vocab_pair_rows, neighbors, min_support)
    return metadata


def _load_and_aggregate(path: Path) -> tuple[dict, dict]:
    qa_states: dict[object, dict] = {}
    metric_values = {name: array("d") for name in METRICS}
    gap_buckets = {
        bucket: {"candidate_count": 0, "qa_ids": set()}
        for bucket in GAP_BUCKETS
    }
    confusion_buckets = {
        bucket: {"candidate_count": 0, "qa_ids": set()}
        for bucket in CONFUSION_BUCKETS
    }
    relation_pairs: dict[tuple[str, str], dict] = {}
    candidate_vocabulary: set[str] = set()
    gold_vocabulary: set[str] = set()
    total_rows = 0
    valid_negative_rows = 0
    invalid_negative_rows = 0
    input_errors = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            total_rows += 1
            qa_id = row["qa_id"]
            state = qa_states.setdefault(qa_id, _new_qa_state(row))
            state["row_count"] += 1
            candidate_vocabulary.add(row["candidate_relation"])
            gold_vocabulary.add(row["gold_relation"])
            gold_score = float(row["gold_score"])
            if state["gold_score"] is None:
                state["gold_score"] = gold_score
            elif not math.isclose(state["gold_score"], gold_score, rel_tol=0.0, abs_tol=1e-7):
                input_errors.append({"qa_id": qa_id, "error": "gold_score_inconsistent"})
            if row["is_gold"]:
                state["gold_count"] += 1
                continue
            if not row["is_valid_negative"]:
                invalid_negative_rows += 1
                state["invalid_negative_count"] += 1
                continue

            score = float(row["candidate_score"])
            gap = float(row["score_gap"])
            confusion = row.get("pairwise_confusion")
            mass = row.get("negative_mass")
            if confusion is None or mass is None:
                input_errors.append({"qa_id": qa_id, "error": "missing_valid_negative_metric"})
                continue
            confusion = float(confusion)
            mass = float(mass)
            valid_negative_rows += 1
            state["valid_negative_count"] += 1
            state["scores"].append(score)
            state["gaps"].append(gap)
            state["confusions"].append(confusion)
            state["masses"].append(mass)
            state["candidates"].append(row["candidate_relation"])
            metric_values["candidate_score"].append(score)
            metric_values["score_gap"].append(gap)
            metric_values["pairwise_confusion"].append(confusion)
            metric_values["negative_mass"].append(mass)
            metric_values["negative_rank"].append(float(row["negative_rank"]))

            gap_bucket = gap_buckets[_bucket_gap(gap)]
            gap_bucket["candidate_count"] += 1
            gap_bucket["qa_ids"].add(qa_id)
            confusion_bucket = confusion_buckets[_bucket_confusion(confusion)]
            confusion_bucket["candidate_count"] += 1
            confusion_bucket["qa_ids"].add(qa_id)

            pair_key = (row["gold_relation"], row["candidate_relation"])
            pair = relation_pairs.setdefault(pair_key, {
                "count": 0,
                "scores": [],
                "gaps": [],
                "confusions": [],
                "beats_gold": 0,
            })
            pair["count"] += 1
            pair["scores"].append(score)
            pair["gaps"].append(gap)
            pair["confusions"].append(confusion)
            pair["beats_gold"] += int(score > gold_score)

    for state in qa_states.values():
        if state["gold_count"] != 1:
            input_errors.append({"qa_id": state["qa_id"], "error": "gold_count_not_one"})
        if state["row_count"] != len(candidate_vocabulary):
            input_errors.append({"qa_id": state["qa_id"], "error": "candidate_count_mismatch"})
        known_true = set(state["all_true_relations"])
        if any(candidate in known_true for candidate in state["candidates"]):
            # Valid-negative rows should never contain an alternative true relation.
            input_errors.append({"qa_id": state["qa_id"], "error": "unfiltered_true_relation"})

    context = {
        "input_file": str(path.resolve()),
        "qa_states": qa_states,
        "metric_values": metric_values,
        "gap_buckets": gap_buckets,
        "confusion_buckets": confusion_buckets,
        "relation_pairs": relation_pairs,
        "candidate_vocabulary": candidate_vocabulary,
        "gold_vocabulary": gold_vocabulary,
        "total_rows": total_rows,
        "valid_negative_rows": valid_negative_rows,
        "invalid_negative_rows": invalid_negative_rows,
        "input_errors": input_errors,
    }
    return context, {"qa_count": len(qa_states)}


def _compute_qa_coverage(context: dict) -> tuple[list[dict], dict, dict]:
    per_qa = []
    top_mass_values = {str(n): [] for n in TOP_N}
    coverage_values = {f"K{int(threshold * 100)}": [] for threshold in COVERAGE_THRESHOLDS}
    hardest_gaps = []
    for state in sorted(
        context["qa_states"].values(),
        key=lambda item: (item["global_index"] is None, item["global_index"] or item["qa_id"]),
    ):
        masses = np.asarray(state["masses"], dtype=np.float64)
        scores = np.asarray(state["scores"], dtype=np.float64)
        gaps = np.asarray(state["gaps"], dtype=np.float64)
        if masses.size:
            order = np.argsort(-masses, kind="stable")
            ordered_masses = masses[order]
            cumulative = np.cumsum(ordered_masses)
            hardest_index = int(np.argmax(scores))
            hardest_gap = float(gaps[hardest_index])
            hardest_negative = state["candidates"][hardest_index]
            hardest_score = float(scores[hardest_index])
            negative_mass_sum = float(np.sum(masses))
            top_mass = {}
            for n in TOP_N:
                top_mass[str(n)] = float(np.sum(ordered_masses[:n]))
                top_mass_values[str(n)].append(top_mass[str(n)])
            for threshold in COVERAGE_THRESHOLDS:
                hits = np.flatnonzero(cumulative >= threshold)
                k = int(hits[0] + 1) if hits.size else int(len(ordered_masses))
                coverage_values[f"K{int(threshold * 100)}"].append(k)
        else:
            hardest_gap = None
            hardest_negative = None
            hardest_score = None
            negative_mass_sum = None
            top_mass = {str(n): None for n in TOP_N}
            for threshold in COVERAGE_THRESHOLDS:
                coverage_values[f"K{int(threshold * 100)}"].append(None)
        if hardest_gap is not None:
            hardest_gaps.append(hardest_gap)
        row = {
            "qa_id": state["qa_id"],
            "global_index": state["global_index"],
            "gold_relation": state["gold_relation"],
            "gold_score": state["gold_score"],
            "number_of_candidates": state["row_count"],
            "number_of_valid_negatives": state["valid_negative_count"],
            "number_of_invalid_negatives": state["invalid_negative_count"],
            "hardest_negative": hardest_negative,
            "hardest_negative_score": hardest_score,
            "hardest_negative_gap": hardest_gap,
            "number_of_negatives_above_gold": int(np.sum(gaps < 0)) if gaps.size else 0,
            "negative_mass_sum": negative_mass_sum,
            **{f"Top{n}_mass": top_mass[str(n)] for n in TOP_N},
            **{
                f"K{int(threshold * 100)}": coverage_values[f"K{int(threshold * 100)}"][-1]
                for threshold in COVERAGE_THRESHOLDS
            },
        }
        per_qa.append(row)

    top_mass_stats = {
        f"Top{n}": _stats([value for value in values if value is not None])
        for n, values in top_mass_values.items()
    }
    coverage_stats = {
        key: _stats([value for value in values if value is not None])
        for key, values in coverage_values.items()
    }
    return per_qa, top_mass_stats, {
        "coverage_values": coverage_values,
        "coverage_stats": coverage_stats,
        "hardest_gap_stats": _stats(hardest_gaps),
    }


def _make_figures(output_dir: Path, context: dict, per_qa: list[dict], top_mass_stats: dict, coverage: dict) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    style_candidates = [
        Path("/mnt/c/Users/Administrator/AppData/Roaming/com.ai4s.workbench/runtime/xdg-config/opencode/skills/publication-figures/openscience.mplstyle"),
        Path(r"C:\Users\Administrator\AppData\Roaming\com.ai4s.workbench\runtime\xdg-config\opencode\skills\publication-figures\openscience.mplstyle"),
    ]
    for style_path in style_candidates:
        if style_path.is_file():
            plt.style.use(str(style_path))
            break
    else:
        plt.style.use("default")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    gap = np.asarray(context["metric_values"]["score_gap"], dtype=np.float64)
    confusion = np.asarray(context["metric_values"]["pairwise_confusion"], dtype=np.float64)

    def save_hist(values, path: Path, title: str, xlabel: str, bins=60, xline=None):
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        axis.hist(values, bins=bins, color="#2a78d6", alpha=0.88, edgecolor="white", linewidth=0.35)
        if xline is not None:
            axis.axvline(xline, color="#e34948", linewidth=1.5, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Valid-negative count")
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    save_hist(gap, figures_dir / "score_gap_distribution.png", "Score-gap distribution", "Gold score - candidate score", xline=0.0)

    positive_confusion = confusion[confusion > 0]
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.hist(
        positive_confusion,
        bins=np.geomspace(max(float(np.min(positive_confusion)), 1e-12), 1.0, 51),
        color="#2a78d6",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
    )
    axis.set_xscale("log")
    axis.axvline(0.5, color="#e34948", linewidth=1.5, linestyle="--")
    axis.set_title("Pairwise-confusion distribution")
    axis.set_xlabel("Pairwise confusion (log scale)")
    axis.set_ylabel("Valid-negative count")
    figure.savefig(figures_dir / "pairwise_confusion_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    for key in ("K80", "K90", "K95"):
        values = [value for value in coverage["coverage_values"][key] if value is not None]
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        if values:
            upper = max(values)
            axis.hist(values, bins=np.arange(0.5, upper + 1.5), color="#1baf7a", edgecolor="white", linewidth=0.35)
        axis.set_title(f"{key} distribution")
        axis.set_xlabel("Minimum valid negatives needed")
        axis.set_ylabel("QA count")
        figure.savefig(figures_dir / f"{key.lower()}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    x_values = np.asarray(TOP_N, dtype=np.float64)
    means = np.asarray([top_mass_stats[f"Top{n}"]["mean"] for n in TOP_N])
    p25 = np.asarray([top_mass_stats[f"Top{n}"]["p25"] for n in TOP_N])
    p75 = np.asarray([top_mass_stats[f"Top{n}"]["p75"] for n in TOP_N])
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.plot(x_values, means, color="#2a78d6", marker="o", markersize=5, linewidth=2)
    axis.fill_between(x_values, p25, p75, color="#2a78d6", alpha=0.18)
    axis.set_xticks(TOP_N)
    axis.set_ylim(0, 1.02)
    axis.set_title("Cumulative valid-negative mass vs. Top-N")
    axis.set_xlabel("Top-N negatives by negative mass")
    axis.set_ylabel("Cumulative negative mass")
    figure.savefig(figures_dir / "cumulative_negative_mass_vs_top_n.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_report(output_dir: Path, global_stats: dict, pair_rows: list[dict]) -> None:
    valid_stats = global_stats["valid_negative_metrics"]
    gap_buckets = global_stats["gap_buckets"]
    top_mass = global_stats["top_n_negative_mass"]
    k_stats = global_stats["coverage_k_statistics"]
    total_valid = global_stats["valid_negative_count"]
    total_qa = global_stats["qa_count"]
    gap_negative = gap_buckets["gap < 0"]
    gap_easy = gap_buckets["gap >= 2"]
    confusion_easy = global_stats["confusion_buckets"]["0 <= confusion < 0.05"]
    hardest_gap_buckets = global_stats["hardest_gap_buckets"]
    confusing_pairs = [row for row in pair_rows if row["count"] >= 20]
    if not confusing_pairs:
        confusing_pairs = pair_rows
    confusing_pairs = sorted(
        confusing_pairs,
        key=lambda row: (
            -row["mean_pairwise_confusion"],
            -row["candidate_beats_gold_fraction"],
            -row["count"],
            row["gold_relation"],
            row["candidate_relation"],
        ),
    )

    def pct(value, denominator):
        return 100.0 * value / max(1, denominator)

    lines = [
        "# Offline Relation Confusion Analysis",
        "",
        f"Input: `{global_stats['input_file']}`",
        f"QA: {total_qa}; candidate rows: {global_stats['total_row_count']}; valid negatives: {total_valid}; invalid negatives: {global_stats['invalid_negative_count']}.",
        "Scores are descriptive outputs from the frozen I0 adapter. This analysis does not train or select a sampler.",
        "",
        "## A. Are most candidate relations easy?",
        f"At candidate level, {pct(gap_easy['candidate_count'], total_valid):.2f}% of valid negatives have gap >= 2, and {pct(confusion_easy['candidate_count'], total_valid):.2f}% have pairwise confusion < 0.05.",
        "This describes the full relation vocabulary; it is not a claim about unseen relations outside this vocabulary.",
        "",
        "## B. Is negative mass concentrated?",
        f"Top-3 cumulative negative mass: mean {top_mass['Top3']['mean']:.3f}, median {top_mass['Top3']['median']:.3f}.",
        f"Top-9 cumulative negative mass: mean {top_mass['Top9']['mean']:.3f}, median {top_mass['Top9']['median']:.3f}.",
        "",
        "## C. Is fixed K=9 reasonable descriptively?",
        f"K90 <= 9 for {pct(k_stats['K90']['count_le_9'], total_qa):.2f}% of QA; K95 <= 9 for {pct(k_stats['K95']['count_le_9'], total_qa):.2f}% of QA.",
        "This is coverage evidence only, not a sampler recommendation.",
        "",
        "## D. Do QA require different K?",
        f"K90 distribution: median {k_stats['K90']['median']:.2f}, p25 {k_stats['K90']['p25']:.2f}, p75 {k_stats['K90']['p75']:.2f}, p90 {k_stats['K90']['p90']:.2f}.",
        f"K95 distribution: median {k_stats['K95']['median']:.2f}, p25 {k_stats['K95']['p25']:.2f}, p75 {k_stats['K95']['p75']:.2f}, p90 {k_stats['K95']['p90']:.2f}.",
        "",
        "## E. How often does a negative beat gold?",
        f"Gap < 0 occurs for {gap_negative['candidate_count']} valid negatives ({pct(gap_negative['candidate_count'], total_valid):.2f}%) across {gap_negative['qa_count']} QA ({pct(gap_negative['qa_count'], total_qa):.2f}%).",
        f"The hardest valid negative beats gold for {hardest_gap_buckets['gap < 0']['qa_count']} QA ({pct(hardest_gap_buckets['gap < 0']['qa_count'], total_qa):.2f}%).",
        "",
        "## F. Which relation pairs are confused?",
        "The raw pair table is in `relation_pair_statistics.csv`. The five highest-mean-confusion pairs with at least 20 observations are:",
    ]
    for row in confusing_pairs[:5]:
        lines.append(
            f"- `{row['gold_relation']}` -> `{row['candidate_relation']}`: "
            f"sample_count={row['count']}, mean_confusion={row['mean_pairwise_confusion']:.4f}, "
            f"mean_gap={row['mean_score_gap']:.4f}, "
            f"beats_gold_rate={row['candidate_beats_gold_fraction']:.4f}."
        )
    lines.extend([
        "",
        "## Limits",
        "- False-negative filtering is based on the train-only graph metadata saved with the candidate rows.",
        "- The 237 relation IDs are represented by 233 unique text labels; label collisions remain collapsed.",
        "- Batch size 32 is the Prompt 3-compatible scoring configuration.",
    ])
    (output_dir / "research_questions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.candidate_scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = _find_scoring_metadata(input_path, args.metadata)
    context, base = _load_and_aggregate(input_path)
    per_qa, top_mass_stats, coverage = _compute_qa_coverage(context)

    total_qa = len(context["qa_states"])
    total_rows = context["total_rows"]
    valid_count = context["valid_negative_rows"]
    gap_bucket_rows = []
    for bucket in GAP_BUCKETS:
        data = context["gap_buckets"][bucket]
        gap_bucket_rows.append({
            "bucket": bucket,
            "candidate_count": data["candidate_count"],
            "candidate_fraction": data["candidate_count"] / max(1, valid_count),
            "qa_count": len(data["qa_ids"]),
            "qa_fraction": len(data["qa_ids"]) / max(1, total_qa),
        })
    confusion_bucket_rows = []
    for bucket in CONFUSION_BUCKETS:
        data = context["confusion_buckets"][bucket]
        confusion_bucket_rows.append({
            "bucket": bucket,
            "candidate_count": data["candidate_count"],
            "candidate_fraction": data["candidate_count"] / max(1, valid_count),
            "qa_count": len(data["qa_ids"]),
            "qa_fraction": len(data["qa_ids"]) / max(1, total_qa),
        })

    pair_rows = []
    for (gold, candidate), data in context["relation_pairs"].items():
        pair_rows.append({
            "gold_relation": gold,
            "candidate_relation": candidate,
            "count": data["count"],
            "mean_candidate_score": float(np.mean(data["scores"])),
            "mean_score_gap": float(np.mean(data["gaps"])),
            "median_score_gap": float(np.median(data["gaps"])),
            "mean_pairwise_confusion": float(np.mean(data["confusions"])),
            "median_pairwise_confusion": float(np.median(data["confusions"])),
            "p90_pairwise_confusion": float(np.percentile(data["confusions"], 90)),
            "candidate_beats_gold_count": data["beats_gold"],
            "candidate_beats_gold_fraction": data["beats_gold"] / max(1, data["count"]),
        })
    pair_rows.sort(key=lambda row: (-row["count"], row["gold_relation"], row["candidate_relation"]))

    hardest_gap_bucket_rows = []
    for bucket in GAP_BUCKETS:
        qa_count = sum(
            row["hardest_negative_gap"] is not None
            and _bucket_gap(row["hardest_negative_gap"]) == bucket
            for row in per_qa
        )
        hardest_gap_bucket_rows.append({
            "bucket": bucket,
            "qa_count": qa_count,
            "qa_fraction": qa_count / max(1, total_qa),
        })

    coverage_k_statistics = {}
    for key, values in coverage["coverage_values"].items():
        clean = [value for value in values if value is not None]
        stats = _stats(clean)
        for limit in (1, 3, 5, 9, 20):
            stats[f"count_le_{limit}"] = int(sum(value <= limit for value in clean))
            stats[f"fraction_le_{limit}"] = float(sum(value <= limit for value in clean) / max(1, len(clean)))
        coverage_k_statistics[key] = stats

    global_stats = {
        "input_file": str(input_path.resolve()),
        "qa_count": total_qa,
        "total_row_count": total_rows,
        "candidate_vocab_size": len(context["candidate_vocabulary"]),
        "gold_vocab_size": len(context["gold_vocabulary"]),
        "valid_negative_count": valid_count,
        "invalid_negative_count": context["invalid_negative_rows"],
        "negative_row_count": total_rows - total_qa,
        "valid_negative_fraction": valid_count / max(1, total_rows - total_qa),
        "valid_row_fraction_of_all_candidates": valid_count / max(1, total_rows),
        "valid_negative_metrics": {
            name: _stats(values)
            for name, values in context["metric_values"].items()
        },
        "gap_buckets": {
            row["bucket"]: row for row in gap_bucket_rows
        },
        "confusion_buckets": {
            row["bucket"]: row for row in confusion_bucket_rows
        },
        "hardest_gap": coverage["hardest_gap_stats"],
        "hardest_gap_buckets": {
            row["bucket"]: row for row in hardest_gap_bucket_rows
        },
        "top_n_negative_mass": top_mass_stats,
        "coverage_k_statistics": coverage_k_statistics,
        "input_checks": {
            "all_gold_counts_one": not any(error["error"] == "gold_count_not_one" for error in context["input_errors"]),
            "all_candidate_counts_equal_vocab": not any(error["error"] == "candidate_count_mismatch" for error in context["input_errors"]),
            "all_valid_negatives_filtered": not any(error["error"] == "unfiltered_true_relation" for error in context["input_errors"]),
            "input_error_count": len(context["input_errors"]),
            "input_errors_sample": context["input_errors"][:20],
        },
    }
    _write_json(output_dir / "global_statistics.json", global_stats)
    _write_csv(output_dir / "gap_buckets.csv", list(gap_bucket_rows[0]), gap_bucket_rows)
    _write_csv(output_dir / "hardest_gap_buckets.csv", list(hardest_gap_bucket_rows[0]), hardest_gap_bucket_rows)
    _write_csv(output_dir / "confusion_buckets.csv", list(confusion_bucket_rows[0]), confusion_bucket_rows)
    _write_csv(output_dir / "relation_pair_statistics.csv", list(pair_rows[0]), pair_rows)
    _write_csv(
        output_dir / "coverage_k_statistics.csv",
        ["coverage", "stat", "value"],
        [
            {"coverage": coverage_name, "stat": stat, "value": value}
            for coverage_name, stats in coverage_k_statistics.items()
            for stat, value in stats.items()
        ],
    )
    with (output_dir / "per_qa_coverage.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_qa:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    vocab_metadata = _build_relation_confusion_vocab(
        output_dir / "confusion_vocab",
        pair_rows,
        context,
        metadata_path,
        args.min_support,
        args.neighbor_top_k,
    )
    _make_figures(output_dir, context, per_qa, top_mass_stats, coverage)
    _write_report(output_dir, global_stats, pair_rows)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "confusion_vocab_dir": str((output_dir / "confusion_vocab").resolve()),
        "qa_count": total_qa,
        "total_rows": total_rows,
        "valid_negative_count": valid_count,
        "min_support": args.min_support,
        "raw_pair_count": vocab_metadata["raw_pair_count"],
        "display_pair_count": vocab_metadata["display_pair_count"],
        "input_error_count": len(context["input_errors"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
