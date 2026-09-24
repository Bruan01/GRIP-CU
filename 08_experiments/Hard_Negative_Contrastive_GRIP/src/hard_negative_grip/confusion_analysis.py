"""Offline analysis of saved full-vocabulary candidate scores.

This module never loads a language model. It reads either Prompt-4
``candidate_scores.jsonl`` rows or a frozen ``confusion_db.jsonl`` table and
describes the valid-negative confusion structure.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .offline_scoring import build_qa_score_rows

NUMERIC_FIELDS = (
    "candidate_score",
    "score_gap",
    "pairwise_confusion",
    "negative_mass",
    "negative_rank",
)
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
TOP_N = (1, 2, 3, 5, 9, 10, 20)
COVERAGE_LEVELS = (0.70, 0.80, 0.90, 0.95, 0.99)
K_REPORT = (0.80, 0.90, 0.95)
K_LEQ = (1, 3, 5, 9, 20)
HARDEST_GAP_CUTOFFS = (0.0, 0.1, 0.5, 1.0)

GAP_BUCKETS = (
    ("gap < 0", lambda gap: gap < 0),
    ("0 <= gap < 0.1", lambda gap: 0 <= gap < 0.1),
    ("0.1 <= gap < 0.5", lambda gap: 0.1 <= gap < 0.5),
    ("0.5 <= gap < 1", lambda gap: 0.5 <= gap < 1),
    ("1 <= gap < 2", lambda gap: 1 <= gap < 2),
    ("gap >= 2", lambda gap: gap >= 2),
)
CONFUSION_BUCKETS = (
    ("0 <= p < 0.05", lambda value: 0 <= value < 0.05),
    ("0.05 <= p < 0.10", lambda value: 0.05 <= value < 0.10),
    ("0.10 <= p < 0.20", lambda value: 0.10 <= value < 0.20),
    ("0.20 <= p < 0.30", lambda value: 0.20 <= value < 0.30),
    ("0.30 <= p < 0.40", lambda value: 0.30 <= value < 0.40),
    ("0.40 <= p <= 0.50", lambda value: 0.40 <= value <= 0.50),
    ("p > 0.50", lambda value: value > 0.50),
)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            **{f"p{p}": None for p in PERCENTILES},
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    summary = {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "median": percentile(values, 50),
        "min": min(values),
        "max": max(values),
    }
    for p in PERCENTILES:
        summary[f"p{p}"] = percentile(values, p)
    return summary


def k_for_coverage(masses_desc: list[float], target: float) -> int:
    """Smallest K such that the top-K negative masses cover ``target``."""
    if not masses_desc:
        return 0
    if target <= 0:
        return 0
    total = 0.0
    for index, mass in enumerate(masses_desc, start=1):
        total += mass
        if total >= target - 1e-12:
            return index
    return len(masses_desc)


def topn_cumulative(masses_desc: list[float], n: int) -> float:
    if n <= 0 or not masses_desc:
        return 0.0
    return float(sum(masses_desc[:n]))


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def expand_confusion_db_row(
    row: dict,
    *,
    relation_order: list[str],
    temperature: float = 1.0,
) -> tuple[list[dict], dict]:
    scores = {
        str(item["relation"]): float(item["score"]) for item in row["all_candidate_scores"]
    }
    pair = row.get("entity_pair")
    head = tail = None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        head, tail = str(pair[0]), str(pair[1])
    known = [str(rel) for rel in (row.get("known_pair_relations") or [])]
    dummy_lengths = {rel: 1 for rel in relation_order}
    return build_qa_score_rows(
        qa_id=str(row.get("question_id") or row.get("qa_id")),
        question=row.get("question"),
        head_entity=head,
        tail_entity=tail,
        gold_relation=str(row.get("positive_relation") or row.get("gold_relation")),
        matched_train_relation=str(row["matched_train_relation"]),
        relation_order=relation_order,
        scores=scores,
        token_lengths=dummy_lengths,
        known_relations=known,
        temperature=temperature,
        model_checkpoint=str(row.get("b1_adapter") or row.get("model_checkpoint") or ""),
        split=str(row.get("split") or "train"),
    )


def iter_qa_groups_from_scores(path: Path) -> Iterator[list[dict]]:
    current_id = None
    group: list[dict] = []
    for row in iter_jsonl(path):
        qa_id = str(row.get("qa_id") or "")
        if not qa_id:
            raise ValueError("candidate_scores.jsonl row is missing qa_id")
        if current_id is None:
            current_id = qa_id
        if qa_id != current_id:
            yield group
            group = []
            current_id = qa_id
        group.append(row)
    if group:
        yield group


def iter_qa_groups(
    *,
    scores_path: Path | None = None,
    confusion_db_path: Path | None = None,
    relation_order: list[str] | None = None,
    temperature: float = 1.0,
) -> Iterator[list[dict]]:
    if scores_path is not None:
        yield from iter_qa_groups_from_scores(scores_path)
        return
    if confusion_db_path is None or relation_order is None:
        raise ValueError("either scores_path or confusion_db_path+relation_order is required")
    for row in iter_jsonl(confusion_db_path):
        rows, _summary = expand_confusion_db_row(
            row, relation_order=relation_order, temperature=temperature
        )
        yield rows


def valid_negatives(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("is_valid_negative")]


def analyze_qa_group(rows: list[dict]) -> dict:
    valid = valid_negatives(rows)
    if not rows:
        raise ValueError("empty QA group")
    gold_row = next((row for row in rows if row.get("is_gold")), None)
    qa_id = str(rows[0]["qa_id"])
    gold_relation = str((gold_row or rows[0])["gold_relation"])
    masses = sorted(
        (float(row["negative_mass"]) for row in valid),
        reverse=True,
    )
    hardest = None
    if valid:
        hardest = max(
            valid,
            key=lambda row: (float(row["candidate_score"]), str(row["candidate_relation"])),
        )
    coverage = {f"K_{int(level * 100)}": k_for_coverage(masses, level) for level in COVERAGE_LEVELS}
    topn = {f"top{n}_mass": topn_cumulative(masses, n) for n in TOP_N}
    return {
        "qa_id": qa_id,
        "gold_relation": gold_relation,
        "n_valid_negatives": len(valid),
        "hardest_negative": None if hardest is None else str(hardest["candidate_relation"]),
        "hardest_negative_score": None if hardest is None else float(hardest["candidate_score"]),
        "hardest_negative_gap": None if hardest is None else float(hardest["score_gap"]),
        "hardest_pairwise_confusion": None
        if hardest is None
        else float(hardest["pairwise_confusion"]),
        **coverage,
        **topn,
        "valid": valid,
    }


def _bucket_counts(
    values: Iterable[tuple[str, float]],
    buckets: tuple[tuple[str, object], ...],
    n_qa: int,
) -> list[dict]:
    candidate_counts = {name: 0 for name, _pred in buckets}
    qa_sets = {name: set() for name, _pred in buckets}
    n_values = 0
    for qa_id, value in values:
        n_values += 1
        for name, predicate in buckets:
            if predicate(value):
                candidate_counts[name] += 1
                qa_sets[name].add(qa_id)
                break
    rows = []
    for name, _predicate in buckets:
        n_cand = candidate_counts[name]
        n_involved = len(qa_sets[name])
        rows.append(
            {
                "bucket": name,
                "candidate_count": n_cand,
                "candidate_fraction": (n_cand / n_values) if n_values else 0.0,
                "qa_count": n_involved,
                "qa_fraction": (n_involved / n_qa) if n_qa else 0.0,
            }
        )
    return rows


def k_leq_coverage(values: list[int], thresholds: Iterable[int]) -> dict:
    n = len(values)
    return {
        f"K_leq_{k}": {
            "qa_count": sum(1 for value in values if value <= k),
            "qa_fraction": (sum(1 for value in values if value <= k) / n) if n else 0.0,
        }
        for k in thresholds
    }


def analyze_score_groups(groups: Iterable[list[dict]]) -> dict:
    field_values = {field: [] for field in NUMERIC_FIELDS}
    gap_pairs: list[tuple[str, float]] = []
    confusion_pairs: list[tuple[str, float]] = []
    per_qa: list[dict] = []
    pair_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "count": 0,
            "score_sum": 0.0,
            "gap_sum": 0.0,
            "confusion_sum": 0.0,
            "confusions": [],
            "beats_gold": 0,
        }
    )
    n_qa = 0
    n_valid = 0
    n_candidates = 0

    for rows in groups:
        n_qa += 1
        n_candidates += len(rows)
        summary = analyze_qa_group(rows)
        valid = summary.pop("valid")
        n_valid += len(valid)
        per_qa.append(summary)
        for row in valid:
            qa_id = str(row["qa_id"])
            for field in NUMERIC_FIELDS:
                value = row.get(field)
                if value is None:
                    raise ValueError(f"{qa_id}: valid negative missing {field}")
                field_values[field].append(float(value))
            gap = float(row["score_gap"])
            confusion = float(row["pairwise_confusion"])
            gap_pairs.append((qa_id, gap))
            confusion_pairs.append((qa_id, confusion))
            key = (str(row["gold_relation"]), str(row["candidate_relation"]))
            bucket = pair_stats[key]
            bucket["count"] += 1
            bucket["score_sum"] += float(row["candidate_score"])
            bucket["gap_sum"] += gap
            bucket["confusion_sum"] += confusion
            bucket["confusions"].append(confusion)
            if gap < 0:
                bucket["beats_gold"] += 1

    hardest_gaps = [
        float(row["hardest_negative_gap"])
        for row in per_qa
        if row["hardest_negative_gap"] is not None
    ]
    topn_stats = {
        f"top{n}": summarize_selected(
            [float(row[f"top{n}_mass"]) for row in per_qa],
            ("mean", "median", "p25", "p75"),
        )
        for n in TOP_N
    }
    k_stats = {}
    for level in K_REPORT:
        key = f"K_{int(level * 100)}"
        values = [int(row[key]) for row in per_qa]
        summary = summarize_values([float(value) for value in values])
        k_stats[key] = {
            **{name: summary[name] for name in ("mean", "std", "median", "min", "max", "p25", "p75", "p90")},
            "count": len(values),
            **k_leq_coverage(values, K_LEQ),
        }

    pair_rows = []
    for (gold, candidate), bucket in pair_stats.items():
        count = bucket["count"]
        pair_rows.append(
            {
                "gold_relation": gold,
                "candidate_relation": candidate,
                "count": count,
                "mean_candidate_score": bucket["score_sum"] / count,
                "mean_score_gap": bucket["gap_sum"] / count,
                "mean_pairwise_confusion": bucket["confusion_sum"] / count,
                "median_pairwise_confusion": percentile(bucket["confusions"], 50),
                "fraction_candidate_beats_gold": bucket["beats_gold"] / count,
            }
        )
    pair_rows.sort(
        key=lambda row: (
            -int(row["count"]),
            -float(row["mean_pairwise_confusion"]),
            str(row["gold_relation"]),
            str(row["candidate_relation"]),
        )
    )

    return {
        "n_qa": n_qa,
        "n_candidate_rows": n_candidates,
        "n_valid_negatives": n_valid,
        "global_statistics": {field: summarize_values(values) for field, values in field_values.items()},
        "gap_buckets": _bucket_counts(gap_pairs, GAP_BUCKETS, n_qa),
        "confusion_buckets": _bucket_counts(confusion_pairs, CONFUSION_BUCKETS, n_qa),
        "hardest_negative_gap": {
            **summarize_values(hardest_gaps),
            "qa_hardest_gap_lt_0": sum(1 for value in hardest_gaps if value < 0),
            "qa_hardest_gap_lt_0.1": sum(1 for value in hardest_gaps if value < 0.1),
            "qa_hardest_gap_lt_0.5": sum(1 for value in hardest_gaps if value < 0.5),
            "qa_hardest_gap_lt_1": sum(1 for value in hardest_gaps if value < 1),
        },
        "topn_negative_mass": topn_stats,
        "coverage_k": k_stats,
        "per_qa": per_qa,
        "relation_pairs": pair_rows,
        "hardest_gaps": hardest_gaps,
        "k80": [int(row["K_80"]) for row in per_qa],
        "k90": [int(row["K_90"]) for row in per_qa],
        "k95": [int(row["K_95"]) for row in per_qa],
        "score_gaps": field_values["score_gap"],
        "pairwise_confusions": field_values["pairwise_confusion"],
    }


def summarize_selected(values: list[float], keys: Iterable[str]) -> dict:
    full = summarize_values(values)
    mapping = {
        "mean": "mean",
        "median": "median",
        "p25": "p25",
        "p75": "p75",
        "std": "std",
        "min": "min",
        "max": "max",
    }
    return {key: full[mapping[key]] for key in keys}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_histogram(path: Path, values: list[float], *, title: str, xlabel: str, bins: int = 40) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(values, bins=bins, color="#3b6ea5", edgecolor="white")
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_integer_histogram(path: Path, values: list[int], *, title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        maximum = 1
    else:
        maximum = max(values)
    bins = list(range(0, maximum + 2))
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(values, bins=bins, color="#3b6ea5", edgecolor="white", align="left")
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("QA count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_topn_mass_plot(path: Path, topn_stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(TOP_N)
    means = [topn_stats[f"top{n}"]["mean"] for n in xs]
    medians = [topn_stats[f"top{n}"]["median"] for n in xs]
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(xs, means, marker="o", label="mean")
    axes.plot(xs, medians, marker="s", label="median")
    axes.set_title("Cumulative valid-negative mass vs Top-N")
    axes.set_xlabel("Top-N valid negatives")
    axes.set_ylabel("cumulative negative_mass")
    axes.set_ylim(0, 1.05)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_analysis_outputs(result: dict, output_dir: Path, *, source: str) -> dict:
    figures = output_dir / "figures"
    global_payload = {
        "source": source,
        "n_qa": result["n_qa"],
        "n_candidate_rows": result["n_candidate_rows"],
        "n_valid_negatives": result["n_valid_negatives"],
        "metrics": result["global_statistics"],
        "hardest_negative_gap": result["hardest_negative_gap"],
        "topn_negative_mass": result["topn_negative_mass"],
        "coverage_k": result["coverage_k"],
    }
    write_json(output_dir / "global_statistics.json", global_payload)
    write_csv(
        output_dir / "gap_buckets.csv",
        result["gap_buckets"],
        ["bucket", "candidate_count", "candidate_fraction", "qa_count", "qa_fraction"],
    )
    write_csv(
        output_dir / "confusion_buckets.csv",
        result["confusion_buckets"],
        ["bucket", "candidate_count", "candidate_fraction", "qa_count", "qa_fraction"],
    )
    coverage_rows = []
    for key, payload in result["coverage_k"].items():
        row = {"metric": key}
        row.update({name: payload[name] for name in ("mean", "std", "median", "min", "max", "p25", "p75", "p90")})
        for k in K_LEQ:
            row[f"qa_fraction_K_leq_{k}"] = payload[f"K_leq_{k}"]["qa_fraction"]
            row[f"qa_count_K_leq_{k}"] = payload[f"K_leq_{k}"]["qa_count"]
        coverage_rows.append(row)
    write_csv(
        output_dir / "coverage_k_statistics.csv",
        coverage_rows,
        [
            "metric",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "p25",
            "p75",
            "p90",
            *[item for k in K_LEQ for item in (f"qa_count_K_leq_{k}", f"qa_fraction_K_leq_{k}")],
        ],
    )
    write_jsonl(output_dir / "per_qa_coverage.jsonl", result["per_qa"])
    write_csv(
        output_dir / "relation_pair_statistics.csv",
        result["relation_pairs"],
        [
            "gold_relation",
            "candidate_relation",
            "count",
            "mean_candidate_score",
            "mean_score_gap",
            "mean_pairwise_confusion",
            "median_pairwise_confusion",
            "fraction_candidate_beats_gold",
        ],
    )
    save_histogram(
        figures / "score_gap_distribution.png",
        result["score_gaps"],
        title="Valid-negative score_gap",
        xlabel="gold_score - candidate_score",
    )
    save_histogram(
        figures / "pairwise_confusion_distribution.png",
        result["pairwise_confusions"],
        title="Valid-negative pairwise_confusion",
        xlabel="P(candidate beats gold)",
        bins=50,
    )
    save_integer_histogram(
        figures / "k80_distribution.png",
        result["k80"],
        title="K80: negatives needed for 80% negative mass",
        xlabel="K80",
    )
    save_integer_histogram(
        figures / "k90_distribution.png",
        result["k90"],
        title="K90: negatives needed for 90% negative mass",
        xlabel="K90",
    )
    save_integer_histogram(
        figures / "k95_distribution.png",
        result["k95"],
        title="K95: negatives needed for 95% negative mass",
        xlabel="K95",
    )
    save_topn_mass_plot(figures / "cumulative_negative_mass_vs_topn.png", result["topn_negative_mass"])
    return global_payload
