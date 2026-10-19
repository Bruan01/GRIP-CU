"""Freeze train-only GRIP-conditioned error-memory negative manifests.

The sampler is deliberately a data-freezing step, not a training-time policy:

* the error memory is counted from frozen B1 free-generation mistakes on the
  training QA split only;
* the candidate pool comes from the frozen B1 score dump after train-only
  known-fact filtering;
* candidates are ranked by a gold-conditioned error propensity plus score
  violation, with a uniform probability floor;
* the resulting K=9 lists are immutable JSONL rows consumed by Stage 2.

Keeping the memory and the score dump separate makes the method auditable and
prevents validation/test predictions from leaking into the negative bank.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .offline_mining import iter_qa_groups_from_scores
from .task_file import (
    canonical_rollout_candidate,
    match_train_relation,
    parse_task_qa_id,
    train_relation_alias_index,
)

DEFAULT_TOTAL_K = 9
DEFAULT_UNIFORM_MIX = 1.0 / 3.0
DEFAULT_ERROR_WEIGHT = 1.0
DEFAULT_VIOLATION_WEIGHT = 1.0
DEFAULT_SAMPLING_TEMPERATURE = 0.25
ERROR_MEMORY_SOURCE = "train_only_error_memory_score_violation_uniform_floor"
ERROR_MEMORY_SELECTION_RULE = (
    "K draws without replacement from p(r)=uniform_mix/|P| + "
    "(1-uniform_mix)*softmax((error_score+score_violation)/temperature)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no JSONL rows")
    return rows


def _rollout_candidate(row: dict) -> str:
    """Return the one frozen rollout prediction recorded for a QA row."""
    for key in ("rollout_hard_negative", "rollout_prediction", "rollout_parsed"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _mapped_rollout_relation(candidate: str, alias_index: dict[str, str]) -> str | None:
    canonical = canonical_rollout_candidate(candidate)
    if canonical:
        mapped = match_train_relation(canonical, alias_index)
        if mapped is not None:
            return mapped
    # This fallback keeps the freezer compatible with manifests that already
    # stored a relation name rather than the canonical ``concept:`` spelling.
    return match_train_relation(candidate, alias_index)


def build_error_memory(
    rollout_rows: Iterable[dict],
    relation_order: list[str],
    *,
    min_support: int = 1,
) -> dict:
    """Build a gold-conditioned train-only confusion memory.

    ``error_score(gold, candidate)`` is later normalized by the largest
    observed count for that gold relation. This preserves the ranking of
    repeated mistakes without letting frequent gold relations dominate other
    QAs.
    """
    if not relation_order:
        raise ValueError("relation_order must be non-empty")
    if min_support < 1:
        raise ValueError("min_support must be positive")
    alias_index = train_relation_alias_index(relation_order)
    relation_set = set(relation_order)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    gold_support: Counter[str] = Counter()
    stats = Counter()
    seen_ids: set[str] = set()

    for row in rollout_rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError("rollout row is missing question_id")
        if question_id in seen_ids:
            raise ValueError(f"duplicate rollout question_id {question_id!r}")
        seen_ids.add(question_id)
        if str(row.get("split") or "train") != "train":
            raise ValueError(f"{question_id}: error memory accepts train rows only")

        gold = str(row.get("matched_train_relation") or "")
        if gold not in relation_set:
            gold = match_train_relation(str(row.get("positive_relation") or ""), alias_index) or ""
        if not gold:
            stats["unmatched_gold"] += 1
            continue
        stats["train_rows"] += 1
        candidate = _rollout_candidate(row)
        if not candidate:
            stats["no_relation_prediction"] += 1
            continue
        mapped = _mapped_rollout_relation(candidate, alias_index)
        if mapped is None:
            stats["oov_or_noncanonical_errors"] += 1
            if canonical_rollout_candidate(candidate):
                stats["oov_errors"] += 1
            else:
                stats["noncanonical_errors"] += 1
            continue
        if mapped == gold:
            stats["gold_predictions"] += 1
            continue
        if row.get("rollout_hard_is_train_vocab") is False and mapped not in relation_set:
            stats["oov_or_noncanonical_errors"] += 1
            stats["oov_errors"] += 1
            continue
        counts[gold][mapped] += 1
        gold_support[gold] += 1
        stats["train_relation_errors"] += 1

    filtered = {
        gold: {candidate: int(count) for candidate, count in counter.items() if count >= min_support}
        for gold, counter in counts.items()
    }
    filtered = {gold: values for gold, values in filtered.items() if values}
    return {
        "relation_order": list(relation_order),
        "min_support": min_support,
        "counts": filtered,
        "gold_support": {gold: int(value) for gold, value in gold_support.items()},
        "n_gold_relations_with_errors": len(filtered),
        "n_error_pairs": sum(len(values) for values in filtered.values()),
        "stats": dict(stats),
    }


def _error_score(error_memory: dict, gold: str, candidate: str) -> tuple[int, float]:
    counts = {
        str(key): int(value)
        for key, value in (error_memory.get("counts", {}).get(gold, {}) or {}).items()
    }
    count = int(counts.get(candidate, 0))
    maximum = max(counts.values(), default=0)
    return count, (float(count) / float(maximum) if maximum else 0.0)


def _score_violation(row: dict, *, temperature: float) -> float:
    value = row.get("pairwise_confusion")
    if value is not None:
        return min(max(float(value), 0.0), 1.0)
    gap = row.get("score_gap")
    if gap is None:
        gold_score = row.get("gold_score")
        candidate_score = row.get("candidate_score")
        if gold_score is None or candidate_score is None:
            return 0.0
        gap = float(gold_score) - float(candidate_score)
    scaled = -float(gap) / float(temperature)
    if scaled >= 0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exponent = math.exp(scaled)
    return exponent / (1.0 + exponent)


def _stable_softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    scaled = [float(value) / temperature for value in values]
    pivot = max(scaled)
    weights = [math.exp(value - pivot) for value in scaled]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        return [1.0 / len(values)] * len(values)
    return [weight / total for weight in weights]


def _draw_without_replacement(
    candidates: list[dict],
    *,
    k: int,
    rng: random.Random,
) -> list[dict]:
    remaining = list(candidates)
    selected: list[dict] = []
    while remaining and len(selected) < k:
        weights = [max(float(item["sampling_probability"]), 0.0) for item in remaining]
        total = sum(weights)
        if total <= 0:
            index = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            index = len(remaining) - 1
            for candidate_index, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    index = candidate_index
                    break
        selected.append(remaining.pop(index))
    return selected


def sample_error_memory_negatives(
    rows: list[dict],
    error_memory: dict,
    *,
    total_k: int = DEFAULT_TOTAL_K,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    error_weight: float = DEFAULT_ERROR_WEIGHT,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    seed: int = 2026,
) -> tuple[dict, dict]:
    """Sample one immutable error-memory row from a frozen score group."""
    if total_k < 1:
        raise ValueError("total_k must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    if error_weight < 0 or violation_weight < 0:
        raise ValueError("sampling weights must be non-negative")
    if error_weight == 0 and violation_weight == 0:
        raise ValueError("at least one sampling weight must be positive")
    if sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if not rows:
        raise ValueError("score group is empty")
    gold_rows = [row for row in rows if bool(row.get("is_gold"))]
    if len(gold_rows) != 1:
        raise ValueError(f"score group must contain exactly one gold row, got {len(gold_rows)}")
    gold_row = gold_rows[0]
    qa_id = str(gold_row.get("qa_id") or "")
    gold = str(gold_row.get("matched_train_relation") or gold_row.get("gold_relation") or "")
    if not qa_id or not gold:
        raise ValueError("score group is missing qa_id or gold relation")
    if str(gold_row.get("split") or "train") != "train":
        raise ValueError(f"{qa_id}: score group is not train-only")

    candidates: list[dict] = []
    for row in rows:
        if not bool(row.get("is_valid_negative")):
            continue
        relation = str(row.get("candidate_relation") or "")
        if not relation or relation == gold:
            continue
        count, error_score = _error_score(error_memory, gold, relation)
        violation = _score_violation(row, temperature=sampling_temperature)
        utility = error_weight * error_score + violation_weight * violation
        candidates.append(
            {
                "relation": relation,
                "error_count": count,
                "error_score": error_score,
                "score_violation": violation,
                "utility": utility,
                "candidate_score": float(row.get("candidate_score", 0.0)),
                "gold_score": float(row.get("gold_score", gold_row.get("gold_score", 0.0))),
                "score_gap": float(row.get("score_gap", 0.0)),
                "candidate_rank_all": row.get("candidate_rank_all"),
            }
        )
    if not candidates:
        raise ValueError(f"{qa_id}: no valid train-only negative candidates")

    softmax = _stable_softmax([item["utility"] for item in candidates], sampling_temperature)
    pool_size = len(candidates)
    for item, soft_probability in zip(candidates, softmax):
        item["sampling_probability"] = (
            uniform_mix / pool_size + (1.0 - uniform_mix) * soft_probability
        )
    qa_seed = int(seed) + parse_task_qa_id(qa_id) * 1_000_003
    selected = _draw_without_replacement(
        candidates,
        k=min(total_k, pool_size),
        rng=random.Random(qa_seed),
    )
    negatives = [str(item["relation"]) for item in selected]
    selected_rows = [
        {
            key: value
            for key, value in item.items()
            if key != "utility"
        }
        for item in selected
    ]
    support = int(sum(
        int(value)
        for value in (error_memory.get("counts", {}).get(gold, {}) or {}).values()
    ))
    manifest = {
        "question_id": qa_id,
        "split": "train",
        "positive_relation": str(gold_row.get("gold_relation") or gold),
        "matched_train_relation": gold,
        "gold_relation": gold,
        "negative_relations": negatives,
        "hard_negative_relations": negatives,
        "uniform_negative_relations": [],
        "selected_negatives": selected_rows,
        "negative_source": ERROR_MEMORY_SOURCE,
        "selection_rule": ERROR_MEMORY_SELECTION_RULE,
        "filter_splits": ["train"],
        "k": len(negatives),
        "actual_k": len(negatives),
        "requested_k": total_k,
        "uniform_mix": uniform_mix,
        "error_weight": error_weight,
        "violation_weight": violation_weight,
        "sampling_temperature": sampling_temperature,
        "error_memory_support": support,
        "error_memory_candidates": int(
            len(error_memory.get("counts", {}).get(gold, {}) or {})
        ),
        "seed": int(seed),
    }
    audit = {
        "question_id": qa_id,
        "gold_relation": gold,
        "pool_size": pool_size,
        "selected_k": len(negatives),
        "error_memory_support": support,
        "selected_error_memory_hits": sum(1 for item in selected if item["error_count"] > 0),
        "selected_mean_score_violation": (
            sum(float(item["score_violation"]) for item in selected) / len(selected)
            if selected
            else 0.0
        ),
        "pool_mean_score_violation": sum(
            float(item["score_violation"]) for item in candidates
        ) / len(candidates),
    }
    return manifest, audit


def _validate_source_metadata(metadata: dict, *, metadata_path: Path) -> None:
    if str(metadata.get("split")) != "train":
        raise ValueError(f"{metadata_path}: score metadata split must be train")
    if list(metadata.get("filter_splits") or []) != ["train"]:
        raise ValueError(
            f"{metadata_path}: error memory requires filter_splits=['train']; "
            f"got {metadata.get('filter_splits')!r}"
        )


def freeze_error_memory_bank(
    *,
    scores_path: Path,
    metadata_path: Path,
    rollout_path: Path,
    output_dir: Path,
    relation_order: list[str],
    total_k: int = DEFAULT_TOTAL_K,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    error_weight: float = DEFAULT_ERROR_WEIGHT,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    seed: int = 2026,
    min_support: int = 1,
    overwrite: bool = False,
) -> dict:
    """Create the frozen JSONL bank and machine-readable audit files."""
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not rollout_path.is_file():
        raise FileNotFoundError(rollout_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be an object")
    _validate_source_metadata(metadata, metadata_path=metadata_path)

    groups = list(iter_qa_groups_from_scores(scores_path))
    if not groups:
        raise ValueError(f"{scores_path} contains no score groups")
    score_ids = {str(group[0].get("qa_id") or "") for group in groups}
    if "" in score_ids:
        raise ValueError("score dump contains a group without qa_id")
    for group in groups:
        if any(str(row.get("split") or "train") != "train" for row in group):
            raise ValueError(f"{group[0].get('qa_id')}: score dump is not train-only")
        if len(group) != len(relation_order):
            raise ValueError(
                f"{group[0].get('qa_id')}: expected {len(relation_order)} candidate rows, "
                f"got {len(group)}"
            )

    rollout_rows = load_jsonl(rollout_path)
    rollout_by_id: dict[str, dict] = {}
    for row in rollout_rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"{rollout_path}: rollout row missing question_id")
        if question_id in rollout_by_id:
            raise ValueError(f"{rollout_path}: duplicate question_id {question_id!r}")
        rollout_by_id[question_id] = row
    if set(rollout_by_id) != score_ids:
        missing = sorted(score_ids.difference(rollout_by_id))
        extra = sorted(set(rollout_by_id).difference(score_ids))
        raise ValueError(
            "rollout and score IDs differ: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    error_memory = build_error_memory(
        rollout_rows,
        relation_order,
        min_support=min_support,
    )
    manifests: list[dict] = []
    audits: list[dict] = []
    for group in groups:
        manifest, audit = sample_error_memory_negatives(
            group,
            error_memory,
            total_k=total_k,
            uniform_mix=uniform_mix,
            error_weight=error_weight,
            violation_weight=violation_weight,
            sampling_temperature=sampling_temperature,
            seed=seed,
        )
        manifests.append(manifest)
        audits.append(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "manifest": output_dir / "error_memory_k.jsonl",
        "memory": output_dir / "error_memory.json",
        "policy": output_dir / "error_memory_policy.json",
        "audit": output_dir / "sampler_audit.json",
    }
    if not overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite frozen error-memory outputs: " + ", ".join(existing)
            )

    selected_hits = [item["selected_error_memory_hits"] for item in audits]
    selected_violation = [item["selected_mean_score_violation"] for item in audits]
    pool_violation = [item["pool_mean_score_violation"] for item in audits]
    pair_counts = Counter()
    for gold, values in (error_memory.get("counts") or {}).items():
        for candidate, count in values.items():
            pair_counts[f"{gold} -> {candidate}"] += int(count)
    policy = {
        "sampler": "GRIP-conditioned error-memory negative sampling",
        "negative_source": ERROR_MEMORY_SOURCE,
        "selection_rule": ERROR_MEMORY_SELECTION_RULE,
        "filter_splits": ["train"],
        "n_qa": len(manifests),
        "relation_vocab_size": len(relation_order),
        "requested_k": total_k,
        "uniform_mix": uniform_mix,
        "error_weight": error_weight,
        "violation_weight": violation_weight,
        "sampling_temperature": sampling_temperature,
        "seed": seed,
        "min_support": min_support,
        "source_scores": str(scores_path),
        "source_scores_sha256": file_sha256(scores_path),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": file_sha256(metadata_path),
        "source_rollout_manifest": str(rollout_path),
        "source_rollout_manifest_sha256": file_sha256(rollout_path),
        "error_memory": error_memory,
        "top_error_pairs": [
            {"pair": pair, "count": count}
            for pair, count in pair_counts.most_common(20)
        ],
        "training_contract": {
            "prompts_unchanged": True,
            "loss_unchanged": "generation + lambda_candidate * InfoNCE",
            "decision_set": "gold + exactly K train-vocabulary negatives where available",
            "frozen_before_stage2": True,
        },
    }
    audit = {
        "n_qa": len(audits),
        "n_manifest_rows": len(manifests),
        "actual_k": {
            "min": min(len(row["negative_relations"]) for row in manifests),
            "max": max(len(row["negative_relations"]) for row in manifests),
            "mean": sum(len(row["negative_relations"]) for row in manifests) / len(manifests),
        },
        "selected_error_memory_hits": {
            "mean": sum(selected_hits) / len(selected_hits),
            "zero_support_qa": sum(1 for value in selected_hits if value == 0),
        },
        "score_violation": {
            "selected_mean": sum(selected_violation) / len(selected_violation),
            "pool_mean": sum(pool_violation) / len(pool_violation),
        },
        "error_memory_stats": error_memory.get("stats", {}),
        "top_error_pairs": policy["top_error_pairs"],
        "source_scores_sha256": policy["source_scores_sha256"],
        "source_rollout_manifest_sha256": policy["source_rollout_manifest_sha256"],
    }

    with output_paths["manifest"].open("w", encoding="utf-8") as stream:
        for row in manifests:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_paths["memory"].write_text(
        json.dumps(error_memory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths["policy"].write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(output_paths["manifest"]),
        "memory": str(output_paths["memory"]),
        "policy": str(output_paths["policy"]),
        "audit": str(output_paths["audit"]),
        "manifest_sha256": file_sha256(output_paths["manifest"]),
        "n_qa": len(manifests),
        "actual_k": audit["actual_k"],
        "error_memory_stats": error_memory.get("stats", {}),
    }
