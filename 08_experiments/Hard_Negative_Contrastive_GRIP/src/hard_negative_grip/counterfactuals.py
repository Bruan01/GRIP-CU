"""Train-only structure-aware counterfactual relation negatives.

This module implements the first executable slice of the selected method idea.
For a relation QA item, a candidate relation is treated as a counterfactual
edge-label assignment when it is absent from the train-known relation set for
the same entity pair.  Candidates are ranked with two independent signals:

* the frozen B1 score violation, and
* similarity between the candidate and gold relation's train-graph profiles.

The resulting bank is immutable JSONL and keeps enough provenance to audit
the candidate pool, the structural score, and the train-only false-fact
filter.  It deliberately does not claim that an absent train fact is false in
the open-world KG; it records the narrower, reproducible train-only contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .official_lists import load_relation_text_mapping, load_triples


STRUCTURE_COUNTERFACTUAL_SOURCE = (
    "train_only_structure_counterfactual_score_violation_uniform_floor"
)
STRUCTURE_COUNTERFACTUAL_SELECTION_RULE = (
    "K draws without replacement from p(r)=uniform_mix/|P| + "
    "(1-uniform_mix)*softmax((structure_similarity*structure_weight + "
    "score_violation*violation_weight)/temperature)"
)
DEFAULT_TOTAL_K = 9
DEFAULT_UNIFORM_MIX = 1.0 / 3.0
DEFAULT_STRUCTURE_WEIGHT = 1.0
DEFAULT_VIOLATION_WEIGHT = 1.0
DEFAULT_TEMPERATURE = 0.25


@dataclass(frozen=True)
class RelationProfile:
    """Compact train-graph signature for one relation label."""

    relation: str
    edge_count: int
    unique_heads: int
    unique_tails: int
    mean_head_out_degree: float
    mean_tail_in_degree: float

    @property
    def features(self) -> tuple[float, ...]:
        # Log compression prevents high-frequency relations from dominating
        # the similarity score solely because of graph size.
        return (
            math.log1p(self.edge_count),
            math.log1p(self.unique_heads),
            math.log1p(self.unique_tails),
            math.log1p(self.mean_head_out_degree),
            math.log1p(self.mean_tail_in_degree),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "edge_count": self.edge_count,
            "unique_heads": self.unique_heads,
            "unique_tails": self.unique_tails,
            "mean_head_out_degree": self.mean_head_out_degree,
            "mean_tail_in_degree": self.mean_tail_in_degree,
        }


def normalize_relation_label(
    relation: str,
    *,
    relation_mapping: Mapping[str, str] | None = None,
    strip_first_underscore: bool = False,
) -> str:
    """Map a raw relation ID to the frozen task label."""
    value = str(relation).strip()
    if relation_mapping is not None:
        value = str(relation_mapping.get(value, value)).strip()
    if strip_first_underscore and "_" in value:
        value = " ".join(value.split("_")[1:]).strip()
    if not value:
        raise ValueError("relation labels must be non-empty")
    return value


def build_relation_profiles(
    triples: Iterable[tuple[str, str, str]],
    *,
    relation_mapping: Mapping[str, str] | None = None,
    strip_first_underscore: bool = False,
) -> dict[str, RelationProfile]:
    """Build deterministic relation signatures from a train triple stream."""
    normalized: list[tuple[str, str, str]] = []
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    relation_edges: defaultdict[str, int] = defaultdict(int)
    relation_heads: defaultdict[str, set[str]] = defaultdict(set)
    relation_tails: defaultdict[str, set[str]] = defaultdict(set)

    for head, raw_relation, tail in triples:
        head = str(head).strip()
        tail = str(tail).strip()
        if not head or not tail:
            raise ValueError("triple entities must be non-empty")
        relation = normalize_relation_label(
            raw_relation,
            relation_mapping=relation_mapping,
            strip_first_underscore=strip_first_underscore,
        )
        normalized.append((head, relation, tail))
        out_degree[head] += 1
        in_degree[tail] += 1
        relation_edges[relation] += 1
        relation_heads[relation].add(head)
        relation_tails[relation].add(tail)

    if not normalized:
        raise ValueError("train triples are empty")

    profiles: dict[str, RelationProfile] = {}
    for relation in relation_edges:
        heads = relation_heads[relation]
        tails = relation_tails[relation]
        mean_head_degree = sum(out_degree[item] for item in heads) / len(heads)
        mean_tail_degree = sum(in_degree[item] for item in tails) / len(tails)
        profiles[relation] = RelationProfile(
            relation=relation,
            edge_count=relation_edges[relation],
            unique_heads=len(heads),
            unique_tails=len(tails),
            mean_head_out_degree=mean_head_degree,
            mean_tail_in_degree=mean_tail_degree,
        )
    return profiles


def load_train_relation_profiles(
    raw_dir: Path,
    *,
    relation_text_file: Path | None = None,
    strip_first_underscore: bool = False,
) -> dict[str, RelationProfile]:
    """Load ``train.txt`` and build the profiles used by the frozen bank."""
    mapping = load_relation_text_mapping(relation_text_file)
    triples = load_triples(raw_dir / "train.txt")
    return build_relation_profiles(
        triples,
        relation_mapping=mapping,
        strip_first_underscore=strip_first_underscore,
    )


def relation_profile_similarity(
    left: RelationProfile | None,
    right: RelationProfile | None,
) -> float:
    """Return a bounded similarity based on normalized profile distance."""
    if left is None or right is None:
        return 0.0
    distances = [
        abs(a - b) / (1.0 + abs(a) + abs(b))
        for a, b in zip(left.features, right.features)
    ]
    return max(0.0, min(1.0, 1.0 - sum(distances) / len(distances)))


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def score_violation(row: Mapping[str, object], *, temperature: float) -> float:
    """Estimate the frozen B1 probability that a candidate beats gold."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if row.get("pairwise_confusion") is not None:
        return max(0.0, min(1.0, float(row["pairwise_confusion"])))
    candidate_score = row.get("candidate_score")
    gold_score = row.get("gold_score")
    if candidate_score is None or gold_score is None:
        return 0.0
    return _stable_sigmoid(
        (float(candidate_score) - float(gold_score)) / float(temperature)
    )


def _stable_softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(value) / temperature for value in values]
    pivot = max(scaled)
    weights = [math.exp(value - pivot) for value in scaled]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        return [1.0 / len(values)] * len(values)
    return [weight / total for weight in weights]


def _stable_seed(seed: int, question_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _draw_without_replacement(
    candidates: list[dict],
    *,
    k: int,
    rng: random.Random,
) -> list[dict]:
    remaining = list(candidates)
    selected: list[dict] = []
    while remaining and len(selected) < k:
        weights = [max(float(row["sampling_probability"]), 0.0) for row in remaining]
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


def sample_structure_counterfactual_group(
    rows: list[dict],
    profiles: Mapping[str, RelationProfile],
    *,
    total_k: int = DEFAULT_TOTAL_K,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 2026,
    known_relations: Iterable[str] = (),
) -> tuple[dict, dict]:
    """Create one immutable structure-counterfactual bank row and its audit."""
    if total_k < 1:
        raise ValueError("total_k must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    if structure_weight < 0 or violation_weight < 0:
        raise ValueError("sampling weights must be non-negative")
    if structure_weight == 0 and violation_weight == 0:
        raise ValueError("at least one sampling weight must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not rows:
        raise ValueError("score group is empty")
    if any(str(row.get("split") or "train") != "train" for row in rows):
        raise ValueError("structure counterfactual bank accepts train rows only")

    gold_rows = [row for row in rows if bool(row.get("is_gold"))]
    if len(gold_rows) != 1:
        raise ValueError(f"score group must contain exactly one gold row, got {len(gold_rows)}")
    gold_row = gold_rows[0]
    question_id = str(gold_row.get("qa_id") or "")
    gold = str(
        gold_row.get("matched_train_relation")
        or gold_row.get("gold_relation")
        or ""
    )
    if not question_id or not gold:
        raise ValueError("score group is missing qa_id or gold relation")

    known = {str(relation) for relation in known_relations}
    gold_profile = profiles.get(gold)
    candidates: list[dict] = []
    for row in rows:
        relation = str(row.get("candidate_relation") or "")
        if not relation or relation == gold or not bool(row.get("is_valid_negative")):
            continue
        if relation in known:
            # Keep a second defensive filter even if the score dump was already
            # labelled, so the bank cannot silently re-admit a known train fact.
            continue
        structure = relation_profile_similarity(gold_profile, profiles.get(relation))
        violation = score_violation(row, temperature=temperature)
        utility = structure_weight * structure + violation_weight * violation
        candidates.append(
            {
                "relation": relation,
                "structure_similarity": structure,
                "score_violation": violation,
                "utility": utility,
                "candidate_score": float(row.get("candidate_score", 0.0)),
                "gold_score": float(row.get("gold_score", gold_row.get("gold_score", 0.0))),
                "score_gap": float(row.get("score_gap", 0.0)),
                "candidate_rank_all": row.get("candidate_rank_all"),
                "counterfactual_valid": True,
                "counterfactual_operation": "replace_relation_on_observed_entity_pair",
            }
        )

    if len(candidates) < total_k:
        raise ValueError(
            f"{question_id}: only {len(candidates)} valid counterfactual candidates; "
            f"requested {total_k}"
        )

    softmax = _stable_softmax(
        [float(item["utility"]) for item in candidates], temperature
    )
    pool_size = len(candidates)
    for item, probability in zip(candidates, softmax):
        item["sampling_probability"] = (
            uniform_mix / pool_size + (1.0 - uniform_mix) * probability
        )

    selected = _draw_without_replacement(
        candidates,
        k=total_k,
        rng=random.Random(_stable_seed(seed, question_id)),
    )
    negatives = [str(item["relation"]) for item in selected]
    selected_rows = [
        {key: value for key, value in item.items() if key != "utility"}
        for item in selected
    ]
    manifest = {
        "question_id": question_id,
        "split": "train",
        "positive_relation": str(gold_row.get("gold_relation") or gold),
        "matched_train_relation": gold,
        "gold_relation": gold,
        "negative_relations": negatives,
        "hard_negative_relations": negatives,
        "uniform_negative_relations": [],
        "selected_negatives": selected_rows,
        "negative_source": STRUCTURE_COUNTERFACTUAL_SOURCE,
        "selection_rule": STRUCTURE_COUNTERFACTUAL_SELECTION_RULE,
        "filter_splits": ["train"],
        "counterfactual_operation": "replace_relation_on_observed_entity_pair",
        "counterfactual_validation": "candidate_absent_from_train_known_pair",
        "k": len(negatives),
        "actual_k": len(negatives),
        "requested_k": total_k,
        "uniform_mix": uniform_mix,
        "structure_weight": structure_weight,
        "violation_weight": violation_weight,
        "sampling_temperature": temperature,
        "seed": int(seed),
    }
    audit = {
        "question_id": question_id,
        "gold_relation": gold,
        "pool_size": pool_size,
        "selected_k": len(negatives),
        "selected_mean_structure_similarity": sum(
            float(item["structure_similarity"]) for item in selected
        ) / len(selected),
        "pool_mean_structure_similarity": sum(
            float(item["structure_similarity"]) for item in candidates
        ) / len(candidates),
        "selected_mean_score_violation": sum(
            float(item["score_violation"]) for item in selected
        ) / len(selected),
        "pool_mean_score_violation": sum(
            float(item["score_violation"]) for item in candidates
        ) / len(candidates),
        "unknown_profile_candidates": sum(
            1 for item in candidates if item["structure_similarity"] == 0.0
        ),
    }
    return manifest, audit


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def iter_qa_groups(rows: Iterable[dict]) -> Iterator[list[dict]]:
    current_id: str | None = None
    group: list[dict] = []
    for row in rows:
        question_id = str(row.get("qa_id") or "")
        if not question_id:
            raise ValueError("candidate score row is missing qa_id")
        if current_id is None:
            current_id = question_id
        if question_id != current_id:
            yield group
            group = []
            current_id = question_id
        group.append(row)
    if group:
        yield group


def write_bank_artifacts(
    *,
    score_path: Path,
    output_dir: Path,
    profiles: Mapping[str, RelationProfile],
    total_k: int = DEFAULT_TOTAL_K,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict:
    """Freeze a score dump into JSONL bank, audit, and metadata files."""
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    score_rows = load_jsonl(score_path)
    groups = list(iter_qa_groups(score_rows))
    if not groups:
        raise ValueError(f"{score_path} contains no score groups")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "manifest": output_dir / "counterfactual_bank.jsonl",
        "audit": output_dir / "sampler_audit.jsonl",
        "profiles": output_dir / "relation_profiles.json",
        "metadata": output_dir / "metadata.json",
    }
    if not overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite frozen structure-counterfactual outputs: "
                + ", ".join(existing)
            )

    bank_path = output_paths["manifest"]
    audit_path = output_paths["audit"]
    manifest_rows: list[dict] = []
    audit_rows: list[dict] = []
    failures: list[dict] = []
    for group in groups:
        try:
            manifest, audit = sample_structure_counterfactual_group(
                group,
                profiles,
                total_k=total_k,
                uniform_mix=uniform_mix,
                structure_weight=structure_weight,
                violation_weight=violation_weight,
                temperature=temperature,
                seed=seed,
            )
        except ValueError as error:
            failures.append({"question_id": group[0].get("qa_id"), "error": str(error)})
            continue
        manifest_rows.append(manifest)
        audit_rows.append(audit)

    if not manifest_rows:
        raise ValueError("no score groups produced a valid counterfactual bank row")

    with bank_path.open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with audit_path.open("w", encoding="utf-8") as stream:
        for row in audit_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_paths["profiles"].write_text(
        json.dumps(
            {relation: profile.as_dict() for relation, profile in sorted(profiles.items())},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "format_version": 1,
        "source": STRUCTURE_COUNTERFACTUAL_SOURCE,
        "score_path": str(score_path),
        "split": "train",
        "filter_splits": ["train"],
        "requested_k": total_k,
        "rows_written": len(manifest_rows),
        "rows_failed": len(failures),
        "parameters": {
            "uniform_mix": uniform_mix,
            "structure_weight": structure_weight,
            "violation_weight": violation_weight,
            "temperature": temperature,
            "seed": seed,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures[:100],
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
