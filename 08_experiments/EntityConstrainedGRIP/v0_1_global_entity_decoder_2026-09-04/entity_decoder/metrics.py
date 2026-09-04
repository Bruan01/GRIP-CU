"""Generation, ranking, and mechanism-probe metrics."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def canonicalize_prediction(prediction: str, entities: Iterable[str]) -> str:
    text = str(prediction).strip()
    normalized = normalize_text(text)
    exact = {normalize_text(entity): str(entity) for entity in entities}
    if normalized in exact:
        return exact[normalized]
    matches = []
    for normalized_entity, entity in exact.items():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_entity)}(?![A-Za-z0-9_])"
        if re.search(pattern, normalized):
            matches.append(entity)
    return matches[0] if len(set(matches)) == 1 else ""


def _strict_prefix_error(prediction: str, entities: Iterable[str]) -> bool:
    value = normalize_text(prediction)
    if not value:
        return False
    normalized_entities = [normalize_text(entity) for entity in entities]
    return value not in normalized_entities and any(entity.startswith(value) for entity in normalized_entities)


def _aggregate(rows: list[dict], entities: list[str]) -> dict:
    total = len(rows)
    if not total:
        return {
            "count": 0,
            "raw_exact_match": None,
            "canonical_entity_exact_match": None,
            "valid_entity_rate": None,
            "invalid_entity_rate": None,
            "invalid_or_hallucinated_entity_rate": None,
            "strict_prefix_error_rate": None,
        }
    raw_correct = canonical_correct = valid = prefix_errors = 0
    for row in rows:
        prediction = str(row.get("prediction_text", row.get("prediction", ""))).strip()
        answer = str(row["answer"])
        raw_correct += prediction.strip() == answer.strip()
        canonical = str(row.get("canonical_prediction") or canonicalize_prediction(prediction, entities))
        canonical_correct += normalize_text(canonical) == normalize_text(answer)
        valid += bool(canonical)
        prefix_errors += _strict_prefix_error(prediction, entities)
    return {
        "count": total,
        "raw_exact_match": raw_correct / total,
        "canonical_entity_exact_match": canonical_correct / total,
        "valid_entity_rate": valid / total,
        "invalid_entity_rate": 1.0 - valid / total,
        "invalid_or_hallucinated_entity_rate": 1.0 - valid / total,
        "strict_prefix_error_rate": prefix_errors / total,
    }


def score_prediction_rows(rows: Iterable[dict], entities: Iterable[str]) -> dict:
    rows, entities = list(rows), list(entities)
    result = _aggregate(rows, entities)
    dimensions = {
        "by_depth": lambda row: str(row["depth_label"]),
        "by_composition": lambda row: str(row.get("composition", "unknown")),
        "by_relation_frequency": lambda row: str(row.get("relation_frequency_bucket", "unknown")),
        "by_answer_character_length": lambda row: str(row.get("answer_character_length_bucket", "unknown")),
        "by_answer_token_length": lambda row: str(row.get("answer_token_length_bucket", "unknown")),
        "by_entity_prefix_ambiguity": lambda row: str(row.get("entity_prefix_ambiguity_bucket", "unknown")),
    }
    for name, getter in dimensions.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[getter(row)].append(row)
        result[name] = {key: _aggregate(group, entities) for key, group in sorted(grouped.items())}
    return result


def ranking_metrics(ranks: Iterable[int]) -> dict:
    ranks = [int(rank) for rank in ranks]
    if not ranks:
        return {"count": 0, "mrr": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0, "mean_rank": None}
    return {
        "count": len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "hit_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hit_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "hit_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }


def compare_decoder_predictions(
    d0_rows: Iterable[dict], d1_rows: Iterable[dict], entities: Iterable[str]
) -> dict:
    """Decompose semantic error transitions after aligning D0/D1 by task_id."""
    entities = list(entities)

    def index(rows: Iterable[dict], label: str) -> dict[str, dict]:
        indexed: dict[str, dict] = {}
        for row in rows:
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise ValueError(f"{label} row missing task_id")
            if task_id in indexed:
                raise ValueError(f"duplicate {label} task_id: {task_id}")
            indexed[task_id] = row
        return indexed

    d0 = index(d0_rows, "D0")
    d1 = index(d1_rows, "D1")
    if set(d0) != set(d1):
        raise ValueError("D0/D1 task_id sets differ")

    names = (
        "d0_invalid_to_d1_correct",
        "d0_invalid_to_d1_wrong",
        "d0_valid_wrong_to_d1_correct",
        "d0_valid_wrong_to_d1_wrong",
        "d0_correct_to_d1_correct",
        "d0_correct_to_d1_wrong",
        "d0_wrong_unchanged",
    )
    counts = {name: 0 for name in names}
    for task_id in sorted(d0):
        before, after = d0[task_id], d1[task_id]
        if normalize_text(before["answer"]) != normalize_text(after["answer"]):
            raise ValueError(f"D0/D1 answer mismatch for task_id: {task_id}")
        answer = normalize_text(before["answer"])
        before_prediction = str(before.get("prediction_text", before.get("prediction", "")))
        after_prediction = str(after.get("prediction_text", after.get("prediction", "")))
        before_canonical = str(before.get("canonical_prediction") or canonicalize_prediction(before_prediction, entities))
        after_canonical = str(after.get("canonical_prediction") or canonicalize_prediction(after_prediction, entities))
        before_correct = normalize_text(before_canonical) == answer
        after_correct = normalize_text(after_canonical) == answer
        before_valid = bool(before_canonical)

        if before_correct:
            counts["d0_correct_to_d1_correct" if after_correct else "d0_correct_to_d1_wrong"] += 1
        elif before_valid:
            counts["d0_valid_wrong_to_d1_correct" if after_correct else "d0_valid_wrong_to_d1_wrong"] += 1
        else:
            counts["d0_invalid_to_d1_correct" if after_correct else "d0_invalid_to_d1_wrong"] += 1
        if not before_correct and not after_correct:
            counts["d0_wrong_unchanged"] += 1

    total = len(d0)
    rates = {name: (count / total if total else None) for name, count in counts.items()}
    return {"count": total, "counts": counts, "rates": rates}
