"""Offline full-vocabulary relation scoring and confusion metrics.

This module does not sample hard negatives and does not train. It turns a
per-question map of ``score(A | Q)`` into the JSONL rows used by the small-scale
confusion dump. Scoring itself stays in ``score_candidates``.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

from .task_file import assistant_gold, is_relation_gold, match_train_relation, question_entity_pair

QUESTION_RE = re.compile(
    r"please answer the following question:\s*(.*?)\s*Response in the following format",
    re.IGNORECASE | re.DOTALL,
)
USER_SPAN_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
GOLD_INVALID_REASON = "gold_relation"
ALTERNATIVE_TRUE_REASON = "alternative_true_relation"
MASS_ATOL = 1e-6
GAP_ATOL = 1e-8


def extract_question_text(text: str) -> str | None:
    """Keep the visible question string when the chat template can be parsed."""
    match = QUESTION_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    match = USER_SPAN_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    stripped = text.strip()
    return stripped or None


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def pairwise_confusion(candidate_score: float, gold_score: float, temperature: float) -> float:
    """P(candidate beats gold) in a two-way softmax, numerically stable.

    Equivalent to ``exp(s_c / T) / (exp(s_g / T) + exp(s_c / T))``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return stable_sigmoid((candidate_score - gold_score) / temperature)


def stable_softmax(scores: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return []
    scaled = [(score / temperature) for score in scores]
    maximum = max(scaled)
    exps = [math.exp(score - maximum) for score in scaled]
    total = sum(exps)
    if total <= 0:
        raise ValueError("softmax denominator must be positive")
    return [value / total for value in exps]


def dense_ranks(items: list[tuple[str, float]]) -> dict[str, int]:
    """Rank by score descending; relation name breaks ties so ranks are unique."""
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    return {name: index + 1 for index, (name, _score) in enumerate(ordered)}


def parse_matchable_relation_qa(
    index: int,
    text: str,
    alias_index: dict[str, str],
) -> dict | None:
    """Return one Stage-2 relation QA item if its gold maps onto the train vocabulary."""
    gold = assistant_gold(text)
    if not is_relation_gold(gold, text):
        return None
    matched = match_train_relation(gold, alias_index)
    if matched is None:
        return None
    pair = question_entity_pair(text)
    return {
        "qa_id": f"task_qa:{index}",
        "text": text,
        "question": extract_question_text(text),
        "gold_relation": gold,
        "matched_train_relation": matched,
        "head_entity": None if pair is None else pair[0],
        "tail_entity": None if pair is None else pair[1],
        "entity_pair": pair,
    }


def iter_matchable_relation_qa(
    qa_texts: list[str],
    alias_index: dict[str, str],
) -> Iterable[dict]:
    """Yield Stage-2 relation QA items whose gold maps onto the train vocabulary."""
    for index, text in enumerate(qa_texts):
        item = parse_matchable_relation_qa(index, str(text), alias_index)
        if item is not None:
            yield item


def classify_candidate(
    *,
    candidate: str,
    gold: str,
    known_relations: Iterable[str] = (),
) -> tuple[bool, bool, str | None]:
    """Return ``(is_gold, is_valid_negative, invalid_reason)``."""
    if candidate == gold:
        return True, False, GOLD_INVALID_REASON
    if candidate in {str(rel) for rel in known_relations}:
        return False, False, ALTERNATIVE_TRUE_REASON
    return False, True, None


def build_qa_score_rows(
    *,
    qa_id: str,
    question: str | None,
    head_entity: str | None,
    tail_entity: str | None,
    gold_relation: str,
    matched_train_relation: str,
    relation_order: list[str],
    scores: dict[str, float],
    token_lengths: dict[str, int],
    known_relations: Iterable[str] = (),
    temperature: float = 1.0,
    model_checkpoint: str,
    split: str = "train",
) -> tuple[list[dict], dict]:
    """Expand one QA's vocab scores into candidate rows plus a QA summary."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if set(scores) != set(relation_order) or len(scores) != len(relation_order):
        raise ValueError("scores must cover the official relation vocabulary exactly")
    if set(token_lengths) != set(relation_order) or len(token_lengths) != len(relation_order):
        raise ValueError("token_lengths must cover the official relation vocabulary exactly")
    if matched_train_relation not in scores:
        raise ValueError(f"gold relation {matched_train_relation!r} is absent from scores")

    gold_score = float(scores[matched_train_relation])
    known = {str(rel) for rel in known_relations}
    labels = {
        rel: classify_candidate(candidate=rel, gold=matched_train_relation, known_relations=known)
        for rel in relation_order
    }
    all_ranks = dense_ranks([(rel, float(scores[rel])) for rel in relation_order])
    valid_negatives = [rel for rel, (_is_gold, is_valid, _reason) in labels.items() if is_valid]
    negative_ranks = dense_ranks([(rel, float(scores[rel])) for rel in valid_negatives])
    masses = stable_softmax([float(scores[rel]) for rel in valid_negatives], temperature)
    mass_by_relation = dict(zip(valid_negatives, masses))

    rows: list[dict] = []
    for relation in relation_order:
        is_gold, is_valid, invalid_reason = labels[relation]
        candidate_score = float(scores[relation])
        row = {
            "qa_id": qa_id,
            "question": question,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "gold_relation": gold_relation,
            "matched_train_relation": matched_train_relation,
            "candidate_relation": relation,
            "is_gold": is_gold,
            "is_valid_negative": is_valid,
            "invalid_reason": invalid_reason,
            "candidate_score": candidate_score,
            "gold_score": gold_score,
            "score_gap": gold_score - candidate_score,
            "pairwise_confusion": (
                None if is_gold else pairwise_confusion(candidate_score, gold_score, temperature)
            ),
            "negative_mass": mass_by_relation.get(relation),
            "candidate_rank_all": all_ranks[relation],
            "negative_rank": negative_ranks.get(relation),
            "candidate_token_length": int(token_lengths[relation]),
            "model_checkpoint": model_checkpoint,
            "split": split,
        }
        rows.append(row)

    hardest = None
    if valid_negatives:
        hardest = max(valid_negatives, key=lambda rel: (float(scores[rel]), rel))
    above_gold = [
        rel for rel in valid_negatives if float(scores[rel]) > gold_score
    ]
    summary = {
        "qa_id": qa_id,
        "gold_relation": gold_relation,
        "matched_train_relation": matched_train_relation,
        "gold_score": gold_score,
        "gold_rank": all_ranks[matched_train_relation],
        "number_of_candidates": len(relation_order),
        "number_of_valid_negatives": len(valid_negatives),
        "hardest_negative": hardest,
        "hardest_negative_score": None if hardest is None else float(scores[hardest]),
        "hardest_gap": None if hardest is None else gold_score - float(scores[hardest]),
        "number_of_negatives_above_gold": len(above_gold),
        "head_entity": head_entity,
        "tail_entity": tail_entity,
        "question": question,
        "split": split,
    }
    return rows, summary


def relabel_qa_score_group(
    group: list[dict],
    *,
    relation_order: list[str],
    known_relations: Iterable[str],
    temperature: float = 1.0,
) -> tuple[list[dict], dict]:
    """Rebuild only labels and negative statistics from an existing score group.

    Candidate scores and token lengths are read from the source rows, so this
    operation never invokes a model or changes the scorer output.
    """
    if not group:
        raise ValueError("cannot relabel an empty QA group")
    first = group[0]
    candidates = [str(row.get("candidate_relation") or "") for row in group]
    if candidates != relation_order:
        raise ValueError("source group candidate order does not match relation vocabulary")
    scores = {str(row["candidate_relation"]): float(row["candidate_score"]) for row in group}
    token_lengths = {
        str(row["candidate_relation"]): int(row["candidate_token_length"])
        for row in group
    }
    gold_rows = [row for row in group if row.get("is_gold")]
    if len(gold_rows) != 1:
        raise ValueError(f"{first.get('qa_id')}: expected exactly one gold row")
    gold_row = gold_rows[0]
    rebuilt_rows, summary = build_qa_score_rows(
        qa_id=str(first["qa_id"]),
        question=first.get("question"),
        head_entity=first.get("head_entity"),
        tail_entity=first.get("tail_entity"),
        gold_relation=str(first["gold_relation"]),
        matched_train_relation=str(first["matched_train_relation"]),
        relation_order=relation_order,
        scores=scores,
        token_lengths=token_lengths,
        known_relations=known_relations,
        temperature=temperature,
        model_checkpoint=str(first.get("model_checkpoint") or ""),
        split=str(first.get("split") or "train"),
    )
    if str(gold_row["candidate_relation"]) != str(first["matched_train_relation"]):
        raise ValueError(f"{first.get('qa_id')}: source gold does not match matched relation")
    return rebuilt_rows, summary
def validate_candidate_rows(
    rows: list[dict],
    *,
    relation_order: list[str],
    known_relations: dict[tuple[str, str], set[str]] | None = None,
    temperature: float = 1.0,
) -> list[str]:
    """Return human-readable check failures. An empty list means the dump is valid."""
    del temperature  # pairwise_confusion already stored; recompute from scores.
    failures: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["qa_id"]), []).append(row)

    vocab = list(relation_order)
    vocab_set = set(vocab)
    for qa_id, group in grouped.items():
        golds = [row for row in group if row.get("is_gold")]
        if len(golds) != 1:
            failures.append(f"{qa_id}: expected exactly one gold, found {len(golds)}")
            continue
        gold_row = golds[0]
        gold_score = float(gold_row["gold_score"])
        matched = str(gold_row["matched_train_relation"])
        candidates = [str(row["candidate_relation"]) for row in group]
        if set(candidates) != vocab_set or len(candidates) != len(vocab):
            failures.append(f"{qa_id}: candidate set does not match the official vocabulary")
        if any(abs(float(row["gold_score"]) - gold_score) > GAP_ATOL for row in group):
            failures.append(f"{qa_id}: gold_score is not constant across candidate rows")
        if any(row.get("is_valid_negative") and row.get("is_gold") for row in group):
            failures.append(f"{qa_id}: gold is marked as a valid negative")

        pair = None
        head, tail = gold_row.get("head_entity"), gold_row.get("tail_entity")
        if head and tail:
            pair = (str(head), str(tail))
        expected_known = set()
        if known_relations is not None and pair is not None:
            expected_known = {str(rel) for rel in known_relations.get(pair, set())}
        for row in group:
            relation = str(row["candidate_relation"])
            is_gold, is_valid, reason = classify_candidate(
                candidate=relation,
                gold=matched,
                known_relations=expected_known,
            )
            if bool(row.get("is_gold")) != is_gold:
                failures.append(f"{qa_id}: is_gold mismatch for {relation}")
            if known_relations is not None and bool(row.get("is_valid_negative")) != is_valid:
                failures.append(f"{qa_id}: is_valid_negative mismatch for {relation}")
            if known_relations is not None and row.get("invalid_reason") != reason:
                failures.append(f"{qa_id}: invalid_reason mismatch for {relation}")
            expected_gap = gold_score - float(row["candidate_score"])
            if abs(float(row["score_gap"]) - expected_gap) > GAP_ATOL:
                failures.append(f"{qa_id}: score_gap mismatch for {relation}")
            confusion = row.get("pairwise_confusion")
            if is_gold:
                if confusion is not None:
                    failures.append(f"{qa_id}: gold pairwise_confusion must be null")
            else:
                if confusion is None or not (0.0 <= float(confusion) <= 1.0):
                    failures.append(f"{qa_id}: pairwise_confusion out of range for {relation}")

        valid = [row for row in group if row.get("is_valid_negative")]
        masses = [row.get("negative_mass") for row in valid]
        if any(mass is None for mass in masses):
            failures.append(f"{qa_id}: valid negative is missing negative_mass")
        elif valid:
            total = sum(float(mass) for mass in masses)
            if abs(total - 1.0) > MASS_ATOL:
                failures.append(f"{qa_id}: negative_mass sums to {total}, expected 1")
        if any(row.get("negative_mass") is not None and not row.get("is_valid_negative") for row in group):
            failures.append(f"{qa_id}: non-valid candidate has negative_mass")

        all_ranks = [int(row["candidate_rank_all"]) for row in group]
        if sorted(all_ranks) != list(range(1, len(group) + 1)):
            failures.append(f"{qa_id}: candidate_rank_all is not a dense unique ranking")
        neg_ranks = [int(row["negative_rank"]) for row in valid if row.get("negative_rank") is not None]
        if sorted(neg_ranks) != list(range(1, len(valid) + 1)):
            failures.append(f"{qa_id}: negative_rank is not a dense unique ranking")
        if gold_row.get("negative_rank") is not None:
            failures.append(f"{qa_id}: gold negative_rank must be null")
    return failures


def summarize_run(
    qa_summaries: list[dict],
    *,
    elapsed_sec: float,
    peak_gpu_bytes: int | None,
    vocab_size: int,
    skipped: int,
) -> dict:
    n_qa = len(qa_summaries)
    n_candidates = n_qa * vocab_size
    elapsed = max(float(elapsed_sec), 0.0)
    gold_ranks = [int(row["gold_rank"]) for row in qa_summaries]
    inverted = sum(1 for row in qa_summaries if int(row["number_of_negatives_above_gold"]) > 0)
    return {
        "qa_count": n_qa,
        "skipped_non_matchable": skipped,
        "total_candidates": n_candidates,
        "vocab_size": vocab_size,
        "mean_candidates_per_qa": float(vocab_size) if n_qa else 0.0,
        "elapsed_sec": elapsed,
        "qa_per_sec": (n_qa / elapsed) if elapsed else 0.0,
        "candidate_per_sec": (n_candidates / elapsed) if elapsed else 0.0,
        "peak_gpu_bytes": peak_gpu_bytes,
        "peak_gpu_gib": None if peak_gpu_bytes is None else peak_gpu_bytes / (1024 ** 3),
        "mean_gold_rank": (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None,
        "qa_with_negative_above_gold": inverted,
    }
