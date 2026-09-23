"""Frozen full-vocabulary confusion DB for listed negative sampling.

The teacher is a frozen B1 adapter. Each relation QA is scored against the
official 198 train-graph relations with the same mean-logprob continuation
scorer used in training. Later negative sampling should read this table and
never rescore live.

Columns are official ``concept:`` names. Gold aliases are aligned with
``matched_train_relation``. Unmatched relation-like golds stay skipped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .official_lists import load_train_relation_order
from .task_file import (
    assistant_answer_prefix,
    assistant_gold,
    is_relation_gold,
    known_pair_relations,
    load_json_payload,
    match_train_relation,
    question_entity_pair,
    train_relation_alias_index,
)

SCORER_NAME = "normalized_continuation_mean_logprob"
SCORER_SPEC = {
    "score": "mean_logprob",
    "normalization": "answer_token_count",
    "includes_eos": False,
    "includes_answer_close_tag": False,
    "continuation": "relation_string_only",
    "prefix_end": "last_<answer>_inclusive",
    "temperature_in_scorer": False,
    "dtype": "bfloat16",
}
DEFAULT_SEED = 2026
DEFAULT_TEACHER = "frozen_b1"
DEFAULT_B1_ADAPTER = (
    "results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter"
)


def prefix_sha256(prefix_text: str) -> str:
    return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()


def scorer_metadata(
    *,
    vocab_size: int,
    seed: int = DEFAULT_SEED,
    b1_adapter: str | None = None,
    candidate_batch_size: int | None = None,
) -> dict:
    return {
        "scorer": SCORER_NAME,
        "scorer_spec": dict(SCORER_SPEC),
        "vocab_size": int(vocab_size),
        "vocab": "nell23k_train_graph_concept_relations",
        "seed": int(seed),
        "teacher": DEFAULT_TEACHER,
        "b1_adapter": b1_adapter,
        "candidate_batch_size": candidate_batch_size,
    }


def question_id_for_index(index: int) -> str:
    return f"task_qa:{index}"


def index_from_question_id(question_id: str) -> int:
    prefix, _, rest = str(question_id).partition(":")
    if prefix != "task_qa" or not rest.isdigit():
        raise ValueError(f"invalid question_id {question_id!r}")
    return int(rest)


def expected_relation_qa(
    qa_texts: list[str],
    alias_index: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Return matched / unmatched-relation / non-relation question IDs."""
    expected: list[str] = []
    unmatched: list[str] = []
    non_relation: list[str] = []
    for index, text in enumerate(qa_texts):
        question_id = question_id_for_index(index)
        gold = assistant_gold(text)
        if not is_relation_gold(gold, text):
            non_relation.append(question_id)
            continue
        if match_train_relation(gold, alias_index) is None:
            unmatched.append(question_id)
            continue
        expected.append(question_id)
    return expected, unmatched, non_relation


def score_map(all_candidate_scores: object) -> dict[str, float]:
    if not isinstance(all_candidate_scores, list) or not all_candidate_scores:
        raise ValueError("all_candidate_scores must be a non-empty list")
    scores: dict[str, float] = {}
    for item in all_candidate_scores:
        if not isinstance(item, dict):
            raise ValueError("candidate score item must be an object")
        relation = str(item.get("relation") or "")
        if not relation or relation in scores:
            raise ValueError("duplicate or empty candidate relation")
        scores[relation] = float(item["score"])
    return scores


def validate_score_table(
    all_candidate_scores: object,
    relation_order: list[str],
    gold: str,
) -> dict[str, float]:
    scores = score_map(all_candidate_scores)
    expected = list(relation_order)
    if len(expected) != len(set(expected)):
        raise ValueError("relation vocabulary contains duplicates")
    if set(scores) != set(expected) or len(scores) != len(expected):
        raise ValueError(
            f"score table must cover the {len(expected)} official relations exactly"
        )
    if gold not in scores:
        raise ValueError(f"gold relation {gold!r} is absent from the score table")
    ranked = sorted(scores, key=lambda rel: (-scores[rel], rel))
    items = list(all_candidate_scores)
    rank_by_relation = {str(item["relation"]): int(item["rank"]) for item in items}
    if [rank_by_relation[rel] for rel in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError("candidate ranks are not a dense 1..V ordering by score")
    return scores


def eligible_negative_relations(
    scores: dict[str, float],
    *,
    gold: str,
    known_relations: Iterable[str] = (),
) -> list[str]:
    """Exclude gold and known true pair relations; keep score-table order."""
    excluded = {gold, *(str(rel) for rel in known_relations)}
    return [rel for rel in scores if rel not in excluded]


def select_confusion_negatives(
    row: dict,
    relation_order: list[str],
    *,
    k: int,
    hard_k: int | None = None,
) -> list[str]:
    """Read one frozen row and return score-ranked negatives.

    Sampling is deterministic: gold and ``known_pair_relations`` are dropped,
    then remaining official relations are sorted by mean logprob. Call this
    once at dataset build time; do not rescore or resample each epoch.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    gold = str(row.get("matched_train_relation") or "")
    scores = validate_score_table(row.get("all_candidate_scores"), relation_order, gold)
    eligible = eligible_negative_relations(
        scores,
        gold=gold,
        known_relations=row.get("known_pair_relations") or (),
    )
    ranked = sorted(eligible, key=lambda rel: (-scores[rel], rel))
    take = k if hard_k is None else min(hard_k, k)
    return ranked[: min(take, len(ranked))]


def annotate_score_row(
    row: dict,
    *,
    prefix_text: str,
    relation_order: list[str],
    metadata: dict,
) -> dict:
    gold = str(row.get("matched_train_relation") or "")
    validate_score_table(row.get("all_candidate_scores"), relation_order, gold)
    annotated = dict(row)
    annotated.update(metadata)
    annotated["prefix_sha256"] = prefix_sha256(prefix_text)
    annotated["prefix_char_len"] = len(prefix_text)
    annotated["confusion_db"] = True
    return annotated


def load_score_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id!r}")
            seen.add(question_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def freeze_confusion_db(
    *,
    task_path: Path,
    raw_dir: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    metadata_path: Path,
    b1_adapter: str | Path | None = DEFAULT_B1_ADAPTER,
    candidate_batch_size: int = 8,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Validate a mined full-vocab score table and write the frozen confusion DB."""
    payload = load_json_payload(task_path)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{task_path} is not a GRIP task file")
    qa_texts = [str(text) for text in payload["qa_samples"]]
    relation_order = load_train_relation_order(raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_relations = known_pair_relations(raw_dir)
    expected, unmatched, non_relation = expected_relation_qa(qa_texts, alias_index)
    expected_set = set(expected)
    rows = load_score_rows(input_jsonl)
    found = [str(row["question_id"]) for row in rows]
    found_set = set(found)
    missing = [qid for qid in expected if qid not in found_set]
    extra = [qid for qid in found if qid not in expected_set]
    if missing or extra:
        raise ValueError(
            f"confusion DB coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )

    metadata = scorer_metadata(
        vocab_size=len(relation_order),
        seed=seed,
        b1_adapter=None if b1_adapter is None else str(b1_adapter),
        candidate_batch_size=candidate_batch_size,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    pair_filter_missing = 0
    known_nonempty = 0
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            question_id = str(row["question_id"])
            index = index_from_question_id(question_id)
            text = qa_texts[index]
            gold = assistant_gold(text)
            matched = match_train_relation(gold, alias_index)
            if matched is None:
                raise ValueError(f"{question_id}: gold does not match the train vocabulary")
            if str(row.get("positive_relation")) != gold:
                raise ValueError(f"{question_id}: positive_relation does not match task gold")
            if str(row.get("matched_train_relation")) != matched:
                raise ValueError(f"{question_id}: matched_train_relation mismatch")
            if int(row.get("seed", seed)) != seed:
                raise ValueError(f"{question_id}: seed must be {seed}")
            if str(row.get("teacher") or "") != DEFAULT_TEACHER:
                raise ValueError(f"{question_id}: teacher must be {DEFAULT_TEACHER}")
            prefix_text = assistant_answer_prefix(text)
            annotated = annotate_score_row(
                row,
                prefix_text=prefix_text,
                relation_order=relation_order,
                metadata=metadata,
            )
            pair = question_entity_pair(text)
            recorded_pair = row.get("entity_pair")
            expected_pair = list(pair) if pair else None
            if recorded_pair != expected_pair:
                raise ValueError(f"{question_id}: entity_pair does not match the question text")
            expected_known = sorted(known_relations.get(pair, set()) if pair else set())
            recorded_known = [str(rel) for rel in (row.get("known_pair_relations") or [])]
            if recorded_known != expected_known:
                raise ValueError(f"{question_id}: known_pair_relations mismatch")
            if pair is None:
                pair_filter_missing += 1
            if expected_known:
                known_nonempty += 1
            stream.write(json.dumps(annotated, ensure_ascii=False) + "\n")
    temp.replace(output_jsonl)

    digest = hashlib.sha256(output_jsonl.read_bytes()).hexdigest()
    summary = {
        "task_file": str(task_path),
        "source_manifest": str(input_jsonl),
        "confusion_db": str(output_jsonl),
        "rows": len(rows),
        "vocab_size": len(relation_order),
        "unmatched_relation_qa": len(unmatched),
        "non_relation_qa": len(non_relation),
        "entity_pair_missing": pair_filter_missing,
        "known_pair_nonempty": known_nonempty,
        "unmatched_question_ids": unmatched,
        "sha256": digest,
        **metadata,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
