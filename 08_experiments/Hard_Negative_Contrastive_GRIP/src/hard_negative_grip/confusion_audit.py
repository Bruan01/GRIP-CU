"""Quality audit for a frozen relation-confusion dump.

Streams ``candidate_scores.jsonl`` once. The CPU path does not load a 7B
teacher and does not train. Optional 20-QA live rescore is a separate GPU
script that only scores, never trains.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .confusion_analysis import iter_jsonl, iter_qa_groups_from_scores
from .confusion_db import expected_relation_qa
from .official_lists import load_train_relation_order
from .offline_scoring import (
    ALTERNATIVE_TRUE_REASON,
    GAP_ATOL,
    MASS_ATOL,
    classify_candidate,
    dense_ranks,
    pairwise_confusion,
    stable_softmax,
)
from .task_file import (
    known_pair_relations,
    load_json_payload,
    train_relation_alias_index,
)

DEFAULT_SEED = 2026
MATH_SAMPLE = 100
RANK_SAMPLE = 100
ALT_TRUE_SAMPLE = 100
VALID_NEG_SAMPLE = 500
SCORER_QA_SAMPLE = 20
MASS_ATOL_REPORT = 1e-6
SCORE_COMPARE_ATOL = 1e-5
FAILURE_CAP = 30
B1_ADAPTER_MARKER = "/b1/adapter"
LISTED_ADAPTER_MARKER = "/listed/adapter"


def is_finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def reservoir_add(pool: list, item: object, seen: int, rng: random.Random, k: int) -> None:
    if k <= 0:
        return
    if len(pool) < k:
        pool.append(item)
        return
    index = rng.randrange(seen)
    if index < k:
        pool[index] = item


def _append_capped(bucket: list[str], item: str, cap: int = FAILURE_CAP) -> None:
    if len(bucket) < cap:
        bucket.append(item)


def load_split_pair_relations(raw_dir: Path) -> dict[str, dict[tuple[str, str], set[str]]]:
    """Index triples per split without merging them."""
    from .official_lists import load_triples

    splits = {
        "train": load_triples(raw_dir / "train.txt"),
        "valid": load_triples(raw_dir / "valid.txt"),
        "test": load_triples(raw_dir / "test.txt"),
    }
    indexed: dict[str, dict[tuple[str, str], set[str]]] = {}
    for name, triples in splits.items():
        mapping: dict[tuple[str, str], set[str]] = {}
        for head, relation, tail in triples:
            mapping.setdefault((head, tail), set()).add(relation)
        indexed[name] = mapping
    return indexed


def expected_gap(row: dict) -> float:
    return float(row["gold_score"]) - float(row["candidate_score"])


def expected_pairwise(row: dict, temperature: float) -> float | None:
    if row.get("is_gold"):
        return None
    return pairwise_confusion(
        float(row["candidate_score"]),
        float(row["gold_score"]),
        temperature,
    )


def recompute_ranks(group: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    all_items = [
        (str(row["candidate_relation"]), float(row["candidate_score"])) for row in group
    ]
    valid_items = [
        (str(row["candidate_relation"]), float(row["candidate_score"]))
        for row in group
        if row.get("is_valid_negative")
    ]
    return dense_ranks(all_items), dense_ranks(valid_items)


def check_group_completeness(
    group: list[dict],
    *,
    relation_order: list[str],
    known_relations: dict[tuple[str, str], set[str]],
) -> list[str]:
    failures: list[str] = []
    if not group:
        return ["empty QA group"]
    qa_id = str(group[0].get("qa_id") or "")
    vocab = list(relation_order)
    vocab_set = set(vocab)
    candidates = [str(row.get("candidate_relation") or "") for row in group]
    if any(not name.strip() for name in candidates):
        failures.append(f"{qa_id}: empty candidate_relation")
    if len(candidates) != len(set(candidates)):
        failures.append(f"{qa_id}: duplicate candidate_relation")
    if set(candidates) != vocab_set or len(candidates) != len(vocab):
        failures.append(
            f"{qa_id}: candidate count {len(candidates)} != vocab {len(vocab)}"
        )
    elif candidates != vocab:
        failures.append(f"{qa_id}: candidate order does not match relation vocab")
    golds = [row for row in group if row.get("is_gold")]
    if len(golds) != 1:
        failures.append(f"{qa_id}: expected exactly one gold, found {len(golds)}")
        return failures
    gold_row = golds[0]
    matched = str(gold_row.get("matched_train_relation") or "")
    if not matched:
        failures.append(f"{qa_id}: empty matched_train_relation")
    if str(gold_row.get("candidate_relation")) != matched:
        failures.append(f"{qa_id}: gold candidate is not matched_train_relation")
    pair = None
    head, tail = gold_row.get("head_entity"), gold_row.get("tail_entity")
    if head and tail:
        pair = (str(head), str(tail))
    expected_known = set()
    if pair is not None:
        expected_known = {str(rel) for rel in known_relations.get(pair, set())}
    for row in group:
        relation = str(row.get("candidate_relation") or "")
        if not is_finite_number(row.get("candidate_score")):
            failures.append(f"{qa_id}: non-finite candidate_score for {relation}")
        if not is_finite_number(row.get("gold_score")):
            failures.append(f"{qa_id}: non-finite gold_score for {relation}")
        if not is_finite_number(row.get("score_gap")):
            failures.append(f"{qa_id}: non-finite score_gap for {relation}")
        token_length = row.get("candidate_token_length")
        if not isinstance(token_length, int) or token_length <= 0:
            failures.append(f"{qa_id}: token_length <= 0 for {relation}")
        is_gold, is_valid, reason = classify_candidate(
            candidate=relation,
            gold=matched,
            known_relations=expected_known,
        )
        if bool(row.get("is_gold")) != is_gold:
            failures.append(f"{qa_id}: is_gold mismatch for {relation}")
        if bool(row.get("is_valid_negative")) != is_valid:
            failures.append(f"{qa_id}: is_valid_negative mismatch for {relation}")
        if row.get("invalid_reason") != reason:
            failures.append(f"{qa_id}: invalid_reason mismatch for {relation}")
        confusion = row.get("pairwise_confusion")
        if is_gold:
            if confusion is not None:
                failures.append(f"{qa_id}: gold pairwise_confusion must be null")
        elif not is_finite_number(confusion) or not (0.0 <= float(confusion) <= 1.0):
            failures.append(f"{qa_id}: pairwise_confusion out of range for {relation}")
        mass = row.get("negative_mass")
        if is_valid:
            if not is_finite_number(mass):
                failures.append(f"{qa_id}: valid negative missing negative_mass")
        elif mass is not None:
            failures.append(f"{qa_id}: non-valid candidate has negative_mass")
    return failures


def check_group_mass(group: list[dict]) -> list[str]:
    qa_id = str(group[0].get("qa_id") or "")
    valid = [row for row in group if row.get("is_valid_negative")]
    if not valid:
        return [f"{qa_id}: no valid negatives"]
    total = sum(float(row["negative_mass"]) for row in valid)
    if abs(total - 1.0) > MASS_ATOL_REPORT:
        return [f"{qa_id}: negative_mass sums to {total}, expected 1"]
    recomputed = stable_softmax(
        [float(row["candidate_score"]) for row in valid],
        1.0,
    )
    stored = [float(row["negative_mass"]) for row in valid]
    if any(abs(left - right) > MASS_ATOL for left, right in zip(stored, recomputed)):
        return [f"{qa_id}: stored negative_mass does not match softmax(candidate_score)"]
    return []


def check_group_ranks(group: list[dict]) -> list[str]:
    qa_id = str(group[0].get("qa_id") or "")
    all_ranks, neg_ranks = recompute_ranks(group)
    failures: list[str] = []
    for row in group:
        relation = str(row["candidate_relation"])
        if int(row["candidate_rank_all"]) != all_ranks[relation]:
            failures.append(f"{qa_id}: candidate_rank_all mismatch for {relation}")
        stored_neg = row.get("negative_rank")
        expected_neg = neg_ranks.get(relation)
        if stored_neg != expected_neg:
            failures.append(f"{qa_id}: negative_rank mismatch for {relation}")
    return failures


def math_row_failures(row: dict, temperature: float) -> list[str]:
    failures: list[str] = []
    qa_id = str(row.get("qa_id") or "")
    relation = str(row.get("candidate_relation") or "")
    gap = expected_gap(row)
    if abs(float(row["score_gap"]) - gap) > GAP_ATOL:
        failures.append(f"{qa_id}/{relation}: score_gap stored={row['score_gap']} expected={gap}")
    expected = expected_pairwise(row, temperature)
    stored = row.get("pairwise_confusion")
    if expected is None:
        if stored is not None:
            failures.append(f"{qa_id}/{relation}: gold pairwise_confusion should be null")
    elif stored is None or abs(float(stored) - expected) > 1e-12:
        failures.append(
            f"{qa_id}/{relation}: pairwise_confusion stored={stored} expected={expected}"
        )
    return failures


def compare_dump_to_frozen_map(
    dump_group: list[dict],
    frozen: dict,
    *,
    relation_order: list[str],
) -> dict:
    dump_scores = {
        str(row["candidate_relation"]): float(row["candidate_score"]) for row in dump_group
    }
    dump_ranks = {
        str(row["candidate_relation"]): int(row["candidate_rank_all"]) for row in dump_group
    }
    frozen_scores = frozen.get("scores") or {}
    frozen_ranks = frozen.get("ranks") or {}
    abs_errors = [
        abs(dump_scores[rel] - frozen_scores[rel])
        for rel in relation_order
        if rel in dump_scores and rel in frozen_scores
    ]
    rank_mismatches = sum(
        1 for rel in relation_order if dump_ranks.get(rel) != frozen_ranks.get(rel)
    )
    return {
        "qa_id": str(dump_group[0]["qa_id"]),
        "n_compared": len(abs_errors),
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "n_score_mismatch_1e5": sum(1 for value in abs_errors if value > SCORE_COMPARE_ATOL),
        "n_rank_mismatch": rank_mismatches,
    }


def compare_score_groups(left: list[dict], right: list[dict]) -> dict:
    left_map = {str(row["candidate_relation"]): row for row in left}
    right_map = {str(row["candidate_relation"]): row for row in right}
    keys = sorted(set(left_map) & set(right_map))
    abs_errors = [
        abs(float(left_map[key]["candidate_score"]) - float(right_map[key]["candidate_score"]))
        for key in keys
    ]
    rank_mismatches = sum(
        1
        for key in keys
        if int(left_map[key]["candidate_rank_all"]) != int(right_map[key]["candidate_rank_all"])
    )
    order_mismatch = [str(row["candidate_relation"]) for row in left] != [
        str(row["candidate_relation"]) for row in right
    ]
    return {
        "n_compared": len(abs_errors),
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "n_score_mismatch_1e5": sum(1 for value in abs_errors if value > SCORE_COMPARE_ATOL),
        "n_rank_mismatch": rank_mismatches,
        "output_order_mismatch": order_mismatch,
    }


def load_frozen_score_maps(path: Path, wanted: set[str] | None = None) -> dict[str, dict]:
    """Stream compact score/rank maps. Does not expand frozen rows to dump schema."""
    found: dict[str, dict] = {}
    if not path.is_file():
        return found
    for row in iter_jsonl(path):
        qa_id = str(row.get("question_id") or row.get("qa_id") or "")
        if wanted is not None and qa_id not in wanted:
            continue
        scores: dict[str, float] = {}
        ranks: dict[str, int] = {}
        for item in row.get("all_candidate_scores") or []:
            relation = str(item.get("relation") or "")
            if not relation:
                continue
            scores[relation] = float(item["score"])
            ranks[relation] = int(item["rank"])
        found[qa_id] = {"scores": scores, "ranks": ranks}
        if wanted is not None and len(found) == len(wanted):
            break
    return found


def load_live_score_groups(path: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    if not path.is_file():
        return found
    for group in iter_qa_groups_from_scores(path):
        found[str(group[0].get("qa_id") or "")] = group
    return found


def inspect_checkpoint(path: str | None) -> dict:
    text = str(path or "")
    adapter = Path(text)
    metadata_path = adapter / "run_metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "path": text,
        "exists": adapter.exists(),
        "is_b1": B1_ADAPTER_MARKER in text.replace("\\", "/"),
        "is_listed": LISTED_ADAPTER_MARKER in text.replace("\\", "/"),
        "run_metadata": metadata,
    }


def gpu_occupancy() -> dict:
    import subprocess

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        nvml = {"available": False, "error": "nvidia-smi not found"}
    else:
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            nvml = {
                "available": False,
                "error": message or f"nvidia-smi exit {completed.returncode}",
            }
        else:
            rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            nvml = {"available": True, "gpus": rows}
    torch_cuda = False
    torch_error = None
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - import/runtime probe
        torch_error = str(exc)
    live_possible = bool(nvml.get("available") or torch_cuda)
    return {
        **nvml,
        "torch_cuda_available": torch_cuda,
        "torch_error": torch_error,
        "live_rescore_possible": live_possible,
    }


def load_live_rescore(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a live-rescore JSON object")
    return payload


def comparison_within_atol(payload: dict | None, *, atol: float = SCORE_COMPARE_ATOL) -> bool:
    if not payload or payload.get("max_abs_error") is None:
        return False
    return (
        float(payload["max_abs_error"]) <= atol
        and int(payload.get("n_rank_mismatch_total") or 0) == 0
        and not payload.get("output_order_mismatch")
    )


def live_rescore_is_ok(payload: dict | None) -> bool:
    if not payload:
        return False
    if int(payload.get("n_qa") or 0) < SCORER_QA_SAMPLE:
        return False
    return all(
        comparison_within_atol(payload.get(key))
        for key in (
            "training_vs_offline",
            "training_run1_vs_run2",
            "offline_run1_vs_run2",
            "dump_vs_training",
            "dump_vs_offline",
        )
    )


def audit_dump(
    *,
    scores_path: Path,
    summary_path: Path,
    metadata_path: Path,
    task_file: Path,
    raw_dir: Path,
    frozen_db_path: Path | None = None,
    compare_scores_path: Path | None = None,
    live_rescore_path: Path | None = None,
    listed_adapter: str | None = None,
    seed: int = DEFAULT_SEED,
    temperature: float = 1.0,
    math_sample: int = MATH_SAMPLE,
    rank_sample: int = RANK_SAMPLE,
    alt_true_sample: int = ALT_TRUE_SAMPLE,
    valid_neg_sample: int = VALID_NEG_SAMPLE,
    scorer_qa_sample: int = SCORER_QA_SAMPLE,
    filter_splits: tuple[str, ...] | None = None,
) -> dict:
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if filter_splits is None:
        filter_splits = tuple(metadata.get("filter_splits") or ("train", "valid", "test"))
    filter_splits = tuple(filter_splits)
    relation_order = load_train_relation_order(raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_all = known_pair_relations(raw_dir, splits=filter_splits)
    split_known = load_split_pair_relations(raw_dir)
    payload = load_json_payload(task_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{task_file} is not a GRIP task file")
    qa_texts = [str(text) for text in payload["qa_samples"]]
    expected_ids, unmatched_ids, non_relation_ids = expected_relation_qa(qa_texts, alias_index)
    expected_set = set(expected_ids)

    rng_math = random.Random(seed)
    rng_rank = random.Random(seed + 1)
    rng_alt = random.Random(seed + 2)
    rng_valid = random.Random(seed + 3)
    rng_scorer = random.Random(seed + 4)

    math_rows: list[dict] = []
    rank_groups: list[list[dict]] = []
    alt_true_rows: list[dict] = []
    valid_neg_rows: list[dict] = []
    scorer_ids: list[str] = []
    n_math_seen = n_rank_seen = n_alt_seen = n_valid_seen = n_qa_seen = 0

    completeness_failures: list[str] = []
    mass_failures: list[str] = []
    rank_failures: list[str] = []
    math_failures: list[str] = []
    n_completeness_failures = 0
    n_mass_failures = 0
    n_rank_failures = 0
    n_math_failures = 0
    n_rows = 0
    n_qa = 0
    n_gold = 0
    n_valid = 0
    n_alt_true = 0
    n_nan = 0
    n_inf = 0
    n_empty_rel = 0
    n_bad_token = 0
    n_duplicate_qa = 0
    n_pair_missing = 0
    n_pair_in_train = 0
    n_pair_in_valid = 0
    n_pair_in_test = 0
    n_alt_from_test = 0
    n_alt_from_valid = 0
    n_alt_from_train = 0
    seen_qa: set[str] = set()
    invalid_reasons: Counter[str] = Counter()
    checkpoints: set[str] = set()
    splits: set[str] = set()
    gold_relations: set[str] = set()
    matched_relations: set[str] = set()
    frozen_comparisons: list[dict] = []
    live_comparisons: list[dict] = []

    frozen_maps: dict[str, dict] = {}
    if frozen_db_path is not None and frozen_db_path.is_file():
        frozen_maps = load_frozen_score_maps(frozen_db_path)
    live_groups: dict[str, list[dict]] = {}
    if compare_scores_path is not None and compare_scores_path.is_file():
        live_groups = load_live_score_groups(compare_scores_path)

    for group in iter_qa_groups_from_scores(scores_path):
        n_qa += 1
        qa_id = str(group[0].get("qa_id") or "")
        if qa_id in seen_qa:
            n_duplicate_qa += 1
            _append_capped(completeness_failures, f"{qa_id}: duplicate QA block")
        seen_qa.add(qa_id)
        n_rows += len(group)
        for item in check_group_completeness(
            group,
            relation_order=relation_order,
            known_relations=known_all,
        ):
            n_completeness_failures += 1
            _append_capped(completeness_failures, item)
        for item in check_group_mass(group):
            n_mass_failures += 1
            _append_capped(mass_failures, item)
        group_rank_failures = check_group_ranks(group)
        n_rank_failures += len(group_rank_failures)
        for item in group_rank_failures:
            _append_capped(rank_failures, item)
        n_rank_seen += 1
        reservoir_add(rank_groups, group, n_rank_seen, rng_rank, rank_sample)

        n_qa_seen += 1
        reservoir_add(scorer_ids, qa_id, n_qa_seen, rng_scorer, scorer_qa_sample)

        if qa_id in frozen_maps:
            frozen_comparisons.append(
                compare_dump_to_frozen_map(group, frozen_maps[qa_id], relation_order=relation_order)
            )
        if qa_id in live_groups:
            live_comparisons.append(
                {"qa_id": qa_id, **compare_score_groups(group, live_groups[qa_id])}
            )

        gold_row = next((row for row in group if row.get("is_gold")), None)
        if gold_row is not None:
            n_gold += 1
            gold_relations.add(str(gold_row.get("gold_relation") or ""))
            matched_relations.add(str(gold_row.get("matched_train_relation") or ""))
        head, tail = group[0].get("head_entity"), group[0].get("tail_entity")
        pair = (str(head), str(tail)) if head and tail else None
        if pair is None:
            n_pair_missing += 1
        else:
            if pair in split_known["train"]:
                n_pair_in_train += 1
            if pair in split_known["valid"]:
                n_pair_in_valid += 1
            if pair in split_known["test"]:
                n_pair_in_test += 1
        for row in group:
            checkpoints.add(str(row.get("model_checkpoint") or ""))
            splits.add(str(row.get("split") or ""))
            relation = str(row.get("candidate_relation") or "")
            if not relation.strip():
                n_empty_rel += 1
            token_length = row.get("candidate_token_length")
            if not isinstance(token_length, int) or token_length <= 0:
                n_bad_token += 1
            for field in ("candidate_score", "gold_score", "score_gap"):
                value = row.get(field)
                if isinstance(value, float) and math.isnan(value):
                    n_nan += 1
                elif isinstance(value, float) and math.isinf(value):
                    n_inf += 1
            n_math_seen += 1
            reservoir_add(math_rows, row, n_math_seen, rng_math, math_sample)
            row_math = math_row_failures(row, temperature)
            n_math_failures += len(row_math)
            for item in row_math:
                _append_capped(math_failures, item)
            if row.get("is_valid_negative"):
                n_valid += 1
                n_valid_seen += 1
                reservoir_add(valid_neg_rows, row, n_valid_seen, rng_valid, valid_neg_sample)
            reason = row.get("invalid_reason")
            if reason:
                invalid_reasons[str(reason)] += 1
            if (not row.get("is_gold")) and row.get("invalid_reason") == ALTERNATIVE_TRUE_REASON:
                n_alt_true += 1
                n_alt_seen += 1
                reservoir_add(alt_true_rows, row, n_alt_seen, rng_alt, alt_true_sample)
                if pair is not None:
                    rel = str(row["candidate_relation"])
                    if rel in split_known["test"].get(pair, set()):
                        n_alt_from_test += 1
                    if rel in split_known["valid"].get(pair, set()):
                        n_alt_from_valid += 1
                    if rel in split_known["train"].get(pair, set()):
                        n_alt_from_train += 1

    missing_expected = sorted(expected_set - seen_qa)
    extra_expected = sorted(seen_qa - expected_set)
    summary_ids = set()
    if summary_path.is_file():
        for row in iter_jsonl(summary_path):
            summary_ids.add(str(row.get("qa_id") or ""))
    summary_missing = sorted(seen_qa - summary_ids)
    summary_extra = sorted(summary_ids - seen_qa)

    sampled_math_failures: list[str] = []
    for row in math_rows:
        sampled_math_failures.extend(math_row_failures(row, temperature))
    sampled_rank_failures: list[str] = []
    for group in rank_groups:
        sampled_rank_failures.extend(check_group_ranks(group))

    alt_true_audit = audit_alternative_true(alt_true_rows, known_all, split_known)
    valid_neg_audit = audit_valid_negatives(valid_neg_rows, known_all, split_known)
    frozen_compare = summarize_comparisons(frozen_comparisons, label="dump_vs_frozen_db")
    live_compare = summarize_comparisons(live_comparisons, label="dump_vs_live_20qa")
    live_rescore = load_live_rescore(live_rescore_path) if live_rescore_path else None

    checkpoint_info = inspect_checkpoint(metadata.get("checkpoint") or next(iter(checkpoints), None))
    listed_info = inspect_checkpoint(listed_adapter) if listed_adapter else None
    gpu = gpu_occupancy()

    completeness_ok = (
        n_completeness_failures == 0
        and not missing_expected
        and not extra_expected
        and not summary_missing
        and not summary_extra
        and n_duplicate_qa == 0
        and n_nan == 0
        and n_inf == 0
        and n_empty_rel == 0
        and n_bad_token == 0
        and n_qa * len(relation_order) == n_rows
        and n_gold == n_qa
    )
    math_ok = n_math_failures == 0 and n_mass_failures == 0 and not sampled_math_failures
    rank_ok = n_rank_failures == 0 and not sampled_rank_failures
    fn_ok = alt_true_audit["n_missing_from_kg"] == 0 and valid_neg_audit["n_known_true_leaked"] == 0
    live_ok = (not live_comparisons) or (
        live_compare["max_abs_error"] is not None
        and live_compare["max_abs_error"] <= SCORE_COMPARE_ATOL
        and live_compare["n_rank_mismatch_total"] == 0
    )
    live_rescore_ok = live_rescore_is_ok(live_rescore) if live_rescore else None


    leakage = {
        "relation_vocab_split": "train.txt insertion order (process.py unique_rel)",
        "relation_vocab_size": len(relation_order),
        "true_relation_filter_splits": [f"{split}.txt" for split in filter_splits],
        "scorer_checkpoint": checkpoint_info,
        "listed_checkpoint": listed_info,
        "checkpoint_training_data": {
            "task_file": str(task_file),
            "qa_samples": len(qa_texts),
            "context_samples": len(payload.get("context_samples") or []),
            "b1_objective": "generation loss only (lambda_candidate=0)",
            "listed_objective": "generation + InfoNCE (lambda_candidate=1)",
            "b1_saw_scored_qa": True,
            "b1_saw_eval_relation_prediction_labels": False,
        },
        "offline_mining_qa": {
            "source": str(task_file),
            "split_label": metadata.get("split", "train"),
            "n_matchable_relation_qa": len(expected_ids),
            "n_unmatched_relation_qa": len(unmatched_ids),
            "n_non_relation_qa": len(non_relation_ids),
            "n_scored": n_qa,
        },
        "pair_overlap": {
            "entity_pair_missing": n_pair_missing,
            "qa_pair_in_train": n_pair_in_train,
            "qa_pair_in_valid": n_pair_in_valid,
            "qa_pair_in_test": n_pair_in_test,
            "alternative_true_from_train": n_alt_from_train,
            "alternative_true_from_valid": n_alt_from_valid,
            "alternative_true_from_test": n_alt_from_test,
        },
        "path_A_test_analysis": {
            "safe": True,
            "reason": (
                "Dump scores Stage-2 train QA only. It is not a test-set confusion table. "
                "Do not treat these ranks as test EM / test 10-way analysis."
            ),
        },
        "path_B_train_sampler": {
            "safe": set(filter_splits).issubset({"train"}),
            "red_flags": (
                [
                    "B1 generation training already saw the scored Stage-2 questions. "
                    "These scores are in-distribution teacher values, not a held-out val set."
                ]
                if set(filter_splits).issubset({"train"})
                else [
                    "known_pair_relations() includes valid.txt or test.txt, including test KG structure. "
                    "A future train sampler that drops alternative_true candidates therefore "
                    "uses non-train KG structure."
                ]
            ),
        },
    }

    issues = []
    if not completeness_ok:
        issues.append("data_completeness")
    if not math_ok:
        issues.append("numerical_consistency")
    if not rank_ok:
        issues.append("rank_consistency")
    if not fn_ok:
        issues.append("false_negative_bug")
    if not leakage["path_B_train_sampler"]["safe"]:
        issues.append("test_structure_in_train_filter")
    if frozen_comparisons:
        issues.append("frozen_20260921_table_is_not_this_dump")
    if compare_scores_path is not None and live_comparisons and not live_ok:
        issues.append("scoring_inconsistency_historical_compare20")
    if live_rescore is None:
        issues.append("live_7b_rescore_unavailable")
    elif not live_rescore_ok:
        issues.append("scoring_inconsistency_live_20qa")

    blocking = [
        item
        for item in issues
        if item
        in {
            "false_negative_bug",
            "scoring_inconsistency_live_20qa",
            "live_7b_rescore_unavailable",
            "numerical_consistency",
            "rank_consistency",
            "data_completeness",
            "test_structure_in_train_filter",
        }
    ]
    status = "REVISE" if blocking else "PASS"

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
        "metadata": metadata,
        "task_file": str(task_file),
        "raw_dir": str(raw_dir),
        "frozen_db_path": None if frozen_db_path is None else str(frozen_db_path),
        "compare_scores_path": None if compare_scores_path is None else str(compare_scores_path),
        "live_rescore_path": None if live_rescore_path is None else str(live_rescore_path),
        "sampled_scorer_qa_ids": scorer_ids,
        "gpu": gpu,
        "dataset": {
            "n_qa_samples": len(qa_texts),
            "n_expected_matchable": len(expected_ids),
            "n_unmatched_relation_qa": len(unmatched_ids),
            "n_non_relation_qa": len(non_relation_ids),
            "n_scored_qa": n_qa,
            "n_candidate_rows": n_rows,
            "vocab_size": len(relation_order),
            "n_gold_rows": n_gold,
            "n_valid_negatives": n_valid,
            "n_alternative_true": n_alt_true,
            "invalid_reasons": dict(invalid_reasons),
            "n_missing_expected_qa": len(missing_expected),
            "missing_expected_qa": missing_expected[:20],
            "n_extra_qa": len(extra_expected),
            "extra_qa": extra_expected[:20],
            "summary_missing": summary_missing[:20],
            "summary_extra": summary_extra[:20],
            "n_duplicate_qa": n_duplicate_qa,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "n_empty_relation": n_empty_rel,
            "n_bad_token_length": n_bad_token,
            "n_entity_pair_missing": n_pair_missing,
            "gold_relations_observed": len(gold_relations),
            "matched_relations_observed": len(matched_relations),
            "splits": sorted(splits),
            "checkpoints": sorted(checkpoints),
        },
        "completeness_ok": completeness_ok,
        "completeness_failures": completeness_failures,
        "n_completeness_failures": n_completeness_failures,
        "numerical": {
            "all_rows_checked": n_rows,
            "mass_failures": mass_failures,
            "n_mass_failures": n_mass_failures,
            "math_failures": math_failures,
            "n_math_failures": n_math_failures,
            "sampled_rows": len(math_rows),
            "sampled_math_failures": sampled_math_failures,
            "ok": math_ok,
        },
        "ranking": {
            "all_qa_checked": n_qa,
            "n_rank_failures": n_rank_failures,
            "rank_failures": rank_failures,
            "sampled_qa": len(rank_groups),
            "sampled_rank_failures": sampled_rank_failures,
            "ok": rank_ok,
        },
        "false_negative": {
            "alternative_true": alt_true_audit,
            "valid_negatives": valid_neg_audit,
            "ok": fn_ok,
            "kg_limitation": (
                "known_pair_relations unions train/valid/test. Facts absent from all three "
                "splits cannot be filtered. QA with a missing entity pair skip pair-level "
                "filtering entirely."
            ),
        },
        "leakage": leakage,
        "scorer_consistency": {
            "training_scorer": "ListedContrastiveTrainer._score_candidate_rows -> score_candidate_rows",
            "offline_scorer": "score_relation_candidates_offline.score_relations -> score_candidates",
            "shared_implementation": "score_packed_candidate_rows mean logprob / answer token count",
            "sampled_qa_ids": scorer_ids,
            "live_7b_rescore": live_rescore,
            "gpu_probe": gpu,
            "dump_vs_frozen_db": frozen_compare,
            "dump_vs_historical_compare20": live_compare,
            "dump_vs_training": None if live_rescore is None else live_rescore.get("dump_vs_training"),
            "dump_vs_offline": None if live_rescore is None else live_rescore.get("dump_vs_offline"),
            "training_vs_offline": None if live_rescore is None else live_rescore.get("training_vs_offline"),
            "ok": live_rescore_ok,
            "cpu_identity_note": (
                "tests/test_score_candidates.py compares the listed trainer scorer, "
                "score_candidate_rows, and score_candidates on CPU toy models."
            ),
        },
        "reproducibility": {
            "training_run1_vs_run2": None if live_rescore is None else live_rescore.get("training_run1_vs_run2"),
            "offline_run1_vs_run2": None if live_rescore is None else live_rescore.get("offline_run1_vs_run2"),
            "dump_vs_historical_compare20": live_compare,
            "dump_vs_frozen_20260921_table": frozen_compare,
            "frozen_note": (
                "results/runs/20260923_confusion_db_frozen_b1 comes from the 20260921 "
                "score-hard mining table, not this production dump. Score drift there is "
                "expected and is not an identity check for this dataset."
            ),
            "gpu_nondeterminism": (
                "bf16 packed attention can move scores by one ulp when candidate_batch_size "
                "changes (documented for 16/32 vs 8). Same batch size should match within 1e-5."
            ),
            "ok": live_rescore_ok,
        },
        "issues": issues,
        "blocking_issues": blocking,
        "final_status": status,
    }


def audit_alternative_true(
    rows: list[dict],
    known_all: dict[tuple[str, str], set[str]],
    split_known: dict[str, dict[tuple[str, str], set[str]]],
) -> dict:
    missing = []
    present = 0
    split_hits = Counter()
    for row in rows:
        head, tail = row.get("head_entity"), row.get("tail_entity")
        relation = str(row.get("candidate_relation") or "")
        if not head or not tail:
            missing.append(
                {
                    "qa_id": row.get("qa_id"),
                    "candidate_relation": relation,
                    "reason": "entity_pair_missing",
                }
            )
            continue
        pair = (str(head), str(tail))
        if relation not in known_all.get(pair, set()):
            missing.append(
                {
                    "qa_id": row.get("qa_id"),
                    "candidate_relation": relation,
                    "head_entity": head,
                    "tail_entity": tail,
                    "reason": "not_in_union_kg",
                }
            )
            continue
        present += 1
        for split, mapping in split_known.items():
            if relation in mapping.get(pair, set()):
                split_hits[split] += 1
    return {
        "sampled": len(rows),
        "n_present_in_kg": present,
        "n_missing_from_kg": len(missing),
        "missing_examples": missing[:20],
        "split_hits": dict(split_hits),
    }


def audit_valid_negatives(
    rows: list[dict],
    known_all: dict[tuple[str, str], set[str]],
    split_known: dict[str, dict[tuple[str, str], set[str]]],
) -> dict:
    leaked = []
    for row in rows:
        head, tail = row.get("head_entity"), row.get("tail_entity")
        relation = str(row.get("candidate_relation") or "")
        if not head or not tail:
            continue
        pair = (str(head), str(tail))
        if relation in known_all.get(pair, set()):
            leaked.append(
                {
                    "qa_id": row.get("qa_id"),
                    "candidate_relation": relation,
                    "head_entity": head,
                    "tail_entity": tail,
                    "in_train": relation in split_known["train"].get(pair, set()),
                    "in_valid": relation in split_known["valid"].get(pair, set()),
                    "in_test": relation in split_known["test"].get(pair, set()),
                }
            )
    return {
        "sampled": len(rows),
        "n_known_true_leaked": len(leaked),
        "leaked_examples": leaked[:20],
    }


def summarize_comparisons(rows: list[dict], *, label: str) -> dict:
    if not rows:
        return {
            "label": label,
            "n_qa": 0,
            "max_abs_error": None,
            "mean_abs_error": None,
            "n_score_mismatch_1e5": 0,
            "n_rank_mismatch_total": 0,
            "output_order_mismatch": False,
        }
    max_err = max(float(row["max_abs_error"] or 0.0) for row in rows)
    mean_err = sum(float(row["mean_abs_error"] or 0.0) for row in rows) / len(rows)
    return {
        "label": label,
        "n_qa": len(rows),
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "n_score_mismatch_1e5": sum(int(row.get("n_score_mismatch_1e5") or 0) for row in rows),
        "n_rank_mismatch_total": sum(int(row.get("n_rank_mismatch") or 0) for row in rows),
        "output_order_mismatch": any(bool(row.get("output_order_mismatch")) for row in rows),
        "examples": rows[:5],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_audit_markdown(result: dict) -> str:
    dataset = result["dataset"]
    leakage = result["leakage"]
    scorer = result["scorer_consistency"]
    fn = result["false_negative"]
    status = result["final_status"]
    gpu = result["gpu"]
    checkpoint = leakage["scorer_checkpoint"]
    meta = result.get("metadata") or {}
    red_flags = leakage["path_B_train_sampler"]["red_flags"]
    red_block = "\n".join(f"- **RED:** {item}" for item in red_flags) or "- none"
    frozen = scorer.get("dump_vs_frozen_db") or {}
    historical = scorer.get("dump_vs_historical_compare20") or {}
    training_vs_offline = scorer.get("training_vs_offline") or {}
    dump_vs_training = scorer.get("dump_vs_training") or {}
    dump_vs_offline = scorer.get("dump_vs_offline") or {}
    train_repeat = result["reproducibility"].get("training_run1_vs_run2") or {}
    offline_repeat = result["reproducibility"].get("offline_run1_vs_run2") or {}

    def fmt_err(payload: dict) -> str:
        if not payload or payload.get("max_abs_error") is None:
            return "not compared"
        return (
            f"n_qa={payload.get('n_qa')} max_abs={payload.get('max_abs_error'):.6g} "
            f"mean_abs={payload.get('mean_abs_error'):.6g} "
            f"rank_mismatches={payload.get('n_rank_mismatch_total')}"
        )

    live_status = (
        "not run"
        if scorer.get("live_7b_rescore") is None
        else ("PASS" if scorer.get("ok") else "FAIL")
    )

    return f"""# Dataset

- Task file: `{result['task_file']}`
- Scores: `{result['scores_path']}`
- QA summaries: `{result['summary_path']}`
- Frozen DB (20260921 mining table, not this dump): `{result.get('frozen_db_path')}`
- Live 20-QA rescore: `{result.get('live_rescore_path')}`
- Split label: `{meta.get('split', dataset.get('splits'))}`
- Stage-2 QA samples: {dataset['n_qa_samples']}
- Matchable relation QA expected: {dataset['n_expected_matchable']}
- Unmatched relation-like golds skipped: {dataset['n_unmatched_relation_qa']}
- Non-relation QA skipped: {dataset['n_non_relation_qa']}
- Scored QA: {dataset['n_scored_qa']}
- Candidate rows: {dataset['n_candidate_rows']}
- Vocab size: {dataset['vocab_size']}
- Gold rows: {dataset['n_gold_rows']}
- Valid negatives: {dataset['n_valid_negatives']}
- Alternative true: {dataset['n_alternative_true']}
- Duplicate QA: {dataset['n_duplicate_qa']}
- Empty relations: {dataset['n_empty_relation']}
- NaN: {dataset['n_nan']}
- Inf: {dataset['n_inf']}
- token_length <= 0: {dataset['n_bad_token_length']}
- Entity pair missing: {dataset['n_entity_pair_missing']}
- Completeness: {'PASS' if result['completeness_ok'] else 'FAIL'} ({result['n_completeness_failures']} failures)

# Model / Checkpoint

- Checkpoint: `{checkpoint.get('path')}`
- Exists: {checkpoint.get('exists')}
- Is B1 adapter: {checkpoint.get('is_b1')}
- Is listed adapter: {checkpoint.get('is_listed')}
- B1 run_metadata: `{json.dumps(checkpoint.get('run_metadata') or {}, ensure_ascii=False)}`
- Listed adapter (not used for scoring): `{None if leakage.get('listed_checkpoint') is None else leakage['listed_checkpoint'].get('path')}`
- Tokenizer: `{meta.get('tokenizer')}`
- dtype: `{meta.get('dtype')}`
- candidate_batch_size: `{meta.get('candidate_batch_size')}`
- scoring_version: `{meta.get('scoring_version')}`
- GPU occupancy at audit time: `{json.dumps(gpu, ensure_ascii=False)}`

This scorer is the **B1 generation-only Stage-2 adapter**, not the listed InfoNCE adapter and not a paper RecurrentGRIP snapshot.

# Candidate Scoring Definition

- `score(A|Q)` = length-normalized continuation mean log-probability
- Prefix ends at the last `<answer>` tag, inclusive
- Continuation is the raw relation string only
- EOS and `</answer>` are excluded
- Temperature is **not** applied in the scorer; pairwise_confusion uses T={meta.get('temperature', 1.0)}
- Training path: `ListedContrastiveTrainer._score_candidate_rows` → `score_candidate_rows`
- Offline path: `score_relations` → `score_candidates`
- Shared implementation: `score_packed_candidate_rows`

# Relation Vocabulary Construction

- Source split: **train.txt only**
- Construction: `process.py` insertion order (`unique_rel = list(set())`)
- Size: {dataset['vocab_size']}
- Gold alignment key: `matched_train_relation`
- Observed matched golds: {dataset['matched_relations_observed']} / {dataset['vocab_size']}

# False-negative Filtering

- Filter source: {', '.join(leakage['true_relation_filter_splits'])} via `known_pair_relations()`
- Gold is always invalid (`invalid_reason=gold_relation`)
- Other known pair relations are scored but marked `is_valid_negative=false` (`alternative_true_relation`)
- Sampled alternative_true: {fn['alternative_true']['sampled']}; present in KG: {fn['alternative_true']['n_present_in_kg']}; missing: {fn['alternative_true']['n_missing_from_kg']}
- Sampled valid negatives: {fn['valid_negatives']['sampled']}; known-true leaked into valid pool: {fn['valid_negatives']['n_known_true_leaked']}
- KG limitation: {fn['kg_limitation']}

# Data Leakage Analysis

- Relation vocab split: {leakage['relation_vocab_split']}
- True-relation filtering splits: {', '.join(leakage['true_relation_filter_splits'])}
- Confusion scorer checkpoint: `{checkpoint.get('path')}`
- Checkpoint training data: `{json.dumps(leakage['checkpoint_training_data'], ensure_ascii=False)}`
- Offline mining QA: `{json.dumps(leakage['offline_mining_qa'], ensure_ascii=False)}`
- Pair overlap: `{json.dumps(leakage['pair_overlap'], ensure_ascii=False)}`

## A. Used as test-set confusion analysis

{leakage['path_A_test_analysis']['reason']}

## B. Used as a future training sampler

{red_block}

B1 **did** see the scored Stage-2 questions during generation-only training. That is in-distribution teacher scoring, not test-label leakage. It **is** leakage if these scores are treated as a held-out validation of B1.

Any future sampler that consumes `is_valid_negative` from this dump uses exactly the KG splits listed above.

# Numerical Consistency

- All candidate rows recomputed for `score_gap` and `pairwise_confusion`: failures={result['numerical']['n_math_failures']}
- All QA `negative_mass` sums: failures={result['numerical']['n_mass_failures']}
- Random {result['numerical']['sampled_rows']} row subsample: failures={len(result['numerical']['sampled_math_failures'])}
- All QA ranks recomputed: failures={result['ranking']['n_rank_failures']}
- Random {result['ranking']['sampled_qa']} QA rank subsample: failures={len(result['ranking']['sampled_rank_failures'])}

# Reproducibility

- Training vs offline 20-QA: {fmt_err(training_vs_offline)}
- Dump vs training 20-QA: {fmt_err(dump_vs_training)}
- Dump vs offline 20-QA: {fmt_err(dump_vs_offline)}
- Training run1 vs run2: {fmt_err(train_repeat)}
- Offline run1 vs run2: {fmt_err(offline_repeat)}
- Dump vs historical compare20 (same dump, earlier 20-QA rerun): {fmt_err(historical)}
- Dump vs frozen 20260921 mining table: {fmt_err(frozen)}
- Frozen-table note: {result['reproducibility'].get('frozen_note')}
- GPU nondeterminism: {result['reproducibility']['gpu_nondeterminism']}
- Live 7B 20-QA rescore status: {live_status}

# Known Limitations

- Open-world KG: facts outside train/valid/test cannot be filtered.
- {dataset['n_entity_pair_missing']} QA have no parsed entity pair, so alternative-true filtering is skipped for those rows.
- {dataset['n_unmatched_relation_qa']} unmatched relation-like golds are outside the 198 train-graph names and are not in this dump.
- Analysis artifacts under `analysis/` may have been built from the frozen 20260921 table; `confusion_vocab/` from this dump. Do not mix them.
- NVML `nvidia-smi` can fail with a driver/library mismatch even when `torch.cuda.is_available()` is true.

# Final Status

**{status}**

Blocking issues: {', '.join(result['blocking_issues']) or 'none'}
All issues: {', '.join(result['issues']) or 'none'}
"""


def write_audit_outputs(result: dict, output_dir: Path, markdown_copies: Iterable[Path] = ()) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit.json", result)
    markdown = render_audit_markdown(result)
    report_path = output_dir / "CONFUSION_DATABASE_AUDIT.md"
    report_path.write_text(markdown, encoding="utf-8")
    for path in markdown_copies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {"audit_json": str(output_dir / "audit.json"), "report": str(report_path)}
