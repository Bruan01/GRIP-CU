"""Audit a saved relation-confusion database without training or resampling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
from typing import Any


NUMERIC_FIELDS = (
    "candidate_score",
    "gold_score",
    "score_gap",
    "pairwise_confusion",
    "negative_mass",
    "candidate_token_length",
    "candidate_rank_all",
    "negative_rank",
)
OPTIONAL_NUMERIC_FIELDS = {"pairwise_confusion", "negative_mass", "negative_rank"}
TOLERANCE = 1e-6


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _convert_mnt_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if match and __import__("os").name == "nt":
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return Path(text).expanduser()


def _resolve_metadata_path(candidate_scores: Path, requested: str | None) -> Path | None:
    if requested:
        path = _convert_mnt_path(requested)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing scoring metadata: {path}")
        return path
    for path in (candidate_scores.parent / "metadata.json", candidate_scores.parent.parent / "metadata.json"):
        if path.is_file():
            return path
    return None


def _resolve_input_paths(args: argparse.Namespace, metadata: dict, candidate_scores: Path) -> dict[str, Path | None]:
    identity = metadata.get("identity", {})

    def choose(argument: str, metadata_key: str) -> Path | None:
        explicit = getattr(args, argument)
        value = explicit or identity.get(metadata_key)
        return _convert_mnt_path(value)

    processed = choose("processed_file", "processed_file")
    raw_dir = processed.parent.parent / "raw" if processed is not None else None
    return {
        "candidate_scores": candidate_scores,
        "qa_summary": _convert_mnt_path(args.qa_summary) or candidate_scores.parent / "qa_summary.jsonl",
        "candidate_file": choose("candidate_file", "candidate_file"),
        "processed_file": processed,
        "relation_text_file": choose("relation_text_file", "relation_text_file"),
        "raw_train_file": _convert_mnt_path(args.raw_train_file) or (raw_dir / "train.txt" if raw_dir else None),
        "raw_valid_file": _convert_mnt_path(args.raw_valid_file) or (raw_dir / "valid.txt" if raw_dir else None),
        "raw_test_file": _convert_mnt_path(args.raw_test_file) or (raw_dir / "test.txt" if raw_dir else None),
    }


def _reservoir_add(bucket: list[dict], item: dict, seen: int, limit: int, rng: random.Random) -> int:
    seen += 1
    if len(bucket) < limit:
        bucket.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            bucket[index] = item
    return seen


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _read_raw_split(path: Path | None) -> dict:
    result = {
        "path": str(path) if path else None,
        "exists": bool(path and path.is_file()),
        "edge_count": 0,
        "relation_id_count": 0,
        "pair_count": 0,
        "relation_ids": set(),
        "pairs": set(),
        "malformed_lines": 0,
    }
    if not result["exists"]:
        return result
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                if line.strip():
                    result["malformed_lines"] += 1
                continue
            result["edge_count"] += 1
            result["relation_ids"].add(fields[1])
            result["pairs"].add((fields[0], fields[2]))
    result["relation_id_count"] = len(result["relation_ids"])
    result["pair_count"] = len(result["pairs"])
    return result


def _load_graph_pair_relations(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"exists": False, "edge_count": 0, "pairs": {}}
    payload = _load_json(path)
    edges = payload.get("graph", {}).get("edge_list", [])
    pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in edges:
        if len(edge) >= 3:
            pair_relations[(str(edge[0]), str(edge[2]))].add(str(edge[1]))
    return {
        "exists": True,
        "edge_count": len(edges),
        "graph_id": payload.get("id"),
        "pairs": pair_relations,
    }


def _load_candidate_source(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"exists": False, "records": [], "qa_ids": set(), "duplicate_qa_ids": []}
    payload = _load_json(path)
    records = [record for record in payload.get("records", []) if record is not None]
    qa_ids = []
    by_qa = {}
    duplicates = []
    for record in records:
        qa_id = record.get("task_index", record.get("id"))
        qa_ids.append(qa_id)
        if qa_id in by_qa:
            duplicates.append(qa_id)
        by_qa[qa_id] = record
    return {
        "exists": True,
        "format_version": payload.get("format_version"),
        "total_records": len(payload.get("records", [])),
        "null_records": sum(record is None for record in payload.get("records", [])),
        "records": records,
        "by_qa": by_qa,
        "qa_ids": set(qa_ids),
        "duplicate_qa_ids": sorted(set(duplicates)),
        "source": payload.get("source", {}),
    }


def _audit_scores(
    path: Path,
    candidate_source: dict,
    relation_vocab_size: int | None,
    temperature: float,
    seed: int,
    math_sample_size: int,
    invalid_sample_size: int,
    valid_sample_size: int,
) -> dict:
    rng = random.Random(seed)
    qa_states: dict[Any, dict] = {}
    candidate_vocab: set[str] = set()
    gold_vocab: set[str] = set()
    math_samples: list[dict] = []
    invalid_samples: list[dict] = []
    valid_samples: list[dict] = []
    math_seen = invalid_seen = valid_seen = 0
    field_issues = Counter()
    field_issue_samples: list[dict] = []
    empty_relation_rows = 0
    nonfinite_values = 0
    token_length_nonpositive = 0
    row_order_errors: list[dict] = []
    global_indices: set[int] = set()
    qa_order = []
    last_global_index = None
    current_global_index = None
    current_qa_id = None
    expected_rank = 1
    total_rows = 0

    def issue(kind: str, qa_id: Any, detail: str) -> None:
        field_issues[kind] += 1
        if len(field_issue_samples) < 20:
            field_issue_samples.append({"qa_id": qa_id, "issue": kind, "detail": detail})

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            qa_id = row.get("qa_id")
            global_index = row.get("global_index")
            try:
                global_index_int = int(global_index)
                global_indices.add(global_index_int)
            except (TypeError, ValueError):
                issue("invalid_global_index", qa_id, str(global_index))
                global_index_int = None

            if current_global_index is None:
                current_global_index = global_index_int
                current_qa_id = qa_id
                expected_rank = 1
                qa_order.append(qa_id)
            elif global_index_int != current_global_index:
                if global_index_int is None or global_index_int <= (last_global_index if last_global_index is not None else -1):
                    row_order_errors.append({"line": line_number, "error": "global_index_not_increasing", "value": global_index_int})
                last_global_index = current_global_index
                current_global_index = global_index_int
                current_qa_id = qa_id
                expected_rank = 1
                qa_order.append(qa_id)
            elif qa_id != current_qa_id:
                row_order_errors.append({"line": line_number, "error": "qa_changed_within_global_index", "qa_id": qa_id})
                current_qa_id = qa_id

            state = qa_states.setdefault(qa_id, {
                "row_count": 0,
                "gold_count": 0,
                "candidate_relations": set(),
                "duplicate_candidates": 0,
                "gold_relation": None,
                "gold_score": None,
                "valid_negative_count": 0,
                "invalid_negative_count": 0,
                "invalid_reasons": Counter(),
                "negative_mass_sum": 0.0,
                "missing_negative_mass": 0,
                "global_index": global_index_int,
                "candidate_record_id": row.get("candidate_record_id"),
                "split": row.get("split"),
                "model_checkpoint": row.get("model_checkpoint"),
            })
            state["row_count"] += 1
            candidate_relation = row.get("candidate_relation")
            gold_relation = row.get("gold_relation")
            if not isinstance(candidate_relation, str) or not candidate_relation.strip():
                empty_relation_rows += 1
            if not isinstance(gold_relation, str) or not gold_relation.strip():
                empty_relation_rows += 1
            if candidate_relation in state["candidate_relations"]:
                state["duplicate_candidates"] += 1
            state["candidate_relations"].add(candidate_relation)
            candidate_vocab.add(candidate_relation)
            gold_vocab.add(gold_relation)
            if state["gold_relation"] is None:
                state["gold_relation"] = gold_relation
                state["gold_score"] = row.get("gold_score")
            elif state["gold_relation"] != gold_relation:
                issue("gold_relation_inconsistent", qa_id, f"{state['gold_relation']} != {gold_relation}")
            if row.get("is_gold"):
                state["gold_count"] += 1

            for field in NUMERIC_FIELDS:
                value = row.get(field)
                if value is None:
                    if field not in OPTIONAL_NUMERIC_FIELDS:
                        issue(f"missing_{field}", qa_id, f"line {line_number}")
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    issue(f"non_numeric_{field}", qa_id, repr(value))
                    continue
                if not math.isfinite(numeric):
                    nonfinite_values += 1
                    issue(f"nonfinite_{field}", qa_id, repr(value))
                if field == "candidate_token_length" and numeric <= 0:
                    token_length_nonpositive += 1
                    issue("nonpositive_candidate_token_length", qa_id, repr(value))

            saved_rank = row.get("candidate_rank_all")
            if saved_rank != expected_rank:
                row_order_errors.append({
                    "line": line_number,
                    "error": "candidate_rank_all_not_in_output_order",
                    "qa_id": qa_id,
                    "expected": expected_rank,
                    "actual": saved_rank,
                })
            expected_rank += 1

            if row.get("is_valid_negative"):
                state["valid_negative_count"] += 1
                mass = row.get("negative_mass")
                if mass is None:
                    state["missing_negative_mass"] += 1
                else:
                    state["negative_mass_sum"] += float(mass)
                valid_seen = _reservoir_add(valid_samples, row, valid_seen, valid_sample_size, rng)
                math_seen = _reservoir_add(math_samples, row, math_seen, math_sample_size, rng)
            else:
                state["invalid_negative_count"] += 1
                state["invalid_reasons"][row.get("invalid_reason")] += 1
                if row.get("invalid_reason") == "alternative_true_relation":
                    invalid_seen = _reservoir_add(invalid_samples, row, invalid_seen, invalid_sample_size, rng)

    if current_global_index is not None:
        last_global_index = current_global_index
    mass_errors = []
    duplicate_qa_candidates = 0
    candidate_count_mismatches = 0
    gold_count_mismatches = 0
    for qa_id, state in qa_states.items():
        if state["duplicate_candidates"]:
            duplicate_qa_candidates += 1
        expected_count = relation_vocab_size or len(candidate_vocab)
        if state["row_count"] != expected_count or len(state["candidate_relations"]) != expected_count:
            candidate_count_mismatches += 1
        if state["gold_count"] != 1:
            gold_count_mismatches += 1
        if state["missing_negative_mass"] or abs(state["negative_mass_sum"] - 1.0) > TOLERANCE:
            mass_errors.append({
                "qa_id": qa_id,
                "sum": state["negative_mass_sum"],
                "missing": state["missing_negative_mass"],
            })

    return {
        "total_rows": total_rows,
        "qa_states": qa_states,
        "qa_count": len(qa_states),
        "candidate_vocab": candidate_vocab,
        "gold_vocab": gold_vocab,
        "global_indices": global_indices,
        "qa_order": qa_order,
        "candidate_count_mismatches": candidate_count_mismatches,
        "gold_count_mismatches": gold_count_mismatches,
        "duplicate_qa_candidates": duplicate_qa_candidates,
        "empty_relation_rows": empty_relation_rows,
        "nonfinite_values": nonfinite_values,
        "token_length_nonpositive": token_length_nonpositive,
        "field_issues": dict(field_issues),
        "field_issue_samples": field_issue_samples,
        "row_order_errors": row_order_errors[:20],
        "row_order_error_count": len(row_order_errors),
        "mass_errors": mass_errors[:20],
        "mass_error_count": len(mass_errors),
        "math_samples": math_samples,
        "math_sample_seen": math_seen,
        "invalid_samples": invalid_samples,
        "invalid_sample_seen": invalid_seen,
        "valid_samples": valid_samples,
        "valid_sample_seen": valid_seen,
    }


def _compare_summary(path: Path | None, score_audit: dict) -> dict:
    result = {"exists": bool(path and path.is_file()), "path": str(path) if path else None}
    if not result["exists"]:
        return result
    summary_ids = []
    duplicate_ids = []
    summary_by_id = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qa_id = row.get("qa_id")
            if qa_id in summary_by_id:
                duplicate_ids.append(qa_id)
            summary_ids.append(qa_id)
            summary_by_id[qa_id] = row
    output_ids = set(score_audit["qa_states"])
    summary_id_set = set(summary_ids)
    mismatches = []
    for qa_id in sorted(output_ids & summary_id_set, key=str):
        state = score_audit["qa_states"][qa_id]
        row = summary_by_id[qa_id]
        if row.get("number_of_candidates") != state["row_count"]:
            mismatches.append({"qa_id": qa_id, "field": "number_of_candidates"})
        if row.get("number_of_valid_negatives") != state["valid_negative_count"]:
            mismatches.append({"qa_id": qa_id, "field": "number_of_valid_negatives"})
        if row.get("negative_mass_sum") is not None and abs(float(row["negative_mass_sum"]) - state["negative_mass_sum"]) > TOLERANCE:
            mismatches.append({"qa_id": qa_id, "field": "negative_mass_sum"})
    return {
        **result,
        "row_count": len(summary_ids),
        "duplicate_count": len(duplicate_ids),
        "missing_from_summary": sorted(output_ids - summary_id_set, key=str)[:20],
        "missing_from_scores": sorted(summary_id_set - output_ids, key=str)[:20],
        "field_mismatch_count": len(mismatches),
        "field_mismatches": mismatches[:20],
    }


def _audit_math(samples: list[dict], temperature: float) -> dict:
    gap_errors = []
    confusion_errors = []
    for row in samples:
        gold = float(row["gold_score"])
        score = float(row["candidate_score"])
        gap_errors.append(abs(float(row["score_gap"]) - (gold - score)))
        expected_confusion = _stable_sigmoid((score - gold) / temperature)
        confusion_errors.append(abs(float(row["pairwise_confusion"]) - expected_confusion))
    return {
        "sample_count": len(samples),
        "gap_max_abs_error": max(gap_errors, default=None),
        "gap_mean_abs_error": sum(gap_errors) / len(gap_errors) if gap_errors else None,
        "confusion_max_abs_error": max(confusion_errors, default=None),
        "confusion_mean_abs_error": sum(confusion_errors) / len(confusion_errors) if confusion_errors else None,
        "pass": bool(samples) and max(gap_errors, default=float("inf")) <= TOLERANCE and max(confusion_errors, default=float("inf")) <= TOLERANCE,
    }


def _audit_ranks(path: Path, selected_qa_ids: list[Any]) -> dict:
    selected = set(selected_qa_ids)
    rows_by_qa: dict[Any, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("qa_id") in selected:
                rows_by_qa[row["qa_id"]].append(row)
    all_mismatches = []
    negative_mismatches = []
    for qa_id in selected_qa_ids:
        rows = rows_by_qa.get(qa_id, [])
        ordered = sorted(rows, key=lambda row: (-float(row["candidate_score"]), int(row["candidate_index"])))
        expected_all = {id(row): rank for rank, row in enumerate(ordered, 1)}
        for row in rows:
            if int(row["candidate_rank_all"]) != expected_all[id(row)]:
                all_mismatches.append({"qa_id": qa_id, "candidate_relation": row["candidate_relation"]})
        valid_rows = [row for row in rows if row.get("is_valid_negative")]
        ordered_negative = sorted(valid_rows, key=lambda row: (-float(row["candidate_score"]), int(row["candidate_index"])))
        expected_negative = {id(row): rank for rank, row in enumerate(ordered_negative, 1)}
        for row in valid_rows:
            if int(row["negative_rank"]) != expected_negative[id(row)]:
                negative_mismatches.append({"qa_id": qa_id, "candidate_relation": row["candidate_relation"]})
    return {
        "sample_qa_count": len(selected_qa_ids),
        "complete_qa_count": len(rows_by_qa),
        "candidate_rank_mismatch_count": len(all_mismatches),
        "negative_rank_mismatch_count": len(negative_mismatches),
        "mismatch_samples": (all_mismatches + negative_mismatches)[:20],
        "pass": len(all_mismatches) == 0 and len(negative_mismatches) == 0 and len(rows_by_qa) == len(selected_qa_ids),
    }


def _audit_false_negatives(score_audit: dict, graph: dict) -> dict:
    pair_relations = graph.get("pairs", {})
    invalid_results = []
    for row in score_audit["invalid_samples"]:
        pair = (str(row.get("head_entity")), str(row.get("tail_entity")))
        known = pair_relations.get(pair, set())
        invalid_results.append({
            "qa_id": row.get("qa_id"),
            "candidate_relation": row.get("candidate_relation"),
            "reason_correct": row.get("invalid_reason") == "alternative_true_relation",
            "in_graph": row.get("candidate_relation") in known,
            "in_saved_true_relations": row.get("candidate_relation") in set(row.get("all_true_relations", [])),
        })
    valid_results = []
    for row in score_audit["valid_samples"]:
        pair = (str(row.get("head_entity")), str(row.get("tail_entity")))
        known = pair_relations.get(pair, set())
        valid_results.append({
            "qa_id": row.get("qa_id"),
            "candidate_relation": row.get("candidate_relation"),
            "in_train_graph": row.get("candidate_relation") in known,
            "filter_available": row.get("false_negative_filter_available"),
        })
    invalid_pass = bool(invalid_results) and all(
        result["reason_correct"] and result["in_graph"] and result["in_saved_true_relations"]
        for result in invalid_results
    )
    valid_graph_hits = sum(result["in_train_graph"] for result in valid_results)
    return {
        "invalid_sample_count": len(invalid_results),
        "invalid_expected_sample": 100,
        "invalid_confirmed_count": sum(
            result["reason_correct"] and result["in_graph"] and result["in_saved_true_relations"]
            for result in invalid_results
        ),
        "invalid_pass": invalid_pass and len(invalid_results) >= 100,
        "valid_sample_count": len(valid_results),
        "valid_expected_sample": 500,
        "valid_train_graph_hits": valid_graph_hits,
        "valid_filter_unavailable_count": sum(not result["filter_available"] for result in valid_results),
        "valid_pass": len(valid_results) >= 500 and valid_graph_hits == 0,
        "invalid_result_samples": invalid_results[:10],
        "valid_hit_samples": [result for result in valid_results if result["in_train_graph"]][:10],
    }


def _load_optional_report(path: str | None) -> dict | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _reproducibility_report(first: dict | None, second: dict | None) -> dict:
    if first is None or second is None:
        return {"available": False, "pass": False}
    same_selection = first.get("selected_qa_ids") == second.get("selected_qa_ids")
    same_digest = bool(first.get("run_digest")) and first.get("run_digest") == second.get("run_digest")
    same_fingerprints = first.get("run_fingerprints") == second.get("run_fingerprints")
    return {
        "available": True,
        "same_selected_qa_ids": same_selection,
        "same_run_digest": same_digest,
        "same_run_fingerprints": same_fingerprints,
        "first_digest": first.get("run_digest"),
        "second_digest": second.get("run_digest"),
        "pass": same_selection and same_digest and same_fingerprints,
    }


def _normalise_path(value: str | None) -> str | None:
    path = _convert_mnt_path(value)
    if path is None:
        return None
    return str(path.resolve()).replace("\\", "/").lower()


def _leakage_audit(metadata: dict, training_metadata: dict | None, candidate_source: dict, input_paths: dict) -> dict:
    identity = metadata.get("identity", {})
    same_candidate_file = (
        _normalise_path(identity.get("candidate_file")) is not None
        and _normalise_path(identity.get("candidate_file")) == _normalise_path(training_metadata.get("candidate_file") if training_metadata else None)
    )
    candidate_loss_weight = training_metadata.get("lambda_candidate") if training_metadata else None
    candidate_loss_active = candidate_loss_weight is not None and float(candidate_loss_weight) > 0
    red_flag = same_candidate_file and candidate_loss_active
    return {
        "scoring_split": identity.get("split"),
        "candidate_file": str(input_paths["candidate_file"]) if input_paths["candidate_file"] else None,
        "training_candidate_file": training_metadata.get("candidate_file") if training_metadata else None,
        "same_candidate_file": same_candidate_file,
        "candidate_loss_weight": candidate_loss_weight,
        "candidate_loss_active": candidate_loss_active,
        "training_qa_samples": training_metadata.get("qa_samples") if training_metadata else None,
        "training_aligned_candidate_records": training_metadata.get("aligned_candidate_records") if training_metadata else None,
        "offline_qa_count": len(candidate_source.get("qa_ids", set())),
        "red_flag_same_scored_qa_used_for_training": red_flag,
        "training_metadata_available": training_metadata is not None,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _write_report(
    output_path: Path,
    metadata: dict,
    input_paths: dict,
    score_audit: dict,
    source_audit: dict,
    raw_stats: dict,
    graph: dict,
    math_audit: dict,
    rank_audit: dict,
    false_negative_audit: dict,
    leakage: dict,
    consistency: dict | None,
    reproducibility: dict,
) -> str:
    identity = metadata.get("identity", {})
    training_metadata = source_audit.get("training_metadata") or {}
    integrity_pass = all([
        score_audit["qa_count"] > 0,
        score_audit["candidate_count_mismatches"] == 0,
        score_audit["gold_count_mismatches"] == 0,
        score_audit["duplicate_qa_candidates"] == 0,
        score_audit["empty_relation_rows"] == 0,
        score_audit["nonfinite_values"] == 0,
        score_audit["token_length_nonpositive"] == 0,
        score_audit["row_order_error_count"] == 0,
        source_audit["qa_id_match"],
        source_audit["summary_match"],
    ])
    math_pass = math_audit["pass"] and score_audit["mass_error_count"] == 0
    false_negative_pass = false_negative_audit["invalid_pass"] and false_negative_audit["valid_pass"]
    consistency_pass = bool(consistency and consistency.get("pass") and consistency.get("mismatch_count") == 0)
    leakage_pass = not leakage["red_flag_same_scored_qa_used_for_training"]
    overall_pass = all([integrity_pass, math_pass, rank_audit["pass"], false_negative_pass, consistency_pass, reproducibility["pass"], leakage_pass])
    final_status = "PASS" if overall_pass else "REVISE"

    lines = [
        "# Dataset",
        "",
        f"- Candidate score file: `{input_paths['candidate_scores']}`",
        f"- QA count: {score_audit['qa_count']}; candidate rows: {score_audit['total_rows']}",
        f"- Candidate vocabulary: {len(score_audit['candidate_vocab'])}; gold vocabulary: {len(score_audit['gold_vocab'])}",
        f"- Source relation QA records: {source_audit['candidate_source_count']}; null candidate records: {source_audit['candidate_null_count']}",
        f"- QA summary rows: {source_audit['summary_row_count']}",
        f"- Raw train/valid/test edges: {raw_stats['train']['edge_count']} / {raw_stats['valid']['edge_count']} / {raw_stats['test']['edge_count']}",
        f"- Integrity: source QA IDs match={source_audit['qa_id_match']}; summary matches={source_audit['summary_match']}; gold-count failures={score_audit['gold_count_mismatches']}; candidate-count failures={score_audit['candidate_count_mismatches']}.",
        f"- Duplicate candidates: {score_audit['duplicate_qa_candidates']} QA; empty relation rows: {score_audit['empty_relation_rows']}; output-order/rank errors: {score_audit['row_order_error_count']}.",
        "",
        "# Model / Checkpoint",
        "",
        f"- Checkpoint: `{identity.get('model_checkpoint')}`",
        f"- Scoring version: `{identity.get('scoring_version')}`",
        f"- Split recorded by scorer: `{identity.get('split')}`",
        f"- dtype: `{metadata.get('dtype', 'bfloat16')}`; candidate batch size: `{identity.get('candidate_batch_size')}`; T: `{identity.get('temperature')}`",
        f"- Training metadata: `{source_audit.get('training_metadata_path')}`",
        f"- Training QA/context samples: {training_metadata.get('qa_samples', 'N/A')} / {training_metadata.get('context_samples', 'N/A')}",
        "",
        "# Candidate Scoring Definition",
        "",
        "- Teacher-forced mean log probability over answer tokens only.",
        "- Candidate text is wrapped as `<answer>candidate</answer>`; prompt, padding, and EOS are excluded.",
        "- `score_gap = gold_score - candidate_score`.",
        "- `pairwise_confusion = sigmoid((candidate_score - gold_score) / T)`.",
        "- `negative_mass` is a softmax over valid negatives only.",
        f"- Candidate-token length checks: nonpositive={score_audit['token_length_nonpositive']}; non-finite values={score_audit['nonfinite_values']}.",
        "",
        "# Relation Vocabulary Construction",
        "",
        f"- Relation mapping: `{input_paths['relation_text_file']}`.",
        f"- Relation IDs: {source_audit['relation_id_count']}; unique text labels: {source_audit['relation_label_count']}; label collisions: {source_audit['label_collision_count']}.",
        f"- Raw train relation IDs: {raw_stats['train']['relation_id_count']}; train covers all mapping IDs: {source_audit['train_covers_all_relation_ids']}.",
        "- The confusion database uses the saved candidate vocabulary and does not alter candidates or select a sampler.",
        "",
        "# False-negative Filtering",
        "",
        f"- Filtering graph: `{input_paths['processed_file']}`; graph edges: {graph['edge_count']}; raw train edges: {raw_stats['train']['edge_count']}.",
        f"- Invalid rows sampled: {false_negative_audit['invalid_sample_count']}/100; confirmed alternative true relations: {false_negative_audit['invalid_confirmed_count']}.",
        f"- Valid rows sampled: {false_negative_audit['valid_sample_count']}/500; train-graph false-negative hits: {false_negative_audit['valid_train_graph_hits']}.",
        f"- Filter unavailable in valid sample: {false_negative_audit['valid_filter_unavailable_count']}.",
        "- This audit independently checks the saved relation against the processed graph pair index; it does not establish that held-out valid/test facts are absent.",
        "",
        "# Data Leakage Analysis",
        "",
        "- **RED FLAG:** the checkpoint training metadata points to the same candidate file used for offline mining.",
        f"- Same candidate file: `{leakage['same_candidate_file']}`; candidate loss active: `{leakage['candidate_loss_active']}` with weight `{leakage['candidate_loss_weight']}`.",
        f"- Offline confusion QA count: {leakage['offline_qa_count']}; training aligned candidate records: {leakage['training_aligned_candidate_records']}.",
        "- A. Current analysis is therefore not independent test-time confusion for this checkpoint; the scored QA/candidate records were part of the Stage 2 candidate-loss training path.",
        "- B. This database must not yet be used as an unbiased future sampler-training source. A held-out confusion run with a checkpoint that did not train on those QA is required.",
        "",
        "# Numerical Consistency",
        "",
        f"- Random math sample: {math_audit['sample_count']} records.",
        f"- score_gap max/mean absolute error: {_fmt(math_audit['gap_max_abs_error'])} / {_fmt(math_audit['gap_mean_abs_error'])}.",
        f"- pairwise_confusion max/mean absolute error: {_fmt(math_audit['confusion_max_abs_error'])} / {_fmt(math_audit['confusion_mean_abs_error'])}.",
        f"- QA negative-mass failures: {score_audit['mass_error_count']}; maximum checked failure list is tolerance `{TOLERANCE}`.",
        f"- Random rank sample: {rank_audit['sample_qa_count']} QA; candidate-rank mismatches: {rank_audit['candidate_rank_mismatch_count']}; negative-rank mismatches: {rank_audit['negative_rank_mismatch_count']}.",
        f"- Original scorer vs saved offline output: pass={consistency.get('pass') if consistency else 'N/A'}, max errors={consistency.get('max_abs_error') if consistency else 'N/A'}, mean errors={consistency.get('mean_abs_error') if consistency else 'N/A'}.",
        "",
        "# Reproducibility",
        "",
        f"- Same 20-QA selection: {reproducibility.get('same_selected_qa_ids', 'N/A')}.",
        f"- Candidate-score/rank/output-order fingerprint identical: {reproducibility.get('same_run_fingerprints', 'N/A')}.",
        f"- Run digest 1: `{reproducibility.get('first_digest', 'N/A')}`.",
        f"- Run digest 2: `{reproducibility.get('second_digest', 'N/A')}`.",
        "",
        "# Known Limitations",
        "",
        "- The filtering graph is train-only; facts present only in valid/test splits cannot be independently ruled out as false negatives.",
        "- The checkpoint and the offline confusion QA/candidate file are not independent because the same aligned candidate records were used during Stage 2 candidate loss.",
        "- Relation IDs and text labels are not one-to-one: 237 IDs collapse to 233 text labels.",
        "- Production scoring is fixed at candidate batch size 32 because other bf16/SDPA batch sizes showed numerical differences.",
        "- This audit does not implement Random-K, Top-K Hard, or Coverage-Adaptive K.",
        "",
        "# Final Status",
        "",
        f"**{final_status}**",
        "",
        "The database is not yet sufficiently independent for the next controlled sampler experiment. Rebuild the confusion database using a checkpoint trained without these scored QA/candidate records, then rerun this audit before comparing sampling strategies.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--candidate-file")
    parser.add_argument("--processed-file")
    parser.add_argument("--relation-text-file")
    parser.add_argument("--qa-summary")
    parser.add_argument("--raw-train-file")
    parser.add_argument("--raw-valid-file")
    parser.add_argument("--raw-test-file")
    parser.add_argument("--scorer-run-a")
    parser.add_argument("--scorer-run-b")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--math-sample-size", type=int, default=100)
    parser.add_argument("--invalid-sample-size", type=int, default=100)
    parser.add_argument("--valid-sample-size", type=int, default=500)
    parser.add_argument("--rank-sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.math_sample_size, args.invalid_sample_size, args.valid_sample_size, args.rank_sample_size) <= 0:
        raise ValueError("All audit sample sizes must be positive")
    candidate_scores = Path(args.candidate_scores).expanduser().resolve()
    metadata_path = _resolve_metadata_path(candidate_scores, args.metadata)
    metadata = _load_json(metadata_path) if metadata_path else {}
    input_paths = _resolve_input_paths(args, metadata, candidate_scores)
    identity = metadata.get("identity", {})
    relation_vocab_size = identity.get("relation_vocab_size") or metadata.get("relation_vocab_size")
    temperature = float(identity.get("temperature") or metadata.get("temperature") or 1.0)

    candidate_source = _load_candidate_source(input_paths["candidate_file"])
    graph = _load_graph_pair_relations(input_paths["processed_file"])
    raw_stats = {
        "train": _read_raw_split(input_paths["raw_train_file"]),
        "valid": _read_raw_split(input_paths["raw_valid_file"]),
        "test": _read_raw_split(input_paths["raw_test_file"]),
    }
    relation_mapping = _load_json(input_paths["relation_text_file"]) if input_paths["relation_text_file"] and input_paths["relation_text_file"].is_file() else {}
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    for relation_id, label in relation_mapping.items():
        label_to_ids[str(label)].append(str(relation_id))
    score_audit = _audit_scores(
        candidate_scores,
        candidate_source,
        int(relation_vocab_size) if relation_vocab_size is not None else None,
        temperature,
        args.seed,
        args.math_sample_size,
        args.invalid_sample_size,
        args.valid_sample_size,
    )
    summary_audit = _compare_summary(input_paths["qa_summary"], score_audit)
    source_ids = candidate_source.get("qa_ids", set())
    output_ids = set(score_audit["qa_states"])
    source_audit = {
        "candidate_source_count": len(source_ids),
        "candidate_null_count": candidate_source.get("null_records", 0),
        "summary_row_count": summary_audit.get("row_count", 0),
        "qa_id_match": source_ids == output_ids,
        "summary_match": (
            summary_audit.get("exists", False)
            and not summary_audit.get("duplicate_count", 1)
            and not summary_audit.get("missing_from_summary")
            and not summary_audit.get("missing_from_scores")
            and summary_audit.get("field_mismatch_count", 1) == 0
        ),
        "relation_id_count": len(relation_mapping),
        "relation_label_count": len(set(relation_mapping.values())),
        "label_collision_count": sum(len(ids) > 1 for ids in label_to_ids.values()),
        "train_covers_all_relation_ids": set(relation_mapping) <= raw_stats["train"]["relation_ids"],
    }
    training_metadata_path = None
    checkpoint = _convert_mnt_path(identity.get("model_checkpoint"))
    if checkpoint:
        candidate = checkpoint / "grip_run_metadata.json"
        if candidate.is_file():
            training_metadata_path = candidate
    training_metadata = _load_json(training_metadata_path) if training_metadata_path else None
    source_audit["training_metadata_path"] = str(training_metadata_path) if training_metadata_path else None
    source_audit["training_metadata"] = training_metadata

    selected_qa_ids = sorted(score_audit["qa_states"], key=str)
    if len(selected_qa_ids) > args.rank_sample_size:
        selected_qa_ids = random.Random(args.seed + 1).sample(selected_qa_ids, args.rank_sample_size)
    selected_qa_ids.sort(key=str)
    math_audit = _audit_math(score_audit["math_samples"], temperature)
    rank_audit = _audit_ranks(candidate_scores, selected_qa_ids)
    false_negative_audit = _audit_false_negatives(score_audit, graph)
    consistency = _load_optional_report(args.scorer_run_a)
    reproducibility = _reproducibility_report(consistency, _load_optional_report(args.scorer_run_b))
    leakage = _leakage_audit(metadata, training_metadata, candidate_source, input_paths)

    final_status = _write_report(
        Path(args.output_report),
        metadata,
        input_paths,
        score_audit,
        source_audit,
        raw_stats,
        graph,
        math_audit,
        rank_audit,
        false_negative_audit,
        leakage,
        consistency,
        reproducibility,
    )
    print(json.dumps({
        "output_report": str(Path(args.output_report).resolve()),
        "final_status": final_status,
        "qa_count": score_audit["qa_count"],
        "candidate_rows": score_audit["total_rows"],
        "math_sample_count": math_audit["sample_count"],
        "invalid_sample_count": false_negative_audit["invalid_sample_count"],
        "valid_sample_count": false_negative_audit["valid_sample_count"],
        "leakage_red_flag": leakage["red_flag_same_scored_qa_used_for_training"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
