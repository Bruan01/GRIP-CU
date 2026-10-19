"""Compare production offline scoring with the Prompt 3 JSONL output."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from grip.offline_relation_scoring import (
    build_pair_relation_index,
    compute_candidate_diagnostics,
    resolve_pair_metadata,
)
from grip.training.candidate_scorer import score_candidate_records
from models.utils import get_hf_llm_tokenizer


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected one record in {path}")
        return payload[0]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-candidate-scores", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--processed-file", required=True)
    parser.add_argument("--relation-text-file", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--model-name", default="qwen-7b")
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--output-file", default="outputs/confusion_debug/regression_20.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qa <= 0 or args.temperature <= 0 or args.candidate_batch_size <= 0:
        raise ValueError("num-qa, temperature, and candidate-batch-size must be positive")

    reference_by_qa = defaultdict(list)
    with Path(args.reference_candidate_scores).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            reference_by_qa[row["qa_id"]].append(row)
    reference_ids = sorted(reference_by_qa)
    if args.num_qa > len(reference_ids):
        raise ValueError("num-qa exceeds reference QA count")
    selected_ids = random.Random(args.seed).sample(reference_ids, args.num_qa)

    relation_text = _load_json(args.relation_text_file)
    relation_labels = sorted(set(relation_text.values()))
    candidate_payload = _load_json(args.candidate_file)
    record_by_qa = {
        record.get("task_index", record.get("id")): record
        for record in candidate_payload.get("records", [])
        if record is not None
    }
    graph_record = _load_json(args.processed_file)
    graph_edges = graph_record["graph"]["edge_list"]
    pair_relations = build_pair_relation_index(graph_edges)

    model, tokenizer = get_hf_llm_tokenizer(
        model_name=HF_DECODER_ONLY_LLMS[args.model_name],
        load_dir=str(Path(args.adapter_dir).expanduser().resolve()),
        model_source="local",
        model_cache_dir=args.model_cache_dir,
        local_files_only=True,
        tokenize_max_length=args.max_length,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map="auto",
    )
    model.eval()
    metric_errors = defaultdict(float)
    metric_error_sums = defaultdict(float)
    metric_error_counts = defaultdict(int)
    mismatch_count = 0
    checked_candidates = 0
    run_fingerprints = {}
    with torch.inference_mode():
        for qa_id in selected_ids:
            record = record_by_qa[qa_id]
            scoring_record = dict(record)
            scoring_record["candidates"] = list(relation_labels)
            score_result = score_candidate_records(
                model,
                tokenizer,
                [scoring_record],
                max_length=args.max_length,
                score_clip=None,
                candidate_batch_size=args.candidate_batch_size,
            )
            scores = score_result.scores[0].detach().float().cpu().tolist()
            pair_metadata = resolve_pair_metadata(record, graph_edges, pair_relations)
            current_rows, _ = compute_candidate_diagnostics(
                relation_labels,
                scores,
                record["positive"],
                pair_metadata["all_true_relations"],
                temperature=args.temperature,
            )
            current_by_relation = {
                row["candidate_relation"]: row for row in current_rows
            }
            reference_rows = {
                row["candidate_relation"]: row for row in reference_by_qa[qa_id]
            }
            if set(current_by_relation) != set(reference_rows):
                raise ValueError(f"Candidate vocabulary mismatch for QA {qa_id}")
            for relation in relation_labels:
                current = current_by_relation[relation]
                reference = reference_rows[relation]
                for field in (
                    "candidate_score",
                    "score_gap",
                    "pairwise_confusion",
                    "negative_mass",
                ):
                    actual = current[field]
                    expected = reference[field]
                    if actual is None or expected is None:
                        error = 0.0 if actual is expected else float("inf")
                    else:
                        error = abs(float(actual) - float(expected))
                    metric_errors[field] = max(metric_errors[field], error)
                    if math.isfinite(error):
                        metric_error_sums[field] += error
                        metric_error_counts[field] += 1
                    if error > 1e-5:
                        mismatch_count += 1
                for field in ("candidate_rank_all", "negative_rank"):
                    if current[field] != reference[field]:
                        mismatch_count += 1
                checked_candidates += 1
            fingerprint_rows = [
                {
                    "candidate_relation": row["candidate_relation"],
                    "candidate_score": row["candidate_score"],
                    "candidate_rank_all": row["candidate_rank_all"],
                    "negative_rank": row["negative_rank"],
                }
                for row in sorted(current_rows, key=lambda row: row["candidate_rank_all"])
            ]
            run_fingerprints[str(qa_id)] = hashlib.sha256(
                json.dumps(fingerprint_rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()

    mean_abs_error = {
        field: metric_error_sums[field] / max(1, metric_error_counts[field])
        for field in metric_error_sums
    }
    run_digest = hashlib.sha256(
        json.dumps(run_fingerprints, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    report = {
        "reference_file": str(Path(args.reference_candidate_scores).resolve()),
        "selected_qa_ids": selected_ids,
        "qa_count": len(selected_ids),
        "candidate_count": checked_candidates,
        "max_abs_error": dict(metric_errors),
        "mean_abs_error": mean_abs_error,
        "mismatch_count": mismatch_count,
        "tolerance": 1e-5,
        "pass": mismatch_count == 0,
        "run_fingerprints": run_fingerprints,
        "run_digest": run_digest,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit("Production scoring regression check failed")


if __name__ == "__main__":
    main()
