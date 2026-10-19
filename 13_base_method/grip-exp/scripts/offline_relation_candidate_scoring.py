"""Production offline relation scoring with resume and QA-level sharding."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from grip.offline_relation_scoring import (
    build_pair_relation_index,
    compute_candidate_diagnostics,
    resolve_pair_metadata,
    validate_candidate_diagnostics,
)
from grip.training.candidate_scorer import score_candidate_records
from models.utils import get_hf_llm_tokenizer


SCORING_VERSION = "offline-relation-scoring-v2-batched-resumable"
CHECKPOINT_FILE = "qa_records.jsonl"
REFERENCE_CANDIDATE_BATCH_SIZE = 32


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected one graph record in {path}, found {len(payload)}")
        return payload[0]
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_checkpoint(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_checkpoint(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    payloads: dict[int, dict] = {}
    valid_lines: list[dict] = []
    had_partial_tail = False
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                had_partial_tail = True
                break
            global_index = payload.get("global_index")
            if global_index is None or global_index in payloads:
                raise ValueError(f"Invalid or duplicate checkpoint global_index in {path}")
            if not {"qa_id", "candidate_rows", "qa_summary"}.issubset(payload):
                raise ValueError(f"Incomplete checkpoint record in {path}")
            payloads[int(global_index)] = payload
            valid_lines.append(payload)
    if had_partial_tail:
        _atomic_write_jsonl(path, valid_lines)
    return payloads


def _resolve_output_dir(base: str | Path, num_shards: int, shard_id: int) -> Path:
    base_path = Path(base)
    if num_shards == 1:
        return base_path
    return base_path / f"shard_{shard_id}_of_{num_shards}"


def _write_final_outputs(output_dir: Path, payloads: list[dict]) -> tuple[dict, dict]:
    ordered = sorted(payloads, key=lambda payload: payload["global_index"])
    candidate_rows = (
        row
        for payload in ordered
        for row in sorted(payload["candidate_rows"], key=lambda item: item["candidate_rank_all"])
    )
    summary_rows = (payload["qa_summary"] for payload in ordered)
    _atomic_write_jsonl(output_dir / "candidate_scores.jsonl", candidate_rows)
    _atomic_write_jsonl(output_dir / "qa_summary.jsonl", summary_rows)

    failures = []
    # A check must pass for every QA, not only for the aggregate output.
    check_names = [
        "one_gold_per_qa",
        "gold_score_consistent",
        "valid_negative_excludes_gold",
        "false_negative_marked",
        "negative_mass_sums_to_one",
        "score_gap_correct",
        "pairwise_confusion_bounded",
        "candidate_ranks_unique",
        "negative_ranks_unique",
    ]
    aggregate_checks = {name: True for name in check_names}
    for payload in ordered:
        result = validate_candidate_diagnostics({
            payload["qa_id"]: payload["candidate_rows"],
        })
        failures.extend(result["failures"])
        for name in check_names:
            aggregate_checks[name] = aggregate_checks[name] and result["checks"][name]
    validation = {
        "all_passed": all(aggregate_checks.values()) and not failures,
        "checks": aggregate_checks,
        "failures": failures,
        "qa_count": len(ordered),
    }
    summary_values = [payload["qa_summary"] for payload in ordered]
    metrics = {
        "mean_gold_rank": sum(row["gold_rank"] for row in summary_values) / max(1, len(summary_values)),
        "qa_with_negative_score_above_gold": sum(
            row["number_of_negatives_above_gold"] > 0 for row in summary_values
        ),
        "qa_with_negative_score_above_gold_rate": sum(
            row["number_of_negatives_above_gold"] > 0 for row in summary_values
        ) / max(1, len(summary_values)),
    }
    return validation, metrics


def _identity(
    args: argparse.Namespace,
    output_dir: Path,
    relation_vocab_size: int,
    relation_id_count: int,
    assigned_global_indices: list[int],
    model_checkpoint: str,
) -> dict:
    return {
        "scoring_version": SCORING_VERSION,
        "git_commit": _git_commit(),
        "model_checkpoint": model_checkpoint,
        "candidate_file": str(Path(args.candidate_file).expanduser().resolve()),
        "processed_file": str(Path(args.processed_file).expanduser().resolve()),
        "relation_text_file": str(Path(args.relation_text_file).expanduser().resolve()),
        "split": args.split,
        "relation_vocab_size": relation_vocab_size,
        "relation_id_count": relation_id_count,
        "requested_num_qa": args.num_qa,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "candidate_batch_size": args.candidate_batch_size,
        "reference_candidate_batch_size": REFERENCE_CANDIDATE_BATCH_SIZE,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "assigned_global_indices": assigned_global_indices,
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--processed-file", required=True)
    parser.add_argument("--relation-text-file", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/confusion_debug/production")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="qwen-7b")
    parser.add_argument("--num-qa", type=int, default=100,
                        help="Number of deterministic sampled QA; 0 processes all relation QA.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--candidate-batch-size", "--candidate_batch_size",
        dest="candidate_batch_size", type=int, default=32,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qa < 0:
        raise ValueError("num-qa must be non-negative; use 0 for all relation QA")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate-batch-size must be positive")
    if args.candidate_batch_size != REFERENCE_CANDIDATE_BATCH_SIZE:
        raise ValueError(
            "Production scoring is fixed at candidate_batch_size=32 to preserve "
            "Prompt 3 numerical results; use benchmark_offline_relation_scoring.py "
            "for other batch sizes."
        )
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

    relation_text = _load_json(args.relation_text_file)
    relation_labels = sorted(set(relation_text.values()))
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    for relation_id, label in relation_text.items():
        label_to_ids[label].append(relation_id)
    relation_collisions = {
        label: sorted(ids)
        for label, ids in label_to_ids.items()
        if len(ids) > 1
    }

    candidate_payload = _load_json(args.candidate_file)
    candidate_records = [
        record for record in candidate_payload.get("records", [])
        if record is not None
    ]
    if args.num_qa > len(candidate_records):
        raise ValueError(
            f"Candidate file contains {len(candidate_records)} relation QA, "
            f"but {args.num_qa} were requested."
        )
    for record in candidate_records:
        if record.get("positive") not in relation_labels:
            raise ValueError(f"Gold relation is absent from vocabulary: {record.get('positive')}")

    graph_record = _load_json(args.processed_file)
    graph_edges = graph_record["graph"]["edge_list"]
    pair_relations = build_pair_relation_index(graph_edges)

    if args.num_qa == 0:
        selected_indices = list(range(len(candidate_records)))
    else:
        selected_indices = random.Random(args.seed).sample(
            range(len(candidate_records)), args.num_qa,
        )
    assigned_indices = [
        index for index in selected_indices
        if index % args.num_shards == args.shard_id
    ]
    selected_by_index = {index: candidate_records[index] for index in assigned_indices}
    output_dir = _resolve_output_dir(args.output_dir, args.num_shards, args.shard_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILE
    metadata_path = output_dir / "metadata.json"
    model_checkpoint = str(Path(args.adapter_dir).expanduser().resolve())
    identity = _identity(
        args, output_dir, len(relation_labels), len(relation_text),
        sorted(assigned_indices), model_checkpoint,
    )

    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError("Resume metadata does not match the current scoring configuration")
        checkpoint = _read_checkpoint(checkpoint_path)
    else:
        existing = [
            output_dir / name for name in (
                CHECKPOINT_FILE, "metadata.json", "run_summary.json",
                "candidate_scores.jsonl", "qa_summary.jsonl",
            ) if (output_dir / name).exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Output already exists; use --resume or --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
        if args.overwrite:
            for path in existing:
                path.unlink()
        checkpoint = {}
        metadata = {
            "schema_version": 2,
            "status": "initialized",
            "created_at": _utc_now(),
            "identity": identity,
        }

    unexpected = sorted(set(checkpoint) - set(assigned_indices))
    if unexpected:
        raise ValueError(f"Checkpoint contains QA outside this shard: {unexpected[:5]}")
    metadata["completed_qa_count"] = len(checkpoint)
    metadata["assigned_qa_count"] = len(assigned_indices)
    metadata["shard_id"] = args.shard_id
    metadata["num_shards"] = args.num_shards
    metadata["relation_vocab_size"] = len(relation_labels)
    metadata["checkpoint_file"] = str(checkpoint_path.resolve())
    metadata["status"] = "running"
    metadata["updated_at"] = _utc_now()
    _atomic_write_json(metadata_path, metadata)

    remaining = [index for index in assigned_indices if index not in checkpoint]
    current_scoring_seconds = 0.0
    current_forward_count = 0
    current_peak_allocated = 0
    current_peak_reserved = 0
    if remaining:
        load_started = time.perf_counter()
        model, tokenizer = get_hf_llm_tokenizer(
            model_name=HF_DECODER_ONLY_LLMS[args.model_name],
            load_dir=model_checkpoint,
            model_source="local",
            model_cache_dir=args.model_cache_dir,
            local_files_only=True,
            tokenize_max_length=args.max_length,
            dtype=TORCH_DTYPE["bfloat16"],
            peft=False,
            device_map="auto",
        )
        model.eval()
        metadata["tokenizer"] = {
            "class": tokenizer.__class__.__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", model_checkpoint),
        }
        metadata["dtype"] = "bfloat16"
        metadata["checkpoint_path"] = model_checkpoint
        metadata["status"] = "running"
        metadata["updated_at"] = _utc_now()
        _atomic_write_json(metadata_path, metadata)
        model_load_seconds = time.perf_counter() - load_started
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        scoring_started = time.perf_counter()
        with torch.inference_mode():
            for global_index in remaining:
                record = selected_by_index[global_index]
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
                token_lengths = score_result.token_lengths[0].detach().cpu().tolist()
                current_forward_count += (
                    len(relation_labels) + args.candidate_batch_size - 1
                ) // args.candidate_batch_size
                pair_metadata = resolve_pair_metadata(record, graph_edges, pair_relations)
                diagnostic_rows, summary = compute_candidate_diagnostics(
                    relation_labels,
                    scores,
                    record["positive"],
                    pair_metadata["all_true_relations"],
                    temperature=args.temperature,
                )
                qa_id = record.get("task_index", record.get("id"))
                common = {
                    "global_index": global_index,
                    "qa_id": qa_id,
                    "candidate_record_id": record.get("id"),
                    "question": record.get("question"),
                    "head_entity": pair_metadata["head_entity"],
                    "tail_entity": pair_metadata["tail_entity"],
                    "metadata_source": pair_metadata["metadata_source"],
                    "false_negative_filter_available": pair_metadata["false_negative_filter_available"],
                    "all_true_relations": pair_metadata["all_true_relations"],
                    "gold_relation": record["positive"],
                    "model_checkpoint": model_checkpoint,
                    "split": args.split,
                }
                rows = [
                    {
                        **common,
                        **row,
                        "candidate_token_length": int(token_length),
                    }
                    for row, token_length in zip(diagnostic_rows, token_lengths)
                ]
                checkpoint_payload = {
                    "global_index": global_index,
                    "qa_id": qa_id,
                    "candidate_rows": rows,
                    "qa_summary": {**common, **summary},
                }
                _append_checkpoint(checkpoint_path, checkpoint_payload)
                checkpoint[global_index] = checkpoint_payload
                metadata["completed_qa_count"] = len(checkpoint)
                metadata["updated_at"] = _utc_now()
                _atomic_write_json(metadata_path, metadata)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_peak_allocated = torch.cuda.max_memory_allocated()
            current_peak_reserved = torch.cuda.max_memory_reserved()
        current_scoring_seconds = time.perf_counter() - scoring_started
    else:
        model_load_seconds = 0.0

    payloads = list(checkpoint.values())
    validation, metrics = _write_final_outputs(output_dir, payloads)
    previous_performance = metadata.get("performance", {})
    total_scoring_seconds = float(previous_performance.get("scoring_seconds", 0.0)) + current_scoring_seconds
    total_forward_count = int(previous_performance.get("candidate_forward_count", 0)) + current_forward_count
    peak_allocated = max(int(previous_performance.get("peak_gpu_memory_allocated_bytes", 0)), current_peak_allocated)
    peak_reserved = max(int(previous_performance.get("peak_gpu_memory_reserved_bytes", 0)), current_peak_reserved)
    summary_values = [payload["qa_summary"] for payload in payloads]
    run_summary = {
        "schema_version": 2,
        "scoring_version": SCORING_VERSION,
        "identity": identity,
        "inputs": {
            "candidate_file": identity["candidate_file"],
            "processed_file": identity["processed_file"],
            "relation_text_file": identity["relation_text_file"],
            "model_checkpoint": model_checkpoint,
            "split": args.split,
        },
        "configuration": {
            "num_qa": args.num_qa,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_length": args.max_length,
            "candidate_batch_size": args.candidate_batch_size,
            "reference_candidate_batch_size": REFERENCE_CANDIDATE_BATCH_SIZE,
            "score_clip": None,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
        "dataset": {
            "selected_qa_count": len(selected_indices),
            "assigned_qa_count": len(assigned_indices),
            "completed_qa_count": len(payloads),
            "total_relation_qa_available": len(candidate_records),
            "total_candidate_count": len(payloads) * len(relation_labels),
            "average_candidates_per_qa": float(len(relation_labels)),
            "graph_edge_count": len(graph_edges),
            "relation_id_count": len(relation_text),
            "relation_vocab_size": len(relation_labels),
            "relation_label_collisions": relation_collisions,
        },
        "performance": {
            "model_load_seconds": model_load_seconds,
            "scoring_seconds": total_scoring_seconds,
            "qa_per_second": len(payloads) / max(total_scoring_seconds, 1e-9),
            "candidate_per_second": len(payloads) * len(relation_labels) / max(total_scoring_seconds, 1e-9),
            "candidate_forward_count": total_forward_count,
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "peak_gpu_memory_allocated_gib": peak_allocated / 2**30,
            "peak_gpu_memory_reserved_gib": peak_reserved / 2**30,
        },
        "metrics": metrics,
        "validation": validation,
        "outputs": {
            "candidate_scores": str((output_dir / "candidate_scores.jsonl").resolve()),
            "qa_summary": str((output_dir / "qa_summary.jsonl").resolve()),
            "qa_checkpoint": str(checkpoint_path.resolve()),
        },
    }
    metadata.update({
        "status": "complete" if validation["all_passed"] and len(payloads) == len(assigned_indices) else "incomplete",
        "updated_at": _utc_now(),
        "number_of_qa": len(payloads),
        "checkpoint_path": model_checkpoint,
        "dataset": {
            "candidate_file": identity["candidate_file"],
            "processed_file": identity["processed_file"],
            "relation_text_file": identity["relation_text_file"],
        },
        "split": args.split,
        "relation_vocab_size": len(relation_labels),
        "candidate_batch_size": args.candidate_batch_size,
        "reference_candidate_batch_size": REFERENCE_CANDIDATE_BATCH_SIZE,
        "temperature": args.temperature,
        "dtype": metadata.get("dtype", "bfloat16"),
        "scoring_version": SCORING_VERSION,
        "performance": run_summary["performance"],
        "validation": validation,
    })
    _atomic_write_json(metadata_path, metadata)
    _atomic_write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary["performance"], ensure_ascii=False))
    print(json.dumps(run_summary["metrics"], ensure_ascii=False))
    print(json.dumps({
        "status": metadata["status"],
        "validation": validation,
        "output_dir": str(output_dir.resolve()),
    }, ensure_ascii=False))
    if metadata["status"] != "complete":
        raise SystemExit("Offline candidate output is incomplete or failed validation")


if __name__ == "__main__":
    main()
