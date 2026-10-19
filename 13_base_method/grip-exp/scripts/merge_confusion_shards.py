"""Validate and merge completed offline confusion-scoring shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.offline_relation_candidate_scoring import (
    REFERENCE_CANDIDATE_BATCH_SIZE,
    SCORING_VERSION,
    _atomic_write_json,
    _write_final_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing shard metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Shard is not complete: {path}")
    return metadata


def _load_checkpoint(path: Path) -> list[dict]:
    checkpoint_path = path / "qa_records.jsonl"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing shard checkpoint: {checkpoint_path}")
    payloads = []
    global_indices = set()
    qa_ids = set()
    with checkpoint_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid checkpoint JSON at {checkpoint_path}:{line_number}") from error
            global_index = payload.get("global_index")
            qa_id = payload.get("qa_id")
            if global_index in global_indices:
                raise ValueError(f"Duplicate global_index {global_index} in {checkpoint_path}")
            if qa_id in qa_ids:
                raise ValueError(f"Duplicate qa_id {qa_id} in {checkpoint_path}")
            if not {"candidate_rows", "qa_summary"}.issubset(payload):
                raise ValueError(f"Incomplete checkpoint record at {checkpoint_path}:{line_number}")
            global_indices.add(global_index)
            qa_ids.add(qa_id)
            payloads.append(payload)
    return payloads


def _identity_without_shard(metadata: dict) -> dict:
    identity = dict(metadata.get("identity", {}))
    for key in ("shard_id", "assigned_global_indices", "output_dir"):
        identity.pop(key, None)
    return identity


def main() -> None:
    args = parse_args()
    shard_paths = [Path(path).expanduser().resolve() for path in args.shard_dir]
    if len(set(shard_paths)) != len(shard_paths):
        raise ValueError("Duplicate shard directory arguments")
    metadata_items = [(path, _load_metadata(path)) for path in shard_paths]
    reference = metadata_items[0][1]
    num_shards = int(reference["num_shards"])
    shard_ids = [int(metadata["shard_id"]) for _, metadata in metadata_items]
    if len(metadata_items) != num_shards:
        raise ValueError(f"Expected {num_shards} shard directories, got {len(metadata_items)}")
    if sorted(shard_ids) != list(range(num_shards)):
        raise ValueError(f"Missing or duplicate shard ids: {sorted(shard_ids)}")

    reference_identity = _identity_without_shard(reference)
    all_payloads = []
    all_global_indices = set()
    all_qa_ids = set()
    assigned_union = set()
    performance = []
    for path, metadata in metadata_items:
        if int(metadata["num_shards"]) != num_shards:
            raise ValueError(f"num_shards mismatch in {path}")
        if _identity_without_shard(metadata) != reference_identity:
            raise ValueError(f"Metadata mismatch in {path}")
        assigned = set(metadata.get("identity", {}).get("assigned_global_indices", []))
        if assigned & assigned_union:
            raise ValueError(f"Overlapping shard assignment in {path}")
        assigned_union.update(assigned)
        payloads = _load_checkpoint(path)
        completed = {payload["global_index"] for payload in payloads}
        if completed != assigned:
            raise ValueError(
                f"Shard {path} is missing completed QA: "
                f"expected {len(assigned)}, found {len(completed)}"
            )
        if int(metadata.get("completed_qa_count", -1)) != len(payloads):
            raise ValueError(f"Completed QA count mismatch in {path}")
        for payload in payloads:
            if payload["global_index"] in all_global_indices:
                raise ValueError(f"Duplicate global_index across shards: {payload['global_index']}")
            if payload["qa_id"] in all_qa_ids:
                raise ValueError(f"Duplicate qa_id across shards: {payload['qa_id']}")
            all_global_indices.add(payload["global_index"])
            all_qa_ids.add(payload["qa_id"])
        all_payloads.extend(payloads)
        performance.append(metadata.get("performance", {}))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / name for name in (
            "candidate_scores.jsonl", "qa_summary.jsonl", "metadata.json", "run_summary.json",
        ) if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Merge output already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    validation, metrics = _write_final_outputs(output_dir, all_payloads)
    relation_vocab_size = int(reference["relation_vocab_size"])
    if int(reference.get("candidate_batch_size", -1)) != REFERENCE_CANDIDATE_BATCH_SIZE:
        raise ValueError(
            "Shard batch size is not the Prompt 3 reference batch size: "
            f"{reference.get('candidate_batch_size')} != {REFERENCE_CANDIDATE_BATCH_SIZE}"
        )
    candidate_count = len(all_payloads) * relation_vocab_size
    scoring_seconds = sum(float(item.get("scoring_seconds", 0.0)) for item in performance)
    forward_count = sum(int(item.get("candidate_forward_count", 0)) for item in performance)
    peak_allocated = max(
        (int(item.get("peak_gpu_memory_allocated_bytes", 0)) for item in performance),
        default=0,
    )
    peak_reserved = max(
        (int(item.get("peak_gpu_memory_reserved_bytes", 0)) for item in performance),
        default=0,
    )
    merged_metadata = {
        "schema_version": 2,
        "status": "complete" if validation["all_passed"] else "invalid",
        "scoring_version": SCORING_VERSION,
        "merge": True,
        "merged_from": [str(path) for path in sorted(shard_paths)],
        "git_commit": reference.get("identity", {}).get("git_commit"),
        "checkpoint_path": reference.get("checkpoint_path", reference.get("identity", {}).get("model_checkpoint")),
        "tokenizer": reference.get("tokenizer"),
        "dataset": reference.get("dataset", {
            "candidate_file": reference.get("identity", {}).get("candidate_file"),
            "processed_file": reference.get("identity", {}).get("processed_file"),
            "relation_text_file": reference.get("identity", {}).get("relation_text_file"),
        }),
        "split": reference.get("split", reference.get("identity", {}).get("split")),
        "relation_vocab_size": relation_vocab_size,
        "number_of_qa": len(all_payloads),
        "shard_id": None,
        "num_shards": num_shards,
        "candidate_batch_size": reference.get("candidate_batch_size", reference.get("identity", {}).get("candidate_batch_size")),
        "reference_candidate_batch_size": REFERENCE_CANDIDATE_BATCH_SIZE,
        "dtype": reference.get("dtype", "bfloat16"),
        "temperature": reference.get("temperature", reference.get("identity", {}).get("temperature")),
        "created_at": reference.get("created_at"),
        "updated_at": reference.get("updated_at"),
        "assigned_global_indices": sorted(assigned_union),
        "completed_qa_count": len(all_payloads),
        "performance": {
            "scoring_seconds": scoring_seconds,
            "qa_per_second": len(all_payloads) / max(scoring_seconds, 1e-9),
            "candidate_per_second": candidate_count / max(scoring_seconds, 1e-9),
            "candidate_forward_count": forward_count,
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "peak_gpu_memory_allocated_gib": peak_allocated / 2**30,
            "peak_gpu_memory_reserved_gib": peak_reserved / 2**30,
        },
        "validation": validation,
        "metrics": metrics,
        "outputs": {
            "candidate_scores": str((output_dir / "candidate_scores.jsonl").resolve()),
            "qa_summary": str((output_dir / "qa_summary.jsonl").resolve()),
        },
    }
    _atomic_write_json(output_dir / "metadata.json", merged_metadata)
    _atomic_write_json(output_dir / "run_summary.json", merged_metadata)
    print(json.dumps({
        "status": merged_metadata["status"],
        "qa_count": len(all_payloads),
        "candidate_count": candidate_count,
        "validation": validation,
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    if merged_metadata["status"] != "complete":
        raise SystemExit("Merged shard validation failed")


if __name__ == "__main__":
    main()
