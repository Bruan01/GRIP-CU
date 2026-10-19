"""Benchmark candidate batch sizes without changing the scoring definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from grip.training.candidate_scorer import score_candidate_records
from models.utils import get_hf_llm_tokenizer


REFERENCE_CANDIDATE_BATCH_SIZE = 32


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
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--relation-text-file", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--model-name", default="qwen-7b")
    parser.add_argument("--num-qa", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--output-file", default="outputs/confusion_debug/batch_benchmark.json")
    return parser.parse_args()


def _is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def main() -> None:
    args = parse_args()
    if args.num_qa <= 0:
        raise ValueError("num-qa must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    relation_text = _load_json(args.relation_text_file)
    relation_labels = sorted(set(relation_text.values()))
    candidate_payload = _load_json(args.candidate_file)
    relation_records = [
        record for record in candidate_payload.get("records", [])
        if record is not None
    ]
    if args.num_qa > len(relation_records):
        raise ValueError("num-qa exceeds available relation QA")
    selected = [
        relation_records[index]
        for index in random.Random(args.seed).sample(range(len(relation_records)), args.num_qa)
    ]

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
    results = []
    score_snapshots = {}
    for batch_size in args.batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            scores = []
            with torch.inference_mode():
                for record in selected:
                    scoring_record = dict(record)
                    scoring_record["candidates"] = list(relation_labels)
                    result = score_candidate_records(
                        model,
                        tokenizer,
                        [scoring_record],
                        max_length=args.max_length,
                        score_clip=None,
                        candidate_batch_size=batch_size,
                    )
                    scores.append(result.scores[0].detach().float().cpu().tolist())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            else:
                peak_allocated = peak_reserved = 0
            elapsed = time.perf_counter() - started
            score_snapshots[batch_size] = scores
            results.append({
                "candidate_batch_size": batch_size,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "qa_per_second": len(selected) / max(elapsed, 1e-9),
                "candidate_per_second": len(selected) * len(relation_labels) / max(elapsed, 1e-9),
                "peak_gpu_memory_allocated_bytes": peak_allocated,
                "peak_gpu_memory_reserved_bytes": peak_reserved,
                "peak_gpu_memory_allocated_gib": peak_allocated / 2**30,
                "peak_gpu_memory_reserved_gib": peak_reserved / 2**30,
            })
        except RuntimeError as error:
            if not _is_oom(error):
                raise
            if torch.cuda.is_available():
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            else:
                peak_allocated = peak_reserved = 0
            results.append({
                "candidate_batch_size": batch_size,
                "status": "oom",
                "error": str(error),
                "peak_gpu_memory_allocated_bytes": peak_allocated,
                "peak_gpu_memory_reserved_bytes": peak_reserved,
                "peak_gpu_memory_allocated_gib": peak_allocated / 2**30,
                "peak_gpu_memory_reserved_gib": peak_reserved / 2**30,
            })

    reference_scores = score_snapshots.get(REFERENCE_CANDIDATE_BATCH_SIZE)
    for result in results:
        scores = score_snapshots.get(result["candidate_batch_size"])
        if scores is None or reference_scores is None:
            result["max_abs_error_vs_reference_batch"] = None
            result["rank_mismatch_count_vs_reference_batch"] = None
            continue
        result["max_abs_error_vs_reference_batch"] = max(
            abs(actual - expected)
            for actual_rows, expected_rows in zip(scores, reference_scores)
            for actual, expected in zip(actual_rows, expected_rows)
        )
        result["rank_mismatch_count_vs_reference_batch"] = sum(
            sorted(range(len(actual_rows)), key=lambda index: (-actual_rows[index], index))
            != sorted(range(len(expected_rows)), key=lambda index: (-expected_rows[index], index))
            for actual_rows, expected_rows in zip(scores, reference_scores)
        )

    report = {
        "model_checkpoint": str(Path(args.adapter_dir).expanduser().resolve()),
        "num_qa": len(selected),
        "relation_vocab_size": len(relation_labels),
        "dtype": "bfloat16",
        "reference_candidate_batch_size": REFERENCE_CANDIDATE_BATCH_SIZE,
        "note": "Production uses batch size 32 for Prompt 3 numerical compatibility; other sizes are diagnostic only.",
        "results": results,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not any(item.get("status") == "ok" and item["candidate_batch_size"] == REFERENCE_CANDIDATE_BATCH_SIZE for item in results):
        raise SystemExit("Reference batch size 32 was not benchmarked successfully")


if __name__ == "__main__":
    main()
