#!/usr/bin/env python3
"""Production offline full-vocabulary confusion mining.

Uses the frozen B1 adapter and the same ``score_candidates`` mean-logprob as
listed InfoNCE. Candidate packing is unchanged; this script adds resume,
QA-level sharding, metadata, and optional batch-size benchmarks.

``--limit 0`` scores every matchable Stage-2 relation QA. This script does not
sample hard negatives and does not train.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE  # noqa: E402
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.offline_mining import (  # noqa: E402
    SCORING_VERSION,
    atomic_append_qa_block,
    atomic_write_jsonl,
    build_metadata,
    compare_score_rows,
    complete_qa_ids_from_scores,
    drop_incomplete_last_qa,
    git_commit_hash,
    load_jsonl,
    shard_items,
)
from hard_negative_grip.offline_scoring import (  # noqa: E402
    build_qa_score_rows,
    parse_matchable_relation_qa,
    summarize_run,
    validate_candidate_rows,
)
from hard_negative_grip.scoring import score_candidates  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    known_pair_relations,
    load_json_payload,
    train_relation_alias_index,
)
from models.utils import get_hf_llm_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--b1_adapter", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Matchable relation QA to score. 0 means the full matchable set.",
    )
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--filter_splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=["train", "valid", "test"],
        help="KG splits used for false-negative filtering.",
    )
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip qa_ids that already have a complete vocabulary block.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time candidate_batch_size in {1,8,16,32} on a few QA and exit.",
    )
    parser.add_argument("--benchmark_qa", type=int, default=2)
    parser.add_argument(
        "--compare_reference",
        type=Path,
        default=None,
        help="Prompt-3 candidate_scores.jsonl for a random 20-QA regression.",
    )
    parser.add_argument("--compare_count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_b1(args: argparse.Namespace):
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, str(args.b1_adapter), is_trainable=False)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def score_relations(
    model,
    tokenizer,
    prefix: str,
    relations: list[str],
    batch_size: int,
) -> tuple[dict[str, float], dict[str, int]]:
    device = next(model.parameters()).device
    scores: list[float] = []
    lengths: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(relations), batch_size):
            chunk = relations[start : start + batch_size]
            scored = score_candidates(model, tokenizer, prefix, chunk, device=device)
            scores.extend(float(value) for value in scored["candidate_score"][0])
            lengths.extend(int(value) for value in scored["candidate_token_length"][0])
    return dict(zip(relations, scores)), dict(zip(relations, lengths))


def peak_gpu_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def load_matchable(args: argparse.Namespace, alias_index: dict[str, str]) -> list[dict]:
    payload = load_json_payload(args.task_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{args.task_file} is not a GRIP task file")
    items = []
    for index, text in enumerate(payload["qa_samples"]):
        item = parse_matchable_relation_qa(index, str(text), alias_index)
        if item is not None:
            items.append(item)
    return items


def align_resume_files(scores_path: Path, summary_path: Path, vocab_size: int) -> set[str]:
    drop_incomplete_last_qa(scores_path, vocab_size=vocab_size)
    complete_scores = complete_qa_ids_from_scores(scores_path, vocab_size=vocab_size)
    summaries = load_jsonl(summary_path)
    summary_ids = {str(row.get("qa_id") or "") for row in summaries}
    keep = complete_scores & summary_ids
    if set(summary_ids) != keep:
        kept_summaries = [row for row in summaries if str(row.get("qa_id")) in keep]
        atomic_write_jsonl(summary_path, kept_summaries)
    if complete_scores != keep:
        score_rows = [
            row for row in load_jsonl(scores_path) if str(row.get("qa_id")) in keep
        ]
        atomic_write_jsonl(scores_path, score_rows)
    return keep


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark(args, model, tokenizer, items, relation_order) -> dict:
    sizes = [1, 8, 16, 32]
    take = items[: max(1, args.benchmark_qa)]
    rows = []
    for batch_size in sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        oom = False
        try:
            for item in take:
                score_relations(
                    model,
                    tokenizer,
                    assistant_answer_prefix(item["text"]),
                    relation_order,
                    batch_size,
                )
        except torch.cuda.OutOfMemoryError:
            oom = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elapsed = max(time.time() - started, 1e-6)
        n_qa = 0 if oom else len(take)
        n_cand = n_qa * len(relation_order)
        peak_bytes = None if oom else peak_gpu_bytes()
        peak_gib = None if peak_bytes is None else peak_bytes / (1024 ** 3)
        rows.append(
            {
                "candidate_batch_size": batch_size,
                "oom": oom,
                "qa_count": n_qa,
                "elapsed_sec": elapsed,
                "qa_per_sec": (n_qa / elapsed) if n_qa else 0.0,
                "candidate_per_sec": (n_cand / elapsed) if n_cand else 0.0,
                "peak_gpu_bytes": peak_bytes,
                "peak_gpu_gib": peak_gib,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    usable = [row for row in rows if not row["oom"]]
    default = 8
    if usable:
        # Prefer a stable size under ~80% of a 24GiB card, not the absolute max.
        safe = [
            row
            for row in usable
            if row["peak_gpu_gib"] is None or row["peak_gpu_gib"] < 20.0
        ]
        pool = safe or usable
        best_rate = max(float(row["candidate_per_sec"]) for row in pool)
        near = [
            row
            for row in pool
            if float(row["candidate_per_sec"]) >= 0.85 * best_rate
        ]
        chosen = min(near, key=lambda row: int(row["candidate_batch_size"]))
        default = int(chosen["candidate_batch_size"])
    return {"rows": rows, "recommended_candidate_batch_size": default}


def score_item(
    *,
    item: dict,
    model,
    tokenizer,
    relation_order: list[str],
    known_relations: dict,
    args: argparse.Namespace,
    checkpoint: str,
) -> tuple[list[dict], dict]:
    prefix = assistant_answer_prefix(item["text"])
    scores, lengths = score_relations(
        model,
        tokenizer,
        prefix,
        relation_order,
        args.candidate_batch_size,
    )
    pair = item["entity_pair"]
    known = known_relations.get(pair, set()) if pair else set()
    return build_qa_score_rows(
        qa_id=item["qa_id"],
        question=item["question"],
        head_entity=item["head_entity"],
        tail_entity=item["tail_entity"],
        gold_relation=item["gold_relation"],
        matched_train_relation=item["matched_train_relation"],
        relation_order=relation_order,
        scores=scores,
        token_lengths=lengths,
        known_relations=known,
        temperature=args.temperature,
        model_checkpoint=checkpoint,
        split=args.split,
    )


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    relation_order = load_train_relation_order(args.raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_relations = known_pair_relations(args.raw_dir, splits=tuple(args.filter_splits))
    items = load_matchable(args, alias_index)
    items = shard_items(items, num_shards=args.num_shards, shard_id=args.shard_id)
    if args.compare_reference is not None:
        reference_rows = load_jsonl(args.compare_reference)
        reference_ids = sorted({str(row["qa_id"]) for row in reference_rows})
        rng = random.Random(args.seed)
        sample_ids = set(rng.sample(reference_ids, k=min(args.compare_count, len(reference_ids))))
        items = [item for item in items if item["qa_id"] in sample_ids]
    elif args.limit:
        items = items[: args.limit]

    output_dir = args.output_dir
    scores_path = output_dir / "candidate_scores.jsonl"
    summary_path = output_dir / "qa_summary.jsonl"
    stats_path = output_dir / "run_stats.json"
    meta_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = str(args.b1_adapter)
    dtype_name = str(TORCH_DTYPE["bfloat16"]).replace("torch.", "")
    model, tokenizer = load_b1(args)
    tokenizer_name = type(tokenizer).__name__

    if args.benchmark:
        result = run_benchmark(args, model, tokenizer, items, relation_order)
        write_metadata(
            output_dir / "benchmark.json",
            result,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    completed: set[str] = set()
    if args.resume:
        completed = align_resume_files(scores_path, summary_path, len(relation_order))
    else:
        for path in (scores_path, summary_path):
            if path.exists():
                path.unlink()

    metadata = build_metadata(
        checkpoint=checkpoint,
        tokenizer=tokenizer_name,
        dataset=str(args.task_file),
        split=args.split,
        relation_vocab_size=len(relation_order),
        number_of_qa=len(items),
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        candidate_batch_size=args.candidate_batch_size,
        dtype=dtype_name,
        temperature=args.temperature,
        filter_splits=list(args.filter_splits),
        git_commit=git_commit_hash(HNG.parents[1]),
    )
    write_metadata(meta_path, metadata)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    processed = 0
    all_summaries: list[dict] = []
    all_rows: list[dict] = []
    pending = [item for item in items if item["qa_id"] not in completed]
    metadata["number_of_qa"] = len(completed) + len(pending)
    write_metadata(meta_path, metadata)
    for item in pending:
        rows, summary = score_item(
            item=item,
            model=model,
            tokenizer=tokenizer,
            relation_order=relation_order,
            known_relations=known_relations,
            args=args,
            checkpoint=checkpoint,
        )
        atomic_append_qa_block(scores_path, summary_path, rows, summary)
        all_rows.extend(rows)
        all_summaries.append(summary)
        processed += 1
        if args.progress_every > 0 and (
            processed == 1 or processed % args.progress_every == 0 or processed == len(pending)
        ):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[offline_score] shard={args.shard_id}/{args.num_shards} "
                f"{processed}/{len(pending)} qa/s={processed / elapsed:.3f} last={item['qa_id']}",
                flush=True,
            )

    elapsed = time.time() - started
    if args.resume and completed:
        # Validate only the newly written block plus the fact that resume skipped
        # already-complete ids. Full-file checks happen in merge.
        pass
    failures = validate_candidate_rows(
        all_rows,
        relation_order=relation_order,
        known_relations=known_relations,
        temperature=args.temperature,
    )
    if failures:
        raise ValueError("offline scoring validation failed:\n" + "\n".join(failures[:20]))

    stats = summarize_run(
        all_summaries,
        elapsed_sec=elapsed,
        peak_gpu_bytes=peak_gpu_bytes(),
        vocab_size=len(relation_order),
        skipped=0,
    )
    stats.update(
        {
            "task_file": str(args.task_file),
            "raw_dir": str(args.raw_dir),
            "b1_adapter": checkpoint,
            "limit": args.limit,
            "temperature": args.temperature,
            "candidate_batch_size": args.candidate_batch_size,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "resumed_complete_qa": len(completed),
            "candidate_scores": str(scores_path),
            "qa_summary": str(summary_path),
            "scoring_version": SCORING_VERSION,
            "checks_passed": True,
        }
    )
    if args.compare_reference is not None:
        reference = [
            row
            for row in load_jsonl(args.compare_reference)
            if str(row["qa_id"]) in {item["qa_id"] for item in items}
        ]
        produced = load_jsonl(scores_path)
        produced = [row for row in produced if str(row["qa_id"]) in {item["qa_id"] for item in items}]
        compare_failures = compare_score_rows(reference, produced, atol=1e-5)
        stats["compare_reference"] = str(args.compare_reference)
        stats["compare_qa"] = sorted({item["qa_id"] for item in items})
        stats["compare_failures"] = compare_failures
        stats["compare_passed"] = not compare_failures
        if compare_failures:
            raise ValueError(
                "production scores drifted from the Prompt-3 reference:\n"
                + "\n".join(compare_failures[:20])
            )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["number_of_qa"] = len(completed) + processed
    write_metadata(meta_path, metadata)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
