#!/usr/bin/env python3
"""Small-scale offline full-vocabulary scoring for Stage-2 relation QA.

Uses the frozen B1 adapter and the same ``score_candidates`` mean-logprob as
listed InfoNCE. Default limit is 100 relation QA items; raise ``--limit`` up to
1000. This script does not sample hard negatives and does not train.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--limit", type=int, default=100, help="Matchable relation QA to score.")
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip qa_ids already present in candidate_scores.jsonl.",
    )
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
    with torch.no_grad():
        for start in range(0, len(relations), batch_size):
            chunk = relations[start : start + batch_size]
            scored = score_candidates(model, tokenizer, prefix, chunk, device=device)
            scores.extend(float(value) for value in scored["candidate_score"][0])
            lengths.extend(int(value) for value in scored["candidate_token_length"][0])
    return dict(zip(relations, scores)), dict(zip(relations, lengths))


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qa_id = str(row.get("qa_id") or "")
            if not qa_id:
                raise ValueError(f"{path}:{line_number}: missing qa_id")
            completed.add(qa_id)
    return completed


def peak_gpu_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive for this small-scale dump")
    if args.limit > 1000:
        raise ValueError("this script is capped at 1000 QA; do not dump the full set here")
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    payload = load_json_payload(args.task_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{args.task_file} is not a GRIP task file")
    relation_order = load_train_relation_order(args.raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_relations = known_pair_relations(args.raw_dir)

    output_dir = args.output_dir
    scores_path = output_dir / "candidate_scores.jsonl"
    summary_path = output_dir / "qa_summary.jsonl"
    stats_path = output_dir / "run_stats.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed_ids(scores_path) if args.resume else set()
    mode = "a" if args.resume and scores_path.is_file() else "w"
    if mode == "w":
        for path in (scores_path, summary_path):
            if path.exists():
                path.unlink()

    model, tokenizer = load_b1(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    processed = 0
    skipped = 0
    all_summaries: list[dict] = []
    all_rows: list[dict] = []
    checkpoint = str(args.b1_adapter)

    with scores_path.open(mode, encoding="utf-8") as score_stream, summary_path.open(
        mode, encoding="utf-8"
    ) as summary_stream:
        for index, text in enumerate(payload["qa_samples"]):
            if processed >= args.limit:
                break
            item = parse_matchable_relation_qa(index, str(text), alias_index)
            if item is None:
                skipped += 1
                continue
            if item["qa_id"] in completed:
                continue
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
            rows, summary = build_qa_score_rows(
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
            for row in rows:
                score_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary_stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            score_stream.flush()
            summary_stream.flush()
            all_rows.extend(rows)
            all_summaries.append(summary)
            processed += 1
            if args.progress_every > 0 and (
                processed == 1 or processed % args.progress_every == 0
            ):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[offline_score] {processed}/{args.limit} "
                    f"qa/s={processed / elapsed:.3f} last={item['qa_id']}",
                    flush=True,
                )

    elapsed = time.time() - started

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
        skipped=skipped,
    )
    stats.update(
        {
            "task_file": str(args.task_file),
            "raw_dir": str(args.raw_dir),
            "b1_adapter": checkpoint,
            "limit": args.limit,
            "temperature": args.temperature,
            "candidate_batch_size": args.candidate_batch_size,
            "candidate_scores": str(scores_path),
            "qa_summary": str(summary_path),
            "checks_passed": True,
        }
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
