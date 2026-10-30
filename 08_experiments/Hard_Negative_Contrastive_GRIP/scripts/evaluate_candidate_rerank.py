#!/usr/bin/env python3
"""Resumable closed-set candidate reranking for GRIP relation QA.

The official GRIP NELL23K evaluation prompt already contains ten candidate
relations.  This script scores those exact candidates under a frozen adapter
and selects the highest teacher-forced continuation score.  It is an
evaluation-time decoding experiment only: it does not change the graph,
candidate lists, training data, or model weights.

Every completed question is appended and fsynced to ``predictions.jsonl``;
``--resume`` can therefore continue after an interrupted process.  A small
pilot can be run with ``--start``/``--limit`` before spending time on all
test questions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

HNG = Path(__file__).resolve().parents[1]
REPO_ROOT = HNG.parents[1]
GRIP_EXP = Path(
    os.environ.get(
        "GRIP_BASELINE_DIR", str(REPO_ROOT / "13_base_method" / "grip-exp")
    )
)
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE  # noqa: E402
from evaluation.utils import normalize_answer  # noqa: E402
from grip.relation_cecr import parse_relation_candidates  # noqa: E402
from grip.tasks import gen_eval_task  # noqa: E402
from grip.training.candidate_scorer import score_candidate_prompt  # noqa: E402
from models.utils import get_hf_llm_tokenizer  # noqa: E402
from utils import set_random_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=None,
        help="Optional LoRA adapter. Omit to score with the local base model.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument(
        "--model_source",
        choices=("local", "modelscope", "hf", "auto"),
        default="local",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tokenize_max_length", type=int, default=4096)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of questions to process; 0 means through the end.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference_predictions", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def exact_match(response: object, target: object) -> bool:
    targets = target if isinstance(target, list) else [target]
    normalized = normalize_answer(str(response))
    return any(normalized == normalize_answer(str(item)) for item in targets)


def load_reference(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return load_jsonl_tolerant(path)


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace, total: int) -> dict[str, Any]:
    correct = [bool(row.get("correct")) for row in rows]
    reference_rows = [row for row in rows if row.get("reference_correct") is not None]
    reference_correct = [bool(row["reference_correct"]) for row in reference_rows]
    summary: dict[str, Any] = {
        "status": "completed" if len(rows) == total else "partial",
        "count": len(rows),
        "requested_count": total,
        "em": sum(correct) / len(correct) if correct else 0.0,
        "candidate_valid_rate": sum(bool(row.get("candidate_valid")) for row in rows) / len(rows)
        if rows
        else 0.0,
        "reference_count": len(reference_rows),
        "reference_em": sum(reference_correct) / len(reference_correct)
        if reference_correct
        else None,
        "rerank_minus_reference_em": (
            (sum(correct) / len(correct)) - (sum(reference_correct) / len(reference_correct))
            if reference_correct
            else None
        ),
        "rerank_gains": sum(
            bool(row.get("correct")) and not bool(row.get("reference_correct"))
            for row in reference_rows
        ),
        "rerank_regressions": sum(
            not bool(row.get("correct")) and bool(row.get("reference_correct"))
            for row in reference_rows
        ),
        "start": args.start,
        "limit": args.limit,
        "input_file": str(args.input_file),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "reference_predictions": str(args.reference_predictions)
        if args.reference_predictions
        else None,
        "score_batch_size": args.score_batch_size,
        "updated_at_utc": now_utc(),
    }
    return summary


def load_model(args: argparse.Namespace):
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        load_dir=str(args.adapter_dir) if args.adapter_dir else None,
        model_source=args.model_source,
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.score_batch_size < 1:
        raise ValueError("--score_batch_size must be positive")
    if args.adapter_dir is not None and not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing adapter: {args.adapter_dir}")

    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    heartbeat_path = args.output_dir / "heartbeat.json"
    summary_path = args.output_dir / "summary.json"
    metadata_path = args.output_dir / "metadata.json"

    input_payload = json.loads(args.input_file.read_text(encoding="utf-8"))
    if isinstance(input_payload, dict):
        input_rows = [input_payload]
    elif isinstance(input_payload, list) and len(input_payload) == 1:
        input_rows = input_payload
    else:
        raise ValueError(
            "expected one standard GRIP graph record, found "
            f"{type(input_payload).__name__} with length "
            f"{len(input_payload) if isinstance(input_payload, list) else 'n/a'}"
        )
    eval_dataset = gen_eval_task(
        eval_mode="grip",
        input_data=input_rows[0],
        tokenizer=None,
        no_graph_context=True,
        use_subgraph=False,
        index_format=False,
    )
    total_questions = len(eval_dataset)
    end = total_questions if args.limit == 0 else min(total_questions, args.start + args.limit)
    if args.start >= total_questions and total_questions:
        raise ValueError(f"--start={args.start} is outside evaluation size {total_questions}")
    selected_indices = list(range(args.start, end))

    existing = load_jsonl_tolerant(prediction_path) if args.resume else []
    done_by_index: dict[int, dict[str, Any]] = {}
    for row in existing:
        if "dataset_index" in row:
            done_by_index[int(row["dataset_index"])] = row
    if args.resume and existing:
        # Remove duplicate or incomplete records before continuing.  The
        # rewrite is atomic, so a killed process cannot damage the valid rows.
        kept = [done_by_index[index] for index in sorted(done_by_index)]
        temporary = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(prediction_path)

    reference_rows = load_reference(args.reference_predictions)
    metadata = {
        "format_version": 1,
        "method": "official_candidate_teacher_forced_rerank",
        "created_at_utc": now_utc(),
        "input_file": str(args.input_file),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "model_mode": "adapter" if args.adapter_dir else "base_model",
        "model_name": args.model_name,
        "model_cache_dir": str(args.model_cache_dir),
        "model_source": args.model_source,
        "local_files_only": args.local_files_only,
        "dtype": "bfloat16",
        "candidate_protocol": "candidate list parsed from the official GRIP prompt; no test graph edges read",
        "total_questions": total_questions,
        "selected_start": args.start,
        "selected_end_exclusive": end,
        "selected_count": len(selected_indices),
        "resume_enabled": args.resume,
        "reference_predictions": str(args.reference_predictions)
        if args.reference_predictions
        else None,
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        heartbeat_path,
        {
            "status": "starting",
            "completed": len([index for index in selected_indices if index in done_by_index]),
            "selected_count": len(selected_indices),
            "updated_at_utc": now_utc(),
        },
    )

    model = tokenizer = None
    started = time.time()
    try:
        missing = [index for index in selected_indices if index not in done_by_index]
        if missing:
            model, tokenizer = load_model(args)
            # ``gen_eval_task`` only needs the tokenizer when a sample is
            # materialized.  Constructing the dataset before model loading
            # keeps the input graph CPU-only; attach the real tokenizer now.
            eval_dataset.tokenizer = tokenizer
            for position, index in enumerate(missing, start=1):
                input_ids, question, target = eval_dataset[index]
                if not isinstance(question, str):
                    question = str(question)
                candidates = parse_relation_candidates(question)
                if not candidates:
                    raise ValueError(f"question {index} has no official candidate list")
                target_values = target if isinstance(target, list) else [target]
                input_ids = input_ids.to(model.device)
                with torch.inference_mode():
                    score_result = score_candidate_prompt(
                        model,
                        tokenizer,
                        input_ids,
                        candidates,
                        candidate_batch_size=args.score_batch_size,
                    )
                score_values = [
                    float(item)
                    for item in score_result.scores[0].detach().cpu().tolist()
                ]
                best_index = max(range(len(candidates)), key=score_values.__getitem__)
                ranked = sorted(
                    zip(candidates, score_values), key=lambda item: item[1], reverse=True
                )
                reference = reference_rows[index] if index < len(reference_rows) else None
                row = {
                    "dataset_index": index,
                    "question_id": f"{input_rows[0].get('id', '0')}:test:{index}",
                    "split": "test",
                    "question": question,
                    "target": [str(item) for item in target_values],
                    "response": candidates[best_index],
                    "correct": exact_match(candidates[best_index], target_values),
                    "candidate_valid": True,
                    "candidate_scores": [
                        {"candidate": candidate, "score": score}
                        for candidate, score in zip(candidates, score_values)
                    ],
                    "top_candidates": [
                        {"candidate": candidate, "score": score}
                        for candidate, score in ranked[:5]
                    ],
                    "gold_rank": next(
                        (rank for rank, (candidate, _) in enumerate(ranked, start=1)
                         if any(normalize_answer(candidate) == normalize_answer(str(item)) for item in target_values)),
                        None,
                    ),
                    "score_margin": (
                        float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else None
                    ),
                    "reference_response": reference.get("response") if reference else None,
                    "reference_correct": (
                        bool(reference.get("correct")) if reference and "correct" in reference else None
                    ),
                }
                append_jsonl_fsync(prediction_path, row)
                done_by_index[index] = row
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": "running",
                        "completed": position + len(selected_indices) - len(missing),
                        "selected_count": len(selected_indices),
                        "last_dataset_index": index,
                        "elapsed_seconds": round(time.time() - started, 1),
                        "updated_at_utc": now_utc(),
                    },
                )
                if args.progress_every > 0 and (
                    position == 1 or position % args.progress_every == 0 or position == len(missing)
                ):
                    rows = [done_by_index[item] for item in selected_indices if item in done_by_index]
                    current = summarize(rows, args=args, total=len(selected_indices))
                    print(
                        f"[rerank] {len(rows)}/{len(selected_indices)} "
                        f"EM={current['em']:.4f} "
                        f"ref={current['reference_em'] if current['reference_em'] is not None else 'n/a'}",
                        flush=True,
                    )
        rows = [done_by_index[index] for index in selected_indices if index in done_by_index]
        summary = summarize(rows, args=args, total=len(selected_indices))
        summary["elapsed_seconds"] = round(time.time() - started, 1)
        atomic_write_json(summary_path, summary)
        atomic_write_json(
            heartbeat_path,
            {
                "status": "completed" if len(rows) == len(selected_indices) else "partial",
                "completed": len(rows),
                "selected_count": len(selected_indices),
                "elapsed_seconds": round(time.time() - started, 1),
                "updated_at_utc": now_utc(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
