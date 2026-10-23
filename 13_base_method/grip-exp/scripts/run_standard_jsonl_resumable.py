"""Resumable standard graph-QA inference with question-level checkpoints.

The original ``run_llm.py`` writes only after every input record has finished
and silently drops generations that do not contain the requested XML tag.  This
runner is intended for long official CLEGR/WN evaluation jobs: it appends one
JSONL row per question, fsyncs every micro-batch, and keeps raw model output for
later auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constants import SYSTEM_PROMPT  # noqa: E402
from evaluation.utils import normalize_answer  # noqa: E402
from grip.paper_protocol import get_paper_protocol, validate_main_eval_flags  # noqa: E402
from grip.tasks import gen_eval_task  # noqa: E402
from models import get_inf_model  # noqa: E402
from utils import extract_tag_content  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def extract_prediction(raw_response: str) -> str:
    answers = extract_tag_content(raw_response, "answer")
    answer = "; ".join(answers).strip()
    return answer if answer else raw_response.strip()


def build_model(args: argparse.Namespace):
    return get_inf_model(
        model_name=args.model_name,
        load_dir=args.load_dir,
        quantization=args.quantization,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        tokenize_max_length=args.tokenize_max_length,
        dtype=args.dtype,
        use_vllm=args.use_vllm,
        gen_max_length=args.gen_max_length,
        # The outer loop owns checkpointing and micro-batching.
        batch_size=-1,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )


def evaluate(output_file: Path) -> dict:
    total = correct = empty = 0
    response_lengths = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = normalize_answer(str(row.get("response", "")))
            targets = row.get("target", [])
            if isinstance(targets, str):
                targets = [targets]
            target_norms = {normalize_answer(str(target)) for target in targets}
            total += 1
            correct += int(prediction in target_norms)
            empty += int(not prediction)
            response_lengths.append(len(str(row.get("response", ""))))
    return {
        "count": total,
        "exact_match": correct / total if total else 0.0,
        "correct": correct,
        "empty_response_count": empty,
        "empty_response_rate": empty / total if total else 0.0,
        "mean_response_characters": sum(response_lengths) / total if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--metadata_file", default=None)
    parser.add_argument("--heartbeat_file", default=None)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_source", default="auto")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--load_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--tokenize_max_length", type=int, default=8192)
    parser.add_argument("--padding_side", default="right")
    parser.add_argument("--truncation_side", default="right")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--gen_max_length", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument(
        "--eval_mode",
        choices=["standard", "grip"],
        default="standard",
        help="Use the original GRIP prompt format for paper-main adapter evaluation.",
    )
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--protocol",
        choices=["custom", "paper_wn18rr", "paper_clegr_reasoning"],
        default="custom",
        help="Use a canonical paper-main evaluation configuration.",
    )
    parser.add_argument("--no_graph_context", action="store_true")
    parser.add_argument("--use_subgraph", action="store_true")
    parser.add_argument("--index_format", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.micro_batch_size < 1:
        raise ValueError("--micro_batch_size must be positive")

    if args.protocol != "custom":
        dataset = args.protocol.removeprefix("paper_")
        protocol = get_paper_protocol(dataset)
        args.no_graph_context = protocol.no_graph_context
        args.use_subgraph = protocol.use_subgraph
        args.eval_mode = "grip"
        args.tokenize_max_length = 4096
        args.gen_max_length = 1000
        validate_main_eval_flags(
            dataset,
            no_graph_context=args.no_graph_context,
            use_subgraph=args.use_subgraph,
        )
    elif args.no_graph_context and args.use_subgraph:
        raise ValueError("--no_graph_context and --use_subgraph cannot both be enabled")

    input_file = Path(args.input_file).resolve()
    output_file = Path(args.output_file).resolve()
    metadata_file = Path(args.metadata_file).resolve() if args.metadata_file else output_file.with_name("metadata.json")
    heartbeat_file = Path(args.heartbeat_file).resolve() if args.heartbeat_file else output_file.with_name("heartbeat.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_file.exists():
        output_file.unlink()
    completed = count_jsonl(output_file)
    records = load_jsonl(input_file)
    total_questions = sum(len(record.get("questions", [])) for record in records)
    start_time = time.time()
    metadata = {
        "status": "running",
        "started_at": now_iso(),
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "output_file": str(output_file),
        "total_records": len(records),
        "total_questions": total_questions,
        "resumed_questions": completed,
        "model": vars(args),
        "python": sys.executable,
    }
    atomic_json(metadata_file, metadata)
    atomic_json(heartbeat_file, {
        "status": "running",
        "updated_at": now_iso(),
        "completed_questions": completed,
        "total_questions": total_questions,
    })

    model = build_model(args)
    remaining_skip = completed
    completed_now = completed
    try:
        with output_file.open("a", encoding="utf-8") as output_handle:
            for record_index, record in enumerate(records):
                eval_dataset = gen_eval_task(
                    eval_mode=args.eval_mode,
                    input_data=record,
                    tokenizer=model.tokenizer,
                    no_graph_context=args.no_graph_context,
                    use_subgraph=args.use_subgraph,
                    index_format=args.index_format,
                )
                entries = []
                for user_content, question, target in eval_dataset:
                    entries.append({
                        "id": record["id"],
                        "question": question,
                        "target": [target],
                        "user_content": user_content,
                    })
                if remaining_skip >= len(entries):
                    remaining_skip -= len(entries)
                    continue
                if remaining_skip:
                    entries = entries[remaining_skip:]
                    remaining_skip = 0

                for start in range(0, len(entries), args.micro_batch_size):
                    batch = entries[start:start + args.micro_batch_size]
                    results = model.inference(
                        user_contents=[entry["user_content"] for entry in batch],
                        system_prompt=SYSTEM_PROMPT,
                    )
                    for entry, result in zip(batch, results):
                        raw_response = str(result.get("response", ""))
                        row = {
                            "id": entry["id"],
                            "question": entry["question"],
                            "response": extract_prediction(raw_response),
                            "raw_response": raw_response,
                            "evidence": "; ".join(extract_tag_content(raw_response, "evidence")).strip(),
                            "target": entry["target"],
                        }
                        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed_now += 1
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    atomic_json(heartbeat_file, {
                        "status": "running",
                        "updated_at": now_iso(),
                        "completed_questions": completed_now,
                        "total_questions": total_questions,
                        "record_index": record_index,
                    })
    except Exception as exc:
        atomic_json(heartbeat_file, {
            "status": "failed",
            "updated_at": now_iso(),
            "completed_questions": completed_now,
            "total_questions": total_questions,
            "error": repr(exc),
        })
        metadata.update({"status": "failed", "finished_at": now_iso(), "error": repr(exc)})
        atomic_json(metadata_file, metadata)
        raise

    summary = evaluate(output_file)
    summary.update({
        "status": "complete",
        "elapsed_seconds": time.time() - start_time,
        "input_sha256": metadata["input_sha256"],
        "output_sha256": sha256_file(output_file),
    })
    atomic_json(output_file.with_name("summary.json"), summary)
    atomic_json(heartbeat_file, {
        "status": "complete",
        "updated_at": now_iso(),
        "completed_questions": completed_now,
        "total_questions": total_questions,
        "summary": summary,
    })
    metadata.update({"status": "complete", "finished_at": now_iso(), "summary": summary})
    atomic_json(metadata_file, metadata)


if __name__ == "__main__":
    main()
