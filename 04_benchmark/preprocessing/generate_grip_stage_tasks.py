#!/usr/bin/env python3
"""Generate GRIP Stage1/Stage2 tasks with a local model or OpenAI-compatible API.

This is intentionally separate from dataset conversion.  Context QA,
summarization, and reasoning QA are model-generated assets; native dataset
questions are kept as evaluation data and are never silently substituted for
these tasks.

Example smoke run (one graph, one sample of each expensive task):

    /home/kieran/miniconda3/envs/guardenv/bin/python \
        04_benchmark/preprocessing/generate_grip_stage_tasks.py \
        --dataset wn18rr --max-records 1 \
        --num-context-qa 1 --num-reason-qa 1 --num-summarization 1

The full run uses the same command without ``--max-records`` and with the
counts recorded in the dataset's generation plan.  Each expensive generator
stage is cached as JSONL and can be resumed safely.

For API-backed generation, pass ``--use-openai-api``. The SiliconFlow key is
read from the environment (or the configured dotenv file) and is never added
to the command line or run-state metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GRIP_ROOT = ROOT / "13_base_method" / "grip-exp"
DATA_ROOT = ROOT / "04_benchmark" / "data"
DEFAULT_MODEL_CACHE = Path("/mnt/e/GRIP-CU/model_cache")

sys.path.insert(0, str(GRIP_ROOT))

from grip.tasks import GripTaskGeneration  # noqa: E402
from models import get_hf_ft_tokenizer  # noqa: E402
from utils import set_random_seed  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_run_state(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist resumable-run metadata next to the stage caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("wn18rr", "codexm", "clegr_facts", "clegr_reasoning"),
    )
    parser.add_argument("--input-split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--reference-split",
        default=None,
        choices=("train", "val", "test"),
        help="Reference split for k-shot QA. Defaults to train when present, otherwise val (WN18RR).",
    )
    parser.add_argument("--input-file", default="")
    parser.add_argument("--reference-file", default="")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--task-cache-dir", default="")
    parser.add_argument("--max-records", type=int, default=0, help="Only process the first N graphs; useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-name", default="qwen-7b")
    parser.add_argument("--task-generator-model-name", default="qwen-7b")
    parser.add_argument("--use-openai-api", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--api-env-file", default=str(GRIP_ROOT / ".env"))
    parser.add_argument("--task-generator-max-retries", type=int, default=5)
    parser.add_argument("--task-generator-request-timeout", type=float, default=120.0)
    parser.add_argument("--model-source", choices=("local", "auto", "hf", "modelscope"), default="local")
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-generator-use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task-generator-batch-size", type=int, default=1)
    parser.add_argument("--task-generator-tokenize-max-length", type=int, default=10000)
    parser.add_argument("--tokenize-max-length", type=int, default=4096)
    parser.add_argument("--task-gen-max-length", type=int, default=1000)
    parser.add_argument("--num-context-qa", type=int, default=8000)
    parser.add_argument("--num-reason-qa", type=int, default=2000)
    parser.add_argument("--num-summarization", type=int, default=6000)
    parser.add_argument("--involve-qa-epochs", type=int, default=10)
    parser.add_argument("--sample-node-attribute-task", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repharse-context-qa", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-upsampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--format-as-instruction", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_openai_api:
        from dotenv import load_dotenv

        env_file = Path(args.api_env_file).expanduser()
        if env_file.is_file():
            load_dotenv(env_file, override=False)
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {args.api_key_env} in the environment or {env_file}."
            )
        # ChatOpenAI resolves this standard provider variable from api_base.
        if args.api_key_env != "SILICONFLOW_API_KEY":
            os.environ["SILICONFLOW_API_KEY"] = api_key
        if not args.api_base:
            args.api_base = "https://api.siliconflow.cn/v1"
        # Use the provider's model ID, while retaining qwen-7b for the local
        # tokenizer loaded below. Local checkpoints are never loaded for API inference.
        if args.task_generator_model_name == "qwen-7b":
            args.task_generator_model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif not args.api_base:
        args.api_base = "https://api.openai.com/v1/"

    started_at = utc_now()
    started_clock = time.monotonic()
    dataset_dir = DATA_ROOT / args.dataset
    input_file = Path(args.input_file) if args.input_file else dataset_dir / "processed" / f"processed_{args.input_split}.json"
    if args.reference_file:
        reference_file = Path(args.reference_file)
        reference_split = args.reference_split or "custom"
    else:
        reference_split = args.reference_split
        if reference_split is None:
            reference_split = "train" if (dataset_dir / "processed" / "processed_train.json").is_file() else "val"
        reference_file = dataset_dir / "processed" / f"processed_{reference_split}.json"
    # Keep the resolved split in the serialized generation config and logs.
    args.reference_split = reference_split
    output_file = Path(args.output_file) if args.output_file else dataset_dir / "tasks" / f"grip_{args.dataset}_tasks.json"
    default_cache_name = (
        "cache_siliconflow_qwen25_7b" if args.use_openai_api else "cache_qwen7b"
    )
    cache_dir = Path(args.task_cache_dir) if args.task_cache_dir else dataset_dir / "tasks" / default_cache_name

    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if not reference_file.is_file():
        raise FileNotFoundError(reference_file)
    input_data = load_records(input_file)
    reference_data = load_records(reference_file)
    if args.max_records:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")
        input_data = input_data[:args.max_records]
    if not input_data:
        raise ValueError(f"No graph records found in {input_file}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = cache_dir / "run_state.json"
    previous_state = None
    if run_state_path.is_file():
        try:
            previous_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_state = {"status": "unreadable_previous_state"}
    if isinstance(previous_state, dict):
        previous_state = {
            key: previous_state.get(key)
            for key in ("status", "pid", "started_at_utc", "completed_at_utc", "failed_at_utc", "elapsed_seconds", "counts")
            if key in previous_state
        }
    run_state = {
        "format_version": 1,
        "status": "running",
        "resumable": True,
        "pid": os.getpid(),
        "started_at_utc": started_at,
        "dataset": args.dataset,
        "input_file": relative_or_absolute(input_file),
        "reference_file": relative_or_absolute(reference_file),
        "cache_dir": relative_or_absolute(cache_dir),
        "config": vars(args),
        "previous_state": previous_state,
        "resume_rule": "Rerun the identical command; completed JSONL cache rows are reused by index and prompt_sha256.",
    }
    write_run_state(run_state_path, run_state)

    print(json.dumps({
        "event": "stage_generation_start",
        "dataset": args.dataset,
        "input_file": relative_or_absolute(input_file),
        "reference_file": relative_or_absolute(reference_file),
        "input_graphs": len(input_data),
        "reference_graphs": len(reference_data),
        "model_name": args.task_generator_model_name,
        "model_cache_dir": str(Path(args.model_cache_dir).resolve()),
        "cache_dir": relative_or_absolute(cache_dir),
        "num_context_qa": args.num_context_qa,
        "num_reason_qa": args.num_reason_qa,
        "num_summarization": args.num_summarization,
    }, ensure_ascii=False), flush=True)
    if args.use_openai_api:
        # Sampling and prompt construction use Python/NumPy RNGs only. Avoid
        # CUDA seed helpers while another experiment owns the local GPU.
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        set_random_seed(args.seed)

    tokenizer = get_hf_ft_tokenizer(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
    )
    generator = GripTaskGeneration(
        graph_list=[record["graph"] for record in input_data],
        title_list=[record.get("title", f"{args.dataset} graph {record.get('id', index)}") for index, record in enumerate(input_data)],
        tokenizer=tokenizer,
        refer_data=reference_data,
        task_generator_model_name=args.task_generator_model_name,
        task_generator_use_vllm=args.task_generator_use_vllm,
        task_generator_batch_size=args.task_generator_batch_size,
        task_cache_dir=str(cache_dir),
        task_gen_max_length=args.task_gen_max_length,
        task_generator_tokenize_max_length=args.task_generator_tokenize_max_length,
        num_context_qa=args.num_context_qa,
        num_reason_qa=args.num_reason_qa,
        num_summarization=args.num_summarization,
        sample_node_attribute_task=args.sample_node_attribute_task,
        repharse_context_qa=args.repharse_context_qa,
        context_upsampling=args.context_upsampling,
        format_as_instruction=args.format_as_instruction,
        involve_qa_epochs=args.involve_qa_epochs,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        model_name=args.model_name,
        use_openai_api=args.use_openai_api,
        api_base=args.api_base,
        api_key="${SILICONFLOW_API_KEY}" if args.use_openai_api else "${OPENAI_API_KEY}",
        task_generator_max_retries=args.task_generator_max_retries,
        task_generator_request_timeout=args.task_generator_request_timeout,
    )
    try:
        task_data = generator()
    except BaseException as error:
        run_state.update({
            "status": "failed_resumable",
            "failed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": round(time.monotonic() - started_clock, 3),
        })
        write_run_state(run_state_path, run_state)
        raise

    per_graph = []
    context_samples: list[str] = []
    qa_samples: list[str] = []
    for record, generated in zip(input_data, task_data):
        context = list(generated.context_samples)
        qa = list(generated.qa_samples)
        context_samples.extend(context)
        qa_samples.extend(qa)
        per_graph.append({
            "id": record.get("id"),
            "title": record.get("title"),
            "context_samples": context,
            "qa_samples": qa,
        })

    payload = {
        "format_version": 2,
        "created_at_utc": utc_now(),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_clock, 3),
        "dataset": args.dataset,
        "generation_status": "complete",
        "source": {
            "input_file": relative_or_absolute(input_file),
            "reference_file": relative_or_absolute(reference_file),
            "input_graphs": len(input_data),
            "reference_graphs": len(reference_data),
        },
        "generation_config": vars(args),
        "counts": {
            "context_samples": len(context_samples),
            "qa_samples": len(qa_samples),
            "graphs": len(per_graph),
        },
        "context_samples": context_samples,
        "qa_samples": qa_samples,
        "per_graph": per_graph,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_file)
    run_state.update({
        "status": "complete",
        "completed_at_utc": payload["completed_at_utc"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "output_file": relative_or_absolute(output_file),
        "counts": payload["counts"],
    })
    write_run_state(run_state_path, run_state)
    print(json.dumps({
        "event": "stage_generation_complete",
        "output_file": relative_or_absolute(output_file),
        "cache_dir": relative_or_absolute(cache_dir),
        "counts": payload["counts"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
