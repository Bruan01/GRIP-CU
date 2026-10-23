#!/usr/bin/env python3
"""Record durable provenance and artifact state for every experiment stage.

This helper is intentionally independent from the model code.  The launcher
calls it at stage start and stage exit so that even a failed or interrupted
GPU stage leaves a reviewable record instead of only a terminal log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
HASH_LIMIT_BYTES = 128 * 1024 * 1024


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            argv, stderr=subprocess.STDOUT, text=True, timeout=30
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def path_record(path: Path, *, hash_large: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    if path.is_dir():
        # A model cache can contain many gigabytes and tens of thousands of
        # files.  Keep directory provenance cheap and hash important files
        # separately below instead of recursively walking the cache at every
        # stage boundary.
        entries = list(path.iterdir())
        files = [item for item in entries if item.is_file()]
        record.update(
            {
                "kind": "directory",
                "entry_count_top_level": len(entries),
                "file_count_top_level": len(files),
                "total_bytes_top_level": sum(item.stat().st_size for item in files),
                "scan_scope": "top_level_only",
            }
        )
        return record
    stat = path.stat()
    record.update(
        {
            "kind": "file",
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    # Hash multi-gigabyte adapter weights at stage finish (when the run has a
    # durable boundary), while keeping stage-start recording fast.
    force_hash = path.name == "adapter_model.safetensors" and hash_large
    if hash_large or force_hash or stat.st_size <= HASH_LIMIT_BYTES:
        try:
            record["sha256"] = sha256_file(path)
        except OSError as error:
            record["sha256_error"] = str(error)
    else:
        record["sha256"] = None
        record["sha256_deferred"] = True
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("start", "finish"), required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--exit_code", type=int, default=0)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--negative_source", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--input_file", type=Path, default=None)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--model_cache_dir", type=Path, default=None)
    parser.add_argument("--s1_adapter", type=Path, default=None)
    parser.add_argument("--b1_adapter", type=Path, default=None)
    parser.add_argument("--scores_dir", type=Path, default=None)
    parser.add_argument("--sampler_dir", type=Path, default=None)
    parser.add_argument("--benchmark_dir", type=Path, default=None)
    parser.add_argument("--rollout_manifest", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--counterfactual_metadata", type=Path, default=None)
    parser.add_argument("--relation_text_file", type=Path, default=None)
    parser.add_argument("--entity_text_file", type=Path, default=None)
    parser.add_argument("--hash_large", action="store_true")
    return parser.parse_args()


def expected_paths(args: argparse.Namespace) -> dict[str, Path]:
    run = args.run_dir
    paths: dict[str, Path] = {
        "run_config": run / "run_config.json",
        "environment": run / "environment.txt",
        "environment_json": run / "environment.json",
        "command": run / "command.txt",
        "stage_status": run / "stage_status.json",
        "stage_history": run / "stage_history.jsonl",
        "artifact_manifest": run / "artifact_manifest.json",
        "resource_usage": run / "resource_usage.json",
        "stage_log": run / f"{args.step}.log",
    }
    if args.scores_dir:
        paths.update(
            {
                "candidate_scores": args.scores_dir / "candidate_scores.jsonl",
                "qa_summary": args.scores_dir / "qa_summary.jsonl",
                "score_metadata": args.scores_dir / "metadata.json",
                "score_stats": args.scores_dir / "run_stats.json",
            }
        )
    if args.benchmark_dir:
        paths["score_benchmark"] = args.benchmark_dir / "benchmark.json"
    if args.rollout_manifest:
        paths["rollout_manifest"] = args.rollout_manifest
    if args.sampler_dir:
        paths.update(
            {
                "counterfactual_bank": args.sampler_dir / "counterfactual_bank.jsonl",
                "sampler_audit": args.sampler_dir / "sampler_audit.jsonl",
                "relation_profiles": args.sampler_dir / "relation_profiles.json",
                "sampler_metadata": args.sampler_dir / "metadata.json",
                "sampler_wiring": args.sampler_dir / "wiring.json",
            }
        )
    paths.update(
        {
            "budget": run / "budget.json",
            "wiring": run / "wiring.json",
            "comparison": run / "comparison.json",
            "train_log": run / "run.log",
            "listed_adapter": run / "listed" / "adapter" / "adapter_model.safetensors",
            "listed_adapter_config": run / "listed" / "adapter" / "adapter_config.json",
            "listed_predictions": run / "listed" / "predictions_correct.jsonl",
            "listed_summary": run / "listed" / "summary.json",
            "listed_metadata": run / "listed" / "metadata.json",
            "swanlab_run": run / "swanlab_run.json",
        }
    )
    return paths


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, path in (
        ("input_file", args.input_file),
        ("eval_file", args.eval_file),
        ("s1_adapter", args.s1_adapter),
        ("b1_adapter", args.b1_adapter),
    ):
        if path is not None:
            paths[label] = path
    for label, path in (("s1_adapter", args.s1_adapter), ("b1_adapter", args.b1_adapter)):
        if path is not None:
            paths[f"{label}_config"] = path / "adapter_config.json"
            paths[f"{label}_weights"] = path / "adapter_model.safetensors"
    if args.raw_dir:
        for name in ("train.txt", "valid.txt", "test.txt", "rel2text.json"):
            paths[f"raw_{name}"] = args.raw_dir / name
    if args.model_cache_dir:
        model_dir = args.model_cache_dir / "Qwen--Qwen2.5-7B-Instruct"
        paths["model_cache"] = model_dir
        paths["model_config"] = model_dir / "config.json"
        paths["model_index"] = model_dir / "model.safetensors.index.json"
    for label, path in (
        ("manifest", args.manifest),
        ("counterfactual_metadata", args.counterfactual_metadata),
        ("relation_text_file", args.relation_text_file),
        ("entity_text_file", args.entity_text_file),
        ("rollout_manifest", args.rollout_manifest),
    ):
        if path is not None:
            paths[label] = path
    return paths


def environment_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:  # pragma: no cover - environment dependent
        torch_info = {"error": repr(error)}
    return {
        "created_at_utc": now_utc(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch_info,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
        ),
        "git_head": git_head(args.repo_root),
        "baseline_git_head": git_head(args.baseline_dir),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text)


def refresh_manifest(args: argparse.Namespace, *, hash_large: bool) -> None:
    expected = expected_paths(args)
    cache: dict[tuple[str, bool], dict[str, Any]] = {}

    def cached_record(path: Path) -> dict[str, Any]:
        key = (str(path), hash_large)
        if key not in cache:
            cache[key] = path_record(path, hash_large=hash_large)
        return dict(cache[key])

    observed = {label: cached_record(path) for label, path in expected.items()}
    external = {label: cached_record(path) for label, path in input_paths(args).items()}
    stage_required = {
        "preflight": ["run_config", "environment", "stage_status", "stage_log"],
        "score": ["candidate_scores", "qa_summary", "score_metadata", "score_stats"],
        "rollout": ["rollout_manifest"],
        "freeze": [
            "counterfactual_bank",
            "sampler_audit",
            "relation_profiles",
            "sampler_metadata",
        ],
        "wire": ["wiring"],
        "train": [
            "budget",
            "listed_adapter",
            "listed_adapter_config",
            "listed_predictions",
            "listed_summary",
            "listed_metadata",
            "train_log",
        ],
    }
    required = stage_required.get(args.step, [])
    missing = [label for label in required if not observed[label]["exists"]]
    payload = {
        "format_version": FORMAT_VERSION,
        "updated_at_utc": now_utc(),
        "run_id": args.run_id or args.run_dir.name,
        "step": args.step,
        "status": args.status or ("running" if args.phase == "start" else "unknown"),
        "required_for_step": required,
        "missing_required_for_step": missing,
        "step_artifacts_complete": not missing,
        "expected": observed,
        "external_inputs": external,
        "hash_policy": {
            "sha256_limit_bytes": HASH_LIMIT_BYTES,
            "hash_large": hash_large,
            "adapter_weights_hashed_on_finish": True,
        },
    }
    write_json(args.run_dir / "artifact_manifest.json", payload)


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.run_dir / "run_config.json"
    config = load_json(run_config_path, {})
    if not isinstance(config, dict):
        config = {}
    config.setdefault("format_version", FORMAT_VERSION)
    config.setdefault("run_id", args.run_id or args.run_dir.name)
    config.setdefault("created_at_utc", now_utc())
    config.update(
        {
            "dataset": args.dataset,
            "negative_source": args.negative_source,
            "last_step": args.step,
            "paths": {
                "run_dir": str(args.run_dir),
                "input_file": str(args.input_file) if args.input_file else None,
                "eval_file": str(args.eval_file) if args.eval_file else None,
                "raw_dir": str(args.raw_dir) if args.raw_dir else None,
                "model_cache_dir": str(args.model_cache_dir) if args.model_cache_dir else None,
                "scores_dir": str(args.scores_dir) if args.scores_dir else None,
                "sampler_dir": str(args.sampler_dir) if args.sampler_dir else None,
                "manifest": str(args.manifest) if args.manifest else None,
                "s1_adapter": str(args.s1_adapter) if args.s1_adapter else None,
                "b1_adapter": str(args.b1_adapter) if args.b1_adapter else None,
                "entity_text_file": str(args.entity_text_file) if args.entity_text_file else None,
            },
            "protocol": {
                "listed_negative_k": int(os.environ.get("LISTED_NEGATIVE_K", "9")),
                "expected_listed_qa": int(os.environ.get("EXPECTED_LISTED", "0")),
                "seed": os.environ.get("SEED"),
                "torch_dtype": os.environ.get("TORCH_DTYPE"),
                "model_name": os.environ.get("MODEL_NAME"),
                "candidate_batch_size": os.environ.get("CANDIDATE_BATCH_SIZE"),
                "filter_splits": ["train"],
                "swanlab_project": os.environ.get("SWANLAB_PROJECT", ""),
                "swanlab_experiment_name": os.environ.get("SWANLAB_EXPERIMENT_NAME", ""),
                "swanlab_group": os.environ.get("SWANLAB_GROUP", ""),
                "swanlab_mode": os.environ.get("SWANLAB_MODE", ""),
                "uniform_mix": os.environ.get("COUNTERFACTUAL_UNIFORM_MIX", "0.3333333333333333"),
                "structure_weight": os.environ.get("COUNTERFACTUAL_STRUCTURE_WEIGHT", "1.0"),
                "violation_weight": os.environ.get("COUNTERFACTUAL_VIOLATION_WEIGHT", "1.0"),
                "temperature": os.environ.get("COUNTERFACTUAL_TEMPERATURE", "0.25"),
            },
        }
    )
    history = config.setdefault("stage_history", [])
    if not isinstance(history, list):
        history = []
        config["stage_history"] = history

    timestamp = now_utc()
    if args.phase == "start":
        event = {
            "step": args.step,
            "status": "running",
            "started_at_utc": timestamp,
            "pid": os.getpid(),
            "command": args.command,
        }
        history.append(event)
        env = environment_payload(args)
        write_json(args.run_dir / "environment.json", env)
        append_text(
            args.run_dir / "environment.txt",
            "\n".join(
                [
                    f"[{args.step}] start={timestamp}",
                    f"git_head={env.get('git_head')}",
                    f"baseline_git_head={env.get('baseline_git_head')}",
                    f"python={env.get('python_executable')}",
                    f"torch={json.dumps(env.get('torch'), ensure_ascii=False)}",
                    f"nvidia_smi={env.get('nvidia_smi')}",
                    f"command={args.command}",
                    "",
                ]
            ),
        )
        append_text(
            args.run_dir / "command.txt",
            f"[{timestamp}] step={args.step}\n{args.command}\n\n",
        )
        write_json(
            args.run_dir / "stage_status.json",
            {
                "step": args.step,
                "status": "running",
                "started_at_utc": timestamp,
                "run_id": config["run_id"],
            },
        )
        append_text(
            args.run_dir / "stage_history.jsonl",
            json.dumps(event, ensure_ascii=False) + "\n",
        )
    else:
        final_status = args.status or ("completed" if args.exit_code == 0 else "failed")
        matching = next(
            (
                item
                for item in reversed(history)
                if isinstance(item, dict)
                and item.get("step") == args.step
                and item.get("status") == "running"
            ),
            None,
        )
        if matching is None:
            matching = {"step": args.step, "started_at_utc": timestamp}
            history.append(matching)
        matching.update(
            {
                "status": final_status,
                "finished_at_utc": timestamp,
                "exit_code": args.exit_code,
            }
        )
        write_json(
            args.run_dir / "stage_status.json",
            {
                **matching,
                "run_id": config["run_id"],
            },
        )
        append_text(
            args.run_dir / "stage_history.jsonl",
            json.dumps(
                {
                    "step": args.step,
                    "status": final_status,
                    "finished_at_utc": timestamp,
                    "exit_code": args.exit_code,
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        append_text(
            args.run_dir / "environment.txt",
            f"[{args.step}] finish={timestamp} status={final_status} exit_code={args.exit_code}\n",
        )

    config["stage_history"] = history
    config["status"] = (
        "running" if args.phase == "start" else (args.status or ("completed" if args.exit_code == 0 else "failed"))
    )
    config["updated_at_utc"] = timestamp
    write_json(run_config_path, config)
    refresh_manifest(args, hash_large=args.hash_large)

    resource_path = args.run_dir / "resource_usage.json"
    resource = load_json(resource_path, {"format_version": FORMAT_VERSION, "stages": []})
    if not isinstance(resource, dict):
        resource = {"format_version": FORMAT_VERSION, "stages": []}
    stages = resource.setdefault("stages", [])
    if args.phase == "start":
        stages.append(
            {
                "step": args.step,
                "started_at_utc": timestamp,
                "gpu_snapshot": command_output(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader"]
                ),
            }
        )
    else:
        stage = next(
            (
                item
                for item in reversed(stages)
                if isinstance(item, dict)
                and item.get("step") == args.step
                and "finished_at_utc" not in item
            ),
            None,
        )
        if stage is None:
            stage = {"step": args.step}
            stages.append(stage)
        stage.update(
            {
                "finished_at_utc": timestamp,
                "exit_code": args.exit_code,
                "status": args.status or ("completed" if args.exit_code == 0 else "failed"),
                "gpu_snapshot_end": command_output(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader"]
                ),
            }
        )
    resource["updated_at_utc"] = timestamp
    write_json(resource_path, resource)


if __name__ == "__main__":
    main()
