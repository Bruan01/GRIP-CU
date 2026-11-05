#!/usr/bin/env python3
"""Prepare, execute, resume and archive the single-seed FB Random-9 control.

Default action is CPU-only preparation. `run` explicitly enables the frozen
32-step training and full validation/test evaluation, with no search/repeats.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HNG = Path(__file__).resolve().parents[1]
ROOT = HNG.parents[1]
sys.path.insert(0, str(HNG / "src"))
from hard_negative_grip.frozen_random import score_groups, uniform_bank_row, validate_frozen_random_manifest
from hard_negative_grip.matched_control import (
    audit_predictions, paired_result, parameter_diff, read_json, sha256,
    training_command, write_csv, write_json,
)
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.decode_io import eval_samples
from hard_negative_grip.task_file import (
    build_qa_assets_from_task_texts, known_pair_relations, load_counterfactual_manifest,
    load_error_memory_manifest, load_graph_record, load_relation_eligibility,
    original_question_ids_from_payload, validate_structure_counterfactual_manifest,
)

REL_HNG = Path("08_experiments/Hard_Negative_Contrastive_GRIP")
REL_BASE = Path("13_base_method/grip-exp")
PYTHON = Path("/home/kieran/miniconda3/envs/guardenv/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_source(value):
    """Resolve only known repository aliases; do not rewrite arbitrary paths."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        path = ROOT/path
        if path.exists():
            return path.resolve()
    marker = "/grip-cu/"
    lowered = str(value).lower()
    if marker in lowered:
        candidate = ROOT/str(value)[lowered.index(marker)+len(marker):]
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(value)


def append_event(run, event):
    with (run/"events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": now(), **event}, ensure_ascii=False)+"\n")


@contextmanager
def locked(run):
    run.mkdir(parents=True, exist_ok=True)
    with (run/"run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller already owns this run") from error
        yield


def capture(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": repr(error)}


def environment(run):
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repository = Path(read_json(run/"contract.json").get("repository_root", str(ROOT))) if (run/"contract.json").exists() else ROOT
    return {"at": now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "git_head": capture(["git", "-C", str(repository), "rev-parse", "HEAD"]),
            "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                            "--format=csv,noheader"]),
            "free_disk_bytes": shutil.disk_usage(run).free}


def normalization(code_root):
    spec = importlib.util.spec_from_file_location("matched_em_normalization", code_root/REL_BASE/"evaluation/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_answer


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def freeze(run, source, relative, artifacts, expected=None):
    source = resolve_source(source)
    digest = sha256(source)
    require(expected is None or digest == expected, f"reference hash mismatch: {relative}")
    target = run/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"refusing to overwrite prepared input: {relative}")
    shutil.copy2(source, target)
    require(sha256(target) == digest, f"input changed during copy: {relative}")
    artifacts[str(relative)] = {"sha256": digest, "bytes": target.stat().st_size,
                                "source": str(source)}
    return target


def prepare(run, config_path):
    if (run/"contract.json").exists():
        contract = verify(run)
        require(sha256(config_path) == contract["design_sha256"], "design config changed")
        return contract
    require(not (run/"inputs").exists(), "partial preparation preserved; use a new run directory")
    design = read_json(config_path)
    require(design["seed"] == 2026 and design["optimizer_steps"] == 32,
            "this control has a single seed=2026 and exactly 32 steps")
    append_event(run, {"stage": "prepare", "status": "running"})
    evidence_path = resolve_source(design["reference_evidence"])
    evidence = read_json(evidence_path)
    ref_path = resolve_source(design["reference_run_config"])
    ref = read_json(ref_path)
    origin = resolve_source(design["reference_root"])
    hashes = evidence["audit_identity"]["inputs"]
    audit_hash = lambda suffix: next(value for key, value in hashes.items() if key.endswith(suffix))
    require(sha256(ref_path) == audit_hash("/train_run/run_config.json"), "unaudited parent run config")
    selection_path = resolve_source(design["reference_selection"])
    require(sha256(selection_path) == audit_hash("/search/selection.json"), "parent selection changed")
    selection = read_json(selection_path)
    require(selection["winner"]["step"] == 32 and selection["test_used_in_this_search_selection"] is False,
            "parent reference is not the validation-selected step32")
    require(ref["seed"] == 2026 and ref["stop_after_step"] == 32 and ref["involve_qa_epochs"] == 5.0,
            "parent training contract changed")
    parent_state = read_json(origin/"search/trials/trial_0001/train_run/trainer_listed/checkpoint-32/trainer_state.json")
    require(parent_state["global_step"] == 32 and parent_state["max_steps"] == design["scheduler_max_steps"],
            "parent actual optimizer/scheduler state differs")
    artifacts = {}
    freeze(run, config_path, "inputs/design.json", artifacts)
    freeze(run, evidence_path, "reference/evidence.json", artifacts)
    freeze(run, ref_path, "reference/run_config.json", artifacts)
    freeze(run, selection_path, "reference/selection.json", artifacts)
    freeze(run, origin/"search/trials/trial_0001/train_run/trainer_listed/checkpoint-32/trainer_state.json",
           "reference/trainer_state.json", artifacts)
    freeze(run, origin/"search/selected_test/cache_identity.json", "reference/cache_identity.json", artifacts)
    freeze(run, origin/"search/selected_test/predictions.jsonl", "reference/predictions.jsonl", artifacts,
           audit_hash("/search/selected_test/predictions.jsonl"))
    freeze(run, ref["input_file"], "inputs/task.json", artifacts)
    freeze(run, ref["relation_eligibility"], "inputs/relation_eligibility.json", artifacts,
           audit_hash("/assets/relation_eligibility.json"))
    raw = resolve_source(ref["raw_dir"])
    for name in ("train.txt", "rel2text.json"):
        freeze(run, raw/name, "inputs/raw/"+name, artifacts)
    freeze(run, ref["paths"]["entity_text_file"], "inputs/entity2text_train.txt", artifacts)
    freeze(run, ref["eval_file"], "inputs/validation.json", artifacts)
    freeze(run, ROOT/"04_benchmark/data/fb15k237_2/processed/processed_test.json", "inputs/test.json", artifacts,
           evidence["audit_identity"]["test_sha256"])
    for source in sorted(resolve_source(ref["s1_adapter"]).iterdir()):
        if source.is_file():
            freeze(run, source, "inputs/stage1_adapter/"+source.name, artifacts)
    for name in ("counterfactual_bank.jsonl", "metadata.json", "wiring.json"):
        freeze(run, origin/"assets/k9"/name, "reference/"+name, artifacts)
    require(sha256(run/"reference/counterfactual_bank.jsonl") == read_json(run/"reference/wiring.json")["manifest_sha256"],
            "parent bank differs from wiring audit")
    # Freeze executable sources with their package layout, not mutable shell runners.
    code_files = set()
    for directory in (HNG/"src", HNG/"scripts"):
        code_files.update(directory.rglob("*.py"))
    baseline = ROOT/REL_BASE
    code_files.update(baseline.glob("*.py"))
    for name in ("arguments", "data", "evaluation", "grip", "models"):
        code_files.update((baseline/name).rglob("*.py"))
    for source in sorted(code_files):
        freeze(run, source, Path("code_snapshot")/source.relative_to(ROOT), artifacts)
    code_root = run/"code_snapshot"
    payload = read_json(run/"inputs/task.json")
    texts = payload["qa_samples"]
    require(len(texts) == design["qa_count"], "training QA count changed")
    eligible = load_relation_eligibility(run/"inputs/relation_eligibility.json", run/"inputs/task.json", run/"inputs/raw")
    parent_bank = load_counterfactual_manifest(run/"reference/counterfactual_bank.jsonl")
    validate_structure_counterfactual_manifest(parent_bank, k=design["k"])
    require(set(parent_bank) == eligible and len(eligible) == design["listed_qa_count"], "parent candidate eligibility changed")
    vocabulary = load_train_relation_order(run/"inputs/raw", relation_text_file=run/"inputs/raw/rel2text.json")
    known = known_pair_relations(run/"inputs/raw", splits=("train",),
                                 relation_text_file=run/"inputs/raw/rel2text.json",
                                 entity_text_file=run/"inputs/entity2text_train.txt")
    scores = origin/"assets/b1_train_scores/candidate_scores.jsonl"
    require(read_json(origin/"assets/b1_train_scores/metadata.json")["filter_splits"] == ["train"],
            "parent scores use a different filtering split")
    (run/"bank").mkdir()
    bank_rows, pool_sizes, seen = [], [], set()
    with (run/"bank/random9.jsonl").open("w", encoding="utf-8") as bank_out, \
         (run/"bank/pool_audit.jsonl").open("w", encoding="utf-8") as pool_out:
        for group in score_groups(scores):
            qid = group[0]["qa_id"]
            require(qid in parent_bank and qid not in seen, "unexpected score group")
            seen.add(qid)
            pairs = {(row["head_entity"], row["tail_entity"]) for row in group}
            require(len(pairs) == 1 and next(iter(pairs)) in known, "unverified train entity pair")
            row, audit = uniform_bank_row(group, relation_order=vocabulary,
                                          known_relations=known[next(iter(pairs))], seed=2026, k=9)
            require(parent_bank[qid]["positive_relation"] == row["positive_relation"], "parent gold changed")
            require(set(parent_bank[qid]["negative_relations"]).issubset(audit["legal_relations"]),
                    "parent bank does not use this legal pool")
            bank_out.write(json.dumps(row, ensure_ascii=False)+"\n")
            pool_out.write(json.dumps(audit, ensure_ascii=False)+"\n")
            bank_rows.append(row)
            pool_sizes.append(audit["pool_size"])
    require(seen == eligible, "score groups do not cover exactly eligible questions")
    manifest = {row["question_id"]: row for row in bank_rows}
    validate_frozen_random_manifest(manifest, k=9)
    ids = original_question_ids_from_payload(payload, len(texts))
    _, random_metas = build_qa_assets_from_task_texts(texts, seed=2026, listed_negative_source="frozen_random",
            relation_order=vocabulary, random_manifest=manifest, question_ids=ids, eligible_question_ids=eligible)
    _, parent_metas = build_qa_assets_from_task_texts(texts, seed=2026, listed_negative_source="structure_counterfactual",
            relation_order=vocabulary, counterfactual_manifest=parent_bank, question_ids=ids, eligible_question_ids=eligible)
    for parent, control in zip(parent_metas, random_metas):
        for key in ("question_id", "positive_relation", "prefix_text", "matched_train_relation",
                    "lambda_candidate", "candidate_eligible"):
            require(parent[key] == control[key], f"non-candidate QA metadata changed: {key}")
        require(bool(parent["listed_relations"]) == bool(control["listed_relations"]), "loss eligibility changed")
    wiring = {"n_qa": len(texts), "listed_qa": len(manifest), "k": 9, "seed": 2026,
              "vocabulary_size": len(vocabulary), "min_pool": min(pool_sizes), "max_pool": max(pool_sizes),
              "same_qa_text_and_generation_prefix": True, "same_candidate_eligibility": True,
              "same_parent_legal_pool": True, "train_only": True, "rows_failed": 0,
              "source_scores": str(scores.resolve()), "source_scores_sha256": sha256(scores)}
    write_json(run/"bank/wiring.json", wiring)
    for name in ("random9.jsonl", "pool_audit.jsonl", "wiring.json"):
        path = run/"bank"/name
        artifacts["bank/"+name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    normalize = normalization(code_root)
    samples = eval_samples(load_graph_record(run/"inputs/test.json"))
    require(len(samples) == design["test_count"], "test count changed")
    parent_predictions = audit_predictions(jsonl(run/"reference/predictions.jsonl"), samples, normalize)
    require(sum(r["correct"] for r in parent_predictions.values()) == evidence["selected_correct"],
            "parent predictions no longer reproduce the audited score")
    parent_cache = read_json(run/"reference/cache_identity.json")
    require(parent_cache["adapter_sha256"] == evidence["audit_identity"]["selected_weights"], "parent weight identity differs")
    model_cache = resolve_source(ref["paths"]["model_cache_dir"])
    cmd = training_command(str(PYTHON), code_root/REL_HNG/"scripts/train_listed_contrastive.py", ref,
                           inputs=run/"inputs", bank=run/"bank/random9.jsonl", output=run/"training", model_cache=model_cache)
    planned = dict(ref, listed_negative_source="frozen_random", negative_sampling_mode="random")
    write_csv(run/"paper/config_match.csv", parameter_diff(ref, planned))
    contract = {"schema": "fb-matched-random-v1", "experiment_id": design["experiment_id"],
                "repository_root": str(ROOT),
                "created_at": now(), "design_sha256": sha256(config_path), "design": design,
                "immutable_artifacts": artifacts, "training_argv": cmd, "model_cache": str(model_cache),
                "scope": design["scope"], "selection": "fixed_step32_before_random_validation_or_test",
                "allowed_differences": ["training_negative_identities", "frozen_random_loader", "checkpoint_retention"],
                "expected_scheduler_max_steps": parent_state["max_steps"],
                "expected_optimizer_steps": parent_state["global_step"]}
    write_json(run/"environment_prepare.json", environment(run))
    write_json(run/"contract.json", contract)
    write_json(run/"execution_plan.json", {"training_argv": cmd, "next": "check; train; full_validation; fixed_test; report"})
    write_json(run/"status.json", {"stage": "prepared", "status": "ready", "at": now(),
                                  "training_complete": False, "test_complete": False})
    append_event(run, {"stage": "prepare", "status": "completed", "wiring": wiring})
    return contract


def verify(run):
    contract = read_json(run/"contract.json")
    for relative, record in contract["immutable_artifacts"].items():
        path = run/relative
        require(path.is_file() and sha256(path) == record["sha256"], "frozen input/code/bank changed: "+relative)
    return contract


def refresh_code(run, contract):
    """Explicitly refreeze controller fixes before any training, retaining history."""
    require(not list((run/"training/trainer_listed").glob("checkpoint-*"))
            and not (run/"training/listed/adapter").exists() and not (run/"selection.json").exists(),
            "cannot refresh code after training or selection; use the existing frozen executable")
    changes = []
    for relative, record in contract["immutable_artifacts"].items():
        if not relative.startswith("code_snapshot/"):
            continue
        source = ROOT/Path(relative).relative_to("code_snapshot")
        require(source.is_file(), "source removed during prelaunch refresh: "+str(source))
        digest = sha256(source)
        if digest != record["sha256"]:
            changes.append({"path": relative, "before_sha256": record["sha256"], "after_sha256": digest})
            history = run/"prelaunch_code_versions"/record["sha256"]/Path(relative).name
            history.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run/relative, history)
            shutil.copy2(source, run/relative)
            record.update(sha256=digest, bytes=(run/relative).stat().st_size)
    check = run/"checks/cli.json"
    if changes and check.exists():
        check.rename(check.with_name("cli_before_refresh_"+str(time.time_ns())+".json"))
    contract["repository_root"] = str(ROOT)
    write_json(run/"contract.json", contract)
    append_event(run, {"stage": "prelaunch_code_refresh", "status": "completed", "changes": changes})
    return contract


def runtime_env(run):
    root = run/"code_snapshot"
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join([str(root), str(root/REL_BASE), str(root/REL_HNG/"src"),
                                               str(root/REL_HNG/"scripts")]),
                "GRIP_BASELINE_DIR": str(root/REL_BASE), "GRIP_REUSE_CACHE": str(run/"reuse_cache"),
                "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    return env


def execute(run, stage, argv):
    logs = run/"logs"
    logs.mkdir(exist_ok=True)
    attempt = len(list(logs.glob(stage+"_*.log")))+1
    log = logs/f"{stage}_{attempt:02d}.log"
    started = time.monotonic()
    event = {"stage": stage, "status": "running", "argv": argv, "log": str(log), "attempt": attempt}
    append_event(run, event)
    write_json(run/"environment_latest.json", environment(run))
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": now(), "argv": argv})+"\n")
        stream.flush()
        process = subprocess.Popen(argv, cwd=run/"code_snapshot", env=runtime_env(run),
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                result = process.poll()
                state = {"at": now(), "stage": stage, "status": "running" if result is None else "finished",
                         "pid": process.pid, "elapsed_seconds": round(time.monotonic()-started, 1), "log": str(log)}
                write_json(run/"status.json", state)
                with (run/"resources.jsonl").open("a", encoding="utf-8") as resources:
                    resources.write(json.dumps({**state, "free_disk_bytes": shutil.disk_usage(run).free,
                        "gpu": capture(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader"])})+"\n")
                if result is not None:
                    break
                time.sleep(20)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    append_event(run, {**event, "status": "completed" if result == 0 else "failed", "returncode": result,
                       "elapsed_seconds": round(time.monotonic()-started, 1)})
    require(result == 0, f"{stage} failed (exit {result}); see {log}")


def cli_check(run, contract):
    if (run/"checks/cli.json").exists():
        require(read_json(run/"checks/cli.json")["listed_qa"] == 3109, "invalid CLI preflight")
        return
    (run/"training").mkdir(exist_ok=True)
    # Parse the exact training command and resolve real assets without loading a model.
    program = (
        "import json,sys; from pathlib import Path; import train_listed_contrastive as t; "
        "from hard_negative_grip.matched_control import parameter_diff,read_json,write_json; "
        "argv=json.loads(Path(sys.argv[1]).read_text())['training_argv']; "
        "sys.argv=argv[1:]; args=t.parse_args(); assets=t.resolve_training_assets(args); "
        "t.write_run_config(Path(args.output_dir), args); "
        "ref=read_json(Path(args.output_dir).parent/'reference/run_config.json'); "
        "actual=read_json(Path(args.output_dir)/'run_config.json'); parameter_diff(ref,actual); "
        "write_json(Path(args.output_dir).parent/'checks/cli.json', "
        "{'n_qa':len(assets[2]),'listed_qa':sum(bool(r['listed_relations']) for r in assets[3]),'model_loaded':False})"
    )
    execute(run, "cli_preflight", [str(PYTHON), "-c", program, str(run/"execution_plan.json")])
    checked = read_json(run/"checks/cli.json")
    require(checked["n_qa"] == contract["design"]["qa_count"] and checked["listed_qa"] == 3109,
            "CLI wiring differs from prepared bank")


def validate_training(run, contract):
    trainer = run/"training/trainer_listed/checkpoint-32/trainer_state.json"
    state = read_json(trainer)
    require(state["global_step"] == contract["expected_optimizer_steps"] and
            state["max_steps"] == contract["expected_scheduler_max_steps"], "optimizer/scheduler budget mismatch")
    metadata = read_json(run/"training/listed/adapter/run_metadata.json")
    require(metadata["trainer_global_step"] == 32 and metadata["trainer_max_steps"] == 120,
            "adapter was not saved at the fixed training budget")
    actual = read_json(run/"training/run_config.json")
    reference = read_json(run/"reference/run_config.json")
    write_csv(run/"paper/config_match.csv", parameter_diff(reference, actual))
    require(actual["listed_negative_source"] == "frozen_random" and actual["negative_sampling_mode"] == "random",
            "trained the wrong negative source")
    write_json(run/"training/validated_budget.json", {"optimizer_steps": 32, "scheduler_max_steps": 120,
              "seed": 2026, "matched_training_fields": True, "at": now()})
    return state


def train(run, contract):
    cli_check(run, contract)
    if (run/"training/run_config.json").exists():
        parameter_diff(read_json(run/"reference/run_config.json"), read_json(run/"training/run_config.json"))
    if (run/"training/listed/adapter/run_metadata.json").exists():
        validate_training(run, contract)
        return
    checkpoints = list((run/"training/trainer_listed").glob("checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
            require((latest/name).is_file(), "incomplete resume checkpoint: "+str(latest/name))
        require(read_json(latest/"trainer_state.json")["global_step"] <= 32, "checkpoint exceeds fixed budget")
    execute(run, "train", contract["training_argv"])
    validate_training(run, contract)


def evaluate(run, contract):
    validate_training(run, contract)
    adapter = run/"training/listed/adapter"
    fingerprints = {name: sha256(adapter/name) for name in ("adapter_config.json", "adapter_model.safetensors")}
    selected = {"selected_at": now(), "selection_rule": contract["selection"], "step": 32,
                "seed": 2026, "adapter_sha256": fingerprints, "search_budget": 0,
                "validation_selects_checkpoint": False, "reopen_after_test": False}
    if (run/"selection.json").exists():
        require(read_json(run/"selection.json")["adapter_sha256"] == fingerprints, "selected weights changed")
    else:
        write_json(run/"selection.json", selected)
    for split in ("validation", "test"):
        out = run/"evaluation"/split
        execute(run, "eval_"+split, [str(PYTHON), str(run/"code_snapshot"/REL_HNG/"scripts/evaluate_adapter.py"),
                "--adapter", str(adapter), "--eval_file", str(run/"inputs"/(split+".json")),
                "--output_dir", str(out), "--model_name", "qwen-7b", "--model_cache_dir", contract["model_cache"],
                "--dtype", "bfloat16", "--gen_max_length", "32", "--progress_every", "100", "--resume",
                "--label", "fb15k237_2_random9_seed2026_fixed32_"+split])


def inventory(run):
    rows = []
    for directory in ("inputs", "reference", "bank", "code_snapshot", "training", "evaluation", "paper", "logs", "checks"):
        base = run/directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rows.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(run/"artifact_manifest.json", {"at": now(), "artifacts": rows,
               "policy": "No automatic cleanup; all four training checkpoints and optimizer states are retained."})


def report(run, contract):
    state = validate_training(run, contract)
    normalize = normalization(run/"code_snapshot")
    validation = audit_predictions(jsonl(run/"evaluation/validation/predictions.jsonl"),
                 eval_samples(load_graph_record(run/"inputs/validation.json")), normalize)
    samples = eval_samples(load_graph_record(run/"inputs/test.json"))
    method = audit_predictions(jsonl(run/"reference/predictions.jsonl"), samples, normalize)
    random_control = audit_predictions(jsonl(run/"evaluation/test/predictions.jsonl"), samples, normalize)
    selected = read_json(run/"selection.json")
    current_weights = {name: sha256(run/"training/listed/adapter"/name)
                       for name in ("adapter_config.json", "adapter_model.safetensors")}
    require(selected["adapter_sha256"] == current_weights, "selected adapter changed before reporting")
    for split, predictions in (("validation", validation), ("test", random_control)):
        identity = read_json(run/"evaluation"/split/"reuse_identity.json")
        require(identity["adapter"] == current_weights, "prediction cache belongs to different weights")
        summary = read_json(run/"evaluation"/split/"summary.json")["all"]
        measured = sum(r["correct"] for r in predictions.values())/len(predictions)
        require(summary["count"] == len(predictions) and abs(summary["em"]-measured) < 1e-12,
                "evaluation summary differs from recomputed predictions")
    result, pairs = paired_result(method, random_control)
    evidence = read_json(run/"reference/evidence.json")
    require(result["sccl_correct"] == evidence["selected_correct"], "parent metric changed")
    direction = "positive" if result["net_correct"] > 0 else "negative" if result["net_correct"] < 0 else "zero"
    result.update({"experiment_id": contract["experiment_id"], "at": now(), "dataset": "fb15k237_2", "seed": 2026,
        "random_validation_em": sum(r["correct"] for r in validation.values())/len(validation),
        "status": "completed", "claim_status": "inconclusive_for_stability",
        "observed_direction_sccl_minus_random9": direction, "comparability": "same fixed training configuration; sampling replaced",
        "limitations": ["Single Stage2 seed; no seed-level uncertainty or significance claim",
                        "Configuration and step were previously selected using SC-CL validation; this is a conditional ablation, not equal-HPO best-method comparison",
                        "Shared training QA contains the documented 464 train-pair repairs",
                        "Historical test exposure remains; this experiment cannot erase it",
                        "Random comparison does not isolate structure from score violation or relation frequency"],
        "paper_role": "claim-carrying", "section_id": "experiments", "item_id": "SC_CTRL_RANDOM",
        "claim_links": ["C2"], "target_display": "candidate-source ablation table", "main_or_appendix": "main",
        "next_action": "write the fixed-configuration single-seed comparison; do not reopen training or tune on test"})
    write_json(run/"paper/result.json", result)
    write_csv(run/"paper/paired_predictions.csv", pairs)
    metrics = [
        {"model": "SC-CL-9 (reference)", "seed": 2026, "n": result["n"], "correct": result["sccl_correct"], "em_percent": 100*result["sccl_em"]},
        {"model": "Random-9 + InfoNCE", "seed": 2026, "n": result["n"], "correct": result["random9_correct"], "em_percent": 100*result["random9_em"]},
    ]
    write_csv(run/"paper/metrics.csv", metrics)
    # Keep every loss field, optimizer step and learning rate available to plots.
    history = state["log_history"]
    fields = sorted({key for row in history for key in row})
    write_csv(run/"paper/training_history.csv", [{key: row.get(key, "") for key in fields} for row in history])
    (run/"paper/table.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\nMethod & Seed & Correct / n & EM (\\%) \\\\\n\\midrule\n"
        + f"SC-CL-9 & 2026 & {result['sccl_correct']} / {result['n']} & {100*result['sccl_em']:.4f} \\\\\n"
        + f"Random-9 + InfoNCE & 2026 & {result['random9_correct']} / {result['n']} & {100*result['random9_em']:.4f} \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    (run/"paper/RESULTS.md").write_text(
        f"# FB15K237-2 固定配置单种子候选来源对照\n\n"
        f"SC-CL-9：{100*result['sccl_em']:.4f}%；Random-9：{100*result['random9_em']:.4f}%；"
        f"SC-CL减Random为{result['delta_sccl_minus_random9_pp']:+.4f}个百分点，净{result['net_correct']:+d}题。\n\n"
        "除候选来源外匹配Stage1、QA、过滤、K、LoRA、优化器和调度；实际训练32步，调度长度120步。"
        "仅seed2026，未估计跨seed波动。所用配置此前按SC-CL验证集选定，因此本结果为该配置条件下的采样替换，"
        "不代表双方等预算调参后的最优性能，也不能单独证明结构分量。共享训练任务包含464条train-pair修复。\n\n"
        f"逐题变化：{json.dumps(result['paired_counts'],ensure_ascii=False)}。负或零差异同样保留。\n", encoding="utf-8")
    inventory(run)
    destination = Path(contract["repository_root"])/"11_paper/artifacts/fb_random9_control_seed2026"
    if destination.exists() and (destination/"result.json").exists():
        require(read_json(destination/"result.json")["experiment_id"] == contract["experiment_id"],
                "paper export belongs to another experiment")
    destination.mkdir(parents=True, exist_ok=True)
    for source in (run/"paper").iterdir():
        if source.is_file():
            shutil.copy2(source, destination/source.name)
    write_json(destination/"evidence_index.json", {"experiment_id": contract["experiment_id"],
               "run_dir": str(run), "contract_sha256": sha256(run/"contract.json"),
               "artifact_manifest_sha256": sha256(run/"artifact_manifest.json"),
               "raw_predictions": {"random": str(run/"evaluation/test/predictions.jsonl"),
                                   "sccl": str(run/"reference/predictions.jsonl")},
               "weights_and_optimizer_states_retained": True})
    write_json(run/"status.json", {"stage": "completed", "status": "completed", "at": now(),
                                   "training_complete": True, "test_complete": True, "result": result})
    append_event(run, {"stage": "report", "status": "completed", "result": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refresh-code", "check", "train", "evaluate", "report", "run"), nargs="?", default="prepare")
    parser.add_argument("--config", type=Path, default=HNG/"configs/fb_random9_matched.json")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    design = read_json(args.config)
    run = (args.run_dir or Path(design["default_run_dir"])).resolve()
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "use the project guardenv Python")
    with locked(run):
        try:
            contract = prepare(run, args.config) if args.action in ("prepare", "run") else verify(run)
            print(json.dumps({"action": args.action, "run_dir": str(run), "experiment_id": contract["experiment_id"]}), flush=True)
            if args.action == "refresh-code":
                refresh_code(run, contract)
            if args.action in ("check", "run"):
                cli_check(run, contract)
            if args.action in ("train", "run"):
                train(run, contract)
            if args.action in ("evaluate", "run"):
                evaluate(run, contract)
            if args.action in ("report", "run"):
                result = report(run, contract)
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            append_event(run, {"stage": args.action, "status": "failed", "error": repr(error)})
            write_json(run/"status.json", {"stage": args.action, "status": "failed", "at": now(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
