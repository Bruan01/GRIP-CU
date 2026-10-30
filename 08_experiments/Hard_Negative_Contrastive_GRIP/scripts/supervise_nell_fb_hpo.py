#!/usr/bin/env python3
"""Durable, single-GPU NELL follow-up monitoring and matched FB HPO queue.

Run in tmux with guardenv. Search decisions use validation EM; the selected
checkpoint is never forced through a longer schedule before test evaluation.
Existing NELL runs are observed without changing their processes or artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import optuna_direct_em_search as hpo

PYTHON = hpo.PYTHON_DEFAULT
HNG, REPO, BASELINE = hpo.HNG, hpo.REPO, hpo.BASELINE
EXISTING = HNG / "results/hpo/fb_nell_direct_em_20261027"
DEFAULT_ROOT = HNG / "results/hpo/nell_fb_guard_20261029"
NELLEXT = EXISTING / "nell23k/followups/trial0_step12_extension"
NELL_SESSION = "nell_step12_extension"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def env():
    result = os.environ.copy()
    result.update(PYTHON=str(PYTHON), GRIP_BASELINE_DIR=str(BASELINE),
                  PYTHONPATH=os.pathsep.join([str(BASELINE), str(HNG / "src"),
                                             str(HNG / "scripts"), result.get("PYTHONPATH", "")]),
                  TOKENIZERS_PARALLELISM="false", GRIP_TORCH_DTYPE="bfloat16",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return result


def adapter_ready(path):
    return all((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("adapter_config.json", "adapter_model.safetensors"))


def fingerprint(adapter):
    return {name: hpo.sha256_file(adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")}


def evaluate(adapter, source, destination, label):
    """A complete cache must match immutable weights, data, and row count."""
    identity = {"adapter": str(adapter), "adapter_sha256": fingerprint(adapter),
                "eval_file": str(source), "eval_sha256": hpo.sha256_file(source),
                "gen_max_length": 32, "dtype": "bfloat16", "label": label}
    destination.mkdir(parents=True, exist_ok=True)
    identity_file = destination / "cache_identity.json"
    expected = len(read(source)["questions"])
    if identity_file.exists() and read(identity_file) != identity:
        raise ValueError(f"Evaluation cache identity changed: {destination}")
    hpo.write_json(identity_file, identity)
    summary_path, predictions = destination / "summary.json", destination / "predictions.jsonl"
    if summary_path.exists():
        summary = read(summary_path)
        count = 0
        if predictions.exists():
            with predictions.open(encoding="utf-8") as stream:
                count = sum(1 for _ in stream)
        if (summary.get("all", {}).get("count") != expected or count != expected
                or summary.get("adapter") != str(adapter)
                or summary.get("eval_file") != str(source)):
            raise ValueError(f"Incomplete or mismatched evaluation: {destination}")
        return float(summary["all"]["em"])
    hpo.run_logged([str(PYTHON), str(hpo.EVALUATOR), "--adapter", str(adapter),
                    "--eval_file", str(source), "--output_dir", str(destination),
                    "--model_name", "qwen-7b", "--model_cache_dir", str(hpo.MODEL_CACHE_DEFAULT),
                    "--dtype", "bfloat16", "--gen_max_length", "32", "--progress_every", "100",
                    "--resume", "--label", label], env(), destination / "evaluation.log", HNG)
    return evaluate(adapter, source, destination, label)


def fb_paths(root):
    work = root / "fb15k237_2"
    return work, work / "b1/stage1_adapter", work / "b1/stage2_adapter", work / "assets/k9"


def prepare_fb_inputs(root):
    """Reproduce released FB entity text and certify train-pair recovery on CPU."""
    sys.path.insert(0, str(HNG / "src"))
    from hard_negative_grip.official_lists import load_train_relation_order
    from hard_negative_grip.offline_scoring import iter_matchable_relation_qa
    from hard_negative_grip.task_file import known_pair_relations, train_relation_alias_index

    work, _, _, _ = fb_paths(root)
    paths = hpo.dataset_paths("fb15k237_2")
    raw = Path(paths["raw"])
    mapping_source = raw / "entity2wikidata.json"
    metadata = read(mapping_source)
    train_entities = set()
    for line in (raw / "train.txt").read_text(encoding="utf-8").splitlines():
        head, _, tail = line.split()
        train_entities.update((head, tail))
    mapping = {}
    for entity in sorted(train_entities):
        item = metadata.get(entity)
        if item is None:
            text = "Unknown"
        else:
            text = (str(item["label"]) + ", " if item.get("label") is not None else "")
            if item.get("description") is not None:
                text += str(item["description"])
            text = (text or "missing").strip().strip(".")
        if any(char in text for char in "\t\n\r"):
            raise ValueError(f"FB entity text cannot be represented as TSV: {entity}")
        mapping[entity] = text
    mapping_file = work / "assets/entity2text_train.txt"
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{entity}\t{label}\n" for entity, label in mapping.items())
    if mapping_file.exists() and mapping_file.read_text(encoding="utf-8") != text:
        raise ValueError(f"Train entity mapping changed: {mapping_file}")
    mapping_file.write_text(text, encoding="utf-8")
    relations = load_train_relation_order(raw, relation_text_file=raw / "rel2text.json")
    known = known_pair_relations(raw, splits=("train",), relation_text_file=raw / "rel2text.json",
                                 entity_text_file=mapping_file)
    pairs = {(head.casefold(), tail.casefold()): (head, tail) for head, tail in known}
    task = read(paths["input"])
    task_source_hash = hpo.sha256_file(Path(paths["input"]))
    items = list(iter_matchable_relation_qa(task["qa_samples"], train_relation_alias_index(relations),
                                           entity_pairs=pairs))
    # A rewritten question with no verified train pair must never disable the
    # known-fact filter. Regenerate that QA from a real train edge with the same
    # gold relation. This is a new training example, not an inferred recovery of
    # the old question's intended entities. Both B1 and SC-CL use this same file.
    relation_pairs = {}
    for pair, labels in known.items():
        for label in labels:
            relation_pairs.setdefault(label, []).append(pair)
    for label in relation_pairs:
        relation_pairs[label].sort()
    repairs = []
    for item in items:
        if item["matched_train_relation"] in known.get(item["entity_pair"], set()):
            continue
        index = int(item["qa_id"].split(":")[-1])
        gold = item["matched_train_relation"]
        pool = relation_pairs[gold]
        number = int(hashlib.sha256(f"2026|{item['qa_id']}|{gold}".encode()).hexdigest(), 16)
        pair = pool[number % len(pool)]
        question = f"What is the relation between word node {pair[0]} and word node {pair[1]}?"
        original = task["qa_samples"][index]
        match = re.search(r"(please answer the following question:\s*)(.*?)(\s+Response in the following format:)",
                          original, flags=re.I | re.S)
        if match is None:
            raise ValueError(f"Cannot safely replace relation QA question: {item['qa_id']}")
        repaired = original[:match.start(2)] + question + original[match.end(2):]
        task["qa_samples"][index] = repaired
        repairs.append({"qa_id": item["qa_id"], "old_question": item["question"],
                        "new_question": question, "gold_relation": gold, "train_entity_pair": pair,
                        "old_text_sha256": hashlib.sha256(original.encode()).hexdigest(),
                        "new_text_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
                        "operation": "regenerate question from deterministic same-relation train edge"})
    derived = work / "assets/grip_fb15k237_2_tasks_train_pair_repaired.json"
    task["train_pair_repair"] = {"source": str(paths["input"]), "source_sha256": task_source_hash,
        "seed": 2026, "regenerated_qa_count": len(repairs),
        "rule": "SHA256(seed|qa_id|gold) selects from sorted canonical train pairs of the same gold relation",
        "original_entities_recovered": False, "valid_test_used": False}
    if not derived.exists():
        hpo.write_json(derived, task)
    else:
        existing = read(derived)
        # DrvFS accepts both /mnt/c/users/... and /mnt/c/Users/... for this
        # workspace. A spelling change must not invalidate identical inputs.
        old_source = existing.get("train_pair_repair", {}).get("source", "")
        if old_source and os.path.samefile(old_source, paths["input"]):
            task["train_pair_repair"]["source"] = old_source
        if existing != task:
            raise ValueError(f"Repaired FB training data changed: {derived}")
    hpo.write_json(work / "task_repairs.json", repairs)
    items = list(iter_matchable_relation_qa(task["qa_samples"], train_relation_alias_index(relations),
                                           entity_pairs=pairs))
    recovered = sum(item["entity_pair"] is not None for item in items)
    supported = sum(item["matched_train_relation"] in known.get(item["entity_pair"], set()) for item in items)
    audit = {"time": now(), "claim_type": "computed", "relation_qa": len(items),
             "train_pairs_recovered": recovered, "gold_is_known_train_fact": supported,
             "passed": len(items) == 3109 and recovered == len(items) and supported == len(items),
             "entity_text_file": str(mapping_file), "mapping_sha256": hpo.sha256_file(mapping_file),
             "mapping_source": str(mapping_source), "mapping_source_sha256": hpo.sha256_file(mapping_source),
             "mapping_rule": "released FB processor: label + comma + description; strip whitespace and periods",
             "entity_selection": "train endpoints only", "context_samples": len(task["context_samples"]),
             "qa_samples": len(task["qa_samples"]), "training_input": str(paths["input"]),
             "original_task_sha256": task_source_hash, "derived_training_input": str(derived),
             "task_sha256": hpo.sha256_file(derived), "regenerated_qa_count": len(repairs),
             "repair_log": str(work / "task_repairs.json"), "valid_test_used_for_repair": False,
             "train_sha256": hpo.sha256_file(raw / "train.txt")}
    hpo.write_json(work / "input_audit.json", audit)
    if not audit["passed"]:
        raise ValueError(f"FB train-only pair audit failed: {audit}")
    return mapping_file


def plan(root):
    work, s1, b1, bank = fb_paths(root)
    return {
        "created_at": now(), "python": str(PYTHON), "model": "Qwen2.5-7B-Instruct",
        "gpu_policy": "one GPU job at a time; wait for existing NELL tmux chain and GPU processes",
        "nell": {"observed_session": NELL_SESSION, "followup": str(NELLEXT),
                 "paper_em_reference": 0.8774, "change_existing_training": False},
        "fb": {"s1_adapter": str(s1), "b1_adapter": str(b1), "bank": str(bank),
               "input_audit": str(work / "input_audit.json"),
               "paper_em_reference": 0.8875, "stages": ["matched_b1", "train_scores", "freeze",
                 "wire", "validation_search", "full_validation_selection", "selected_test"],
               "b1_training": {"rank": 24, "alpha": 48, "stage1_epochs": 1,
                 "stage2_epochs": 5, "microbatch": 1, "accumulation": 512, "seed": 2026},
               "hpo": {"trials": 6, "rungs": [8, 16, 32], "screen_size": 1000,
                 "prune_earliest_step": 16, "full_validation_finalists": 2,
                 "selection": "full validation EM including matched B1; retain checkpoint unchanged",
                 "no_go": "do not evaluate SC-CL test if full-validation EM does not beat B1"},
               "test_protocol": "existing fixed 10000-question local test; compare paper reference with scope caveat"},
        "prior_test_exposure": "NELL test outcomes were inspected before this campaign; further NELL exploration is adaptive.",
        "paper_source": "https://arxiv.org/html/2511.07457v2",
        "failure_policy": "record failure and stop that stage; restart launcher to resume completed stages/checkpoints",
        "monitor": {"interval_seconds": 60, "stale_log_warning_seconds": 7200,
                    "files": ["status.json", "STATUS.md", "events.jsonl", "supervisor.log"]},
    }


class Supervisor:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "supervisor.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.action, self.child = "starting", None

    def event(self, kind, **values):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **values}, ensure_ascii=False) + "\n")

    def snapshot(self, **values):
        active = probe(["ps", "-eo", "pid,ppid,etime,pcpu,args"])
        active["stdout"] = "\n".join(line for line in active.get("stdout", "").splitlines()
                                          if str(PYTHON) in line or NELL_SESSION in line)
        evidence = {}
        for label, path in {
            "nell_step8_test": EXISTING / "nell23k/followups/trial0_step8_replay/test/summary.json",
            "nell_step12_decision": NELLEXT / "validation_decision.json",
            "nell_step12_test": NELLEXT / "final_result.json",
            "fb_selection": self.root / "fb15k237_2/search/selection.json",
            "fb_result": self.root / "fb15k237_2/search/final_result.json",
            "fb_input_audit": self.root / "fb15k237_2/input_audit.json",
            "fb_no_go": self.root / "fb15k237_2/search/no_go.json",
        }.items():
            if path.exists():
                try:
                    evidence[label] = {"path": str(path), "data": read(path), "provenance": "parsed"}
                except (OSError, ValueError) as error:
                    evidence[label] = {"error": str(error)}
        logs = list(NELLEXT.rglob("*.log")) + list(self.root.rglob("*.log"))
        logs = [p for p in logs if p.name != "supervisor.log"]
        latest = max(logs, key=lambda p: p.stat().st_mtime, default=None)
        age = time.time() - latest.stat().st_mtime if latest else None
        progress = None
        if latest:
            with latest.open("rb") as stream:
                stream.seek(max(0, latest.stat().st_size - 12000))
                tail = stream.read().decode("utf-8", errors="replace")
            matches = re.findall(r"\|\s*(\d+)/(\d+)", tail)
            if matches:
                step, horizon = matches[-1]
                progress = {"optimizer_step": int(step), "schedule_horizon_steps": int(horizon)}
        gpu = probe(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"])
        state = {"updated_at": now(), "supervisor_pid": os.getpid(), "action": self.action,
                 "child_pid": self.child.pid if self.child and self.child.poll() is None else None,
                 "gpu": gpu, "processes": active, "evidence": evidence,
                 "latest_log": str(latest) if latest else None, "latest_log_age_seconds": age,
                 "latest_training_progress": progress,
                 "stale_log_warning": age is not None and age > 7200,
                 "host_reboot_survival": False, **values}
        hpo.write_json(self.root / "status.json", state)
        text = f"# NELL / FB HPO monitor\n\nUpdated: {state['updated_at']}\n\nAction: `{self.action}`\n\n"
        text += f"Supervisor PID: {os.getpid()}; child: {state['child_pid']}\n\nGPU: {gpu.get('stdout', gpu.get('error', gpu.get('stderr')))}\n\n"
        text += f"Latest log: `{latest}`; age: {round(age) if age is not None else '?'} seconds\n\n"
        if progress:
            text += f"Latest optimizer step: {progress['optimizer_step']} (schedule horizon {progress['schedule_horizon_steps']}).\n\n"
        if state["stale_log_warning"]:
            text += "WARNING: no log update in two hours; inspect the active process. It has not been killed.\n\n"
        for label, item in evidence.items():
            text += f"## {label}\n\n```json\n{json.dumps(item, indent=2, ensure_ascii=False)}\n```\n\n"
        text += "Search uses validation EM. Paper references: NELL 87.74%, FB 88.75% (arXiv v2).\n\n"
        text += "FB uses the existing fixed 10,000-question local test protocol. NELL follow-ups are adaptive after prior test inspection.\n"
        (self.root / "STATUS.md").write_text(text, encoding="utf-8")
        return state

    def wait_for_nell(self):
        self.action = "waiting_for_existing_nell_step12_train_validation_test_chain"
        while True:
            check = probe(["tmux", "has-session", "-t", f"={NELL_SESSION}"])
            if check.get("returncode") != 0:
                break
            self.snapshot()
            time.sleep(60)
        if not (NELLEXT / "validation_decision.json").exists():
            self.event("nell_chain_ended_without_validation_decision", followup=str(NELLEXT))
        else:
            self.event("nell_chain_completed", decision=read(NELLEXT / "validation_decision.json"))
        # Any remaining GPU process may be another authorized job. Wait, never kill it.
        self.action = "waiting_for_gpu_idle_before_fb"
        while True:
            check = probe(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
            if check.get("returncode") != 0:
                raise RuntimeError(f"Cannot inspect GPU availability: {check}")
            if not check.get("stdout", "").strip():
                break
            self.snapshot(waiting_for_gpu_processes=check["stdout"])
            time.sleep(60)

    def run(self, name, command, command_env=None):
        self.action = name
        log_path = self.root / "logs" / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)
        self.event("stage_start", stage=name, command=command, log=str(log_path))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{now()}] {json.dumps(command)}\n")
            stream.flush()
            self.child = subprocess.Popen(command, cwd=BASELINE, env=command_env or env(),
                                          stdin=subprocess.DEVNULL, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True,
                                          pass_fds=(self.lock.fileno(),))
            while self.child.poll() is None:
                self.snapshot(log=str(log_path))
                try:
                    self.child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            code = self.child.returncode
        self.event("stage_finish", stage=name, returncode=code, log=str(log_path))
        self.snapshot(returncode=code, log=str(log_path))
        if code:
            raise RuntimeError(f"{name} failed with exit {code}; inspect {log_path}; restart launcher to resume")

    def fb_pipeline(self):
        work, s1, b1, bank = fb_paths(self.root)
        paths = hpo.dataset_paths("fb15k237_2")
        for key in ("input", "valid", "test"):
            if not Path(paths[key]).is_file():
                raise FileNotFoundError(paths[key])
        mapping_file = prepare_fb_inputs(self.root)
        paths["input"] = work / "assets/grip_fb15k237_2_tasks_train_pair_repaired.json"
        if not (adapter_ready(s1) and adapter_ready(b1)):
            self.run("fb_matched_b1", [str(PYTHON), str(BASELINE / "scripts/train_task_file_adapter.py"),
                "--task_file", str(paths["input"]), "--output_dir", str(work / "b1"),
                "--model_name", "qwen-7b", "--model_source", "local", "--model_cache_dir", str(hpo.MODEL_CACHE_DEFAULT),
                "--local_files_only", "--dtype", "bfloat16", "--tokenize_max_length", "4096",
                "--stage1_epochs", "1", "--stage2_epochs", "5", "--per_device_train_batch_size", "1",
                "--gradient_accumulation_steps", "512", "--learning_rate", "1e-3", "--weight_decay", "1e-4",
                "--adam_epsilon", "1e-8", "--lora_r", "24", "--lora_alpha", "48", "--target_modules",
                "down_proj", "up_proj", "gate_proj", "--s1_stop_loss_threshold", "0.15",
                "--s2_stop_loss_threshold", "0.15", "--s1_min_epoch", "1", "--s2_min_epoch", "1",
                "--gradient_checkpointing", "--stage1_checkpoint_interval", "10",
                "--stage2_checkpoint_interval", "10", "--seed", "2026"])
        if not (adapter_ready(s1) and adapter_ready(b1)):
            raise RuntimeError("FB baseline training did not produce both complete adapters")
        scores = work / "assets/b1_train_scores"
        stage_env = env()
        stage_env.update(DATASET="fb15k237_2", NEGATIVE_SOURCE="structure_counterfactual", SKIP_TMUX="1",
            INPUT_FILE=str(paths["input"]), EVAL_FILE=str(paths["valid"]), RAW_DIR=str(paths["raw"]),
            RELATION_TEXT_FILE=str(Path(paths["raw"]) / "rel2text.json"), S1_ADAPTER=str(s1), B1_ADAPTER=str(b1),
            ENTITY_TEXT_FILE=str(mapping_file),
            MODEL_CACHE_DIR=str(hpo.MODEL_CACHE_DEFAULT), ARTIFACT_ROOT=str(work / "assets"),
            SCORES_DIR=str(scores), SAMPLER_DIR=str(bank), MANIFEST=str(bank / "counterfactual_bank.jsonl"),
            COUNTERFACTUAL_METADATA=str(bank / "metadata.json"), EXPECTED_LISTED="3109", LISTED_NEGATIVE_K="9",
            CANDIDATE_BATCH_SIZE="4", SEED="2026", TORCH_DTYPE="bfloat16")
        for stage, artifact in [("score", scores / "run_stats.json"),
                                ("freeze", bank / "metadata.json"), ("wire", bank / "wiring.json")]:
            if artifact.exists():
                continue
            stage_env.update(STEP=stage, RUN_ID=f"fb_guard_20261029_{stage}", RUN_DIR=str(work / stage))
            self.run(f"fb_{stage}", ["bash", str(hpo.RUNNER)], stage_env.copy())
        if not (work / "search/final_result.json").exists() and not (work / "search/no_go.json").exists():
            self.run("fb_validation_hpo", [str(PYTHON), "-u", str(Path(__file__).resolve()),
                                          "--root", str(self.root), "--worker-fb-search"])


def fb_search(root, spec=None):
    """Bounded search with protected controls, delayed pruning and frozen finalists."""
    import optuna

    if spec is None:
        work, s1, b1, bank = fb_paths(root)
        out = work / "search"
        paths = hpo.dataset_paths("fb15k237_2")
        paths["input"] = work / "assets/grip_fb15k237_2_tasks_train_pair_repaired.json"
        paths["entity_text_file"] = work / "assets/entity2text_train.txt"
        paths["epochs"] = 5
        dataset, prefix, paper_reference = "fb15k237_2", "fb", 0.8875
        rungs, trial_budget, search_only = (8, 16, 32), 6, False
        prune_after, screen_size = 16, 1000
        resource_cap = rungs[-1]
    else:
        dataset = spec["dataset"]
        prefix = dataset
        s1, b1, bank, out = (Path(spec[key]) for key in ["s1_adapter", "b1_adapter", "sampler", "search_dir"])
        paths = {key: Path(spec[key]) for key in ["input", "valid", "test", "raw"]}
        paths.update(epochs=spec["epochs"], expected_listed=spec["expected_listed"])
        for key in ["entity_text_file", "relation_eligibility"]:
            if spec.get(key): paths[key] = Path(spec[key])
        paper_reference = float(spec["paper_reference"])
        rungs = tuple(spec["rungs"])
        trial_budget = int(spec.get("trial_budget", 12))
        search_only = bool(spec.get("search_only", False))
        prune_after, screen_size = int(spec.get("prune_after", 32)), int(spec.get("screen_size", 1000))
        resource_cap = int(spec.get("resource_cap", rungs[-1]))
        if resource_cap < rungs[0]:
            raise ValueError("Resource cap is below the first Stage-2 rung")
    out.mkdir(parents=True, exist_ok=True)
    paths["b1_adapter"] = b1
    paths["save_steps"] = 4
    hpo.validate_assets(dataset, paths, s1, bank, PYTHON, hpo.MODEL_CACHE_DEFAULT)
    stage1_identity = fingerprint(s1)
    frozen_identity = out / "fixed_stage1_identity.json"
    if frozen_identity.exists() and read(frozen_identity) != stage1_identity:
        raise ValueError("Fixed Stage-1 weights changed during Stage-2 search")
    hpo.write_json(frozen_identity, stage1_identity)
    screen = out / f"valid_screen_{screen_size}.json"
    hpo.prepare_validation_screen(Path(paths["valid"]), screen, screen_size, 2026)
    baseline_screen = evaluate(b1, screen, out / "b1_screen", f"{prefix}_b1_validation_screen")
    study = optuna.create_study(study_name=f"{prefix}_sccl_validation_checkpoint_v1",
        storage=f"sqlite:///{out / 'study.sqlite3'}", direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=3),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=prune_after, reduction_factor=2))
    if not study.trials:
        study.enqueue_trial({"lambda_candidate": 1.0, "temperature": 1.0,
                             "learning_rate": 1e-3, "weight_decay": 1e-4})
        study.enqueue_trial({"lambda_candidate": 0.11229873660012951, "temperature": 0.8863961214238169,
                             "learning_rate": 0.000947622749310518, "weight_decay": 8.889902164812924e-05})
    # A fresh process must not replay the same startup random proposals while
    # earlier trials are still RUNNING at a temporary resource cap. Model
    # training always uses seed 2026; this separate search RNG seed is recorded.
    sampler_seed = 2026
    if spec is not None:
        sampler_seed += len(study.trials)
        study.sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=3)

    def execute_trial(trial):
        params = {"lambda_candidate": trial.suggest_float("lambda_candidate", 0.05, 2.0, log=True),
                  "temperature": trial.suggest_float("temperature", 0.5, 2.0, log=True),
                  "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3)}
        folder = out / "trials" / f"trial_{trial.number:04d}"
        hpo.write_json(folder / "trial_config.json", {"params": params, "trial": trial.number,
            "s1_adapter": str(s1), "b1_reference": str(b1), "epochs": paths["epochs"], "rungs": list(rungs),
            "search_stage": 2, "stage1_hyperparameters_searched": False,
            "actual_lora": read(s1 / "adapter_config.json"), "seed": 2026})
        best_em = -1.0
        for rung in (value for value in rungs if value <= resource_cap):
            frozen = folder / f"step_{rung:03d}/adapter"
            metrics_path = frozen.parent / "metrics.json"
            if not metrics_path.exists():
                if not frozen.exists():
                    hpo.write_json(out / "live_progress.json", {"dataset": dataset, "trial": trial.number,
                        "rung": rung, "phase": "stage2_training", "search_stage": 2,
                        "log": str(folder / f"train_rung_{rung}.log")})
                    checkpoints = list((folder / "train_run/trainer_listed").glob("checkpoint-*"))
                    resume = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[1]), default=None)
                    if resume is not None and int(resume.name.rsplit("-", 1)[1]) > rung:
                        raise RuntimeError(f"Cannot reconstruct missing step {rung} from later checkpoint {resume}")
                    hpo.train_rung(python=PYTHON, dataset=dataset, paths=paths, adapter=s1, sampler=bank,
                        model_cache=hpo.MODEL_CACHE_DEFAULT, trial_dir=folder, trial_number=trial.number,
                        rung_step=rung, fresh=resume is None, resume_checkpoint=resume, params=params)
                    live = folder / "train_run/listed/adapter"
                    if not adapter_ready(live):
                        raise RuntimeError(f"Missing trained adapter {live}")
                    latest = max((folder / "train_run/trainer_listed").glob("checkpoint-*"),
                                 key=lambda p: int(p.name.rsplit("-", 1)[1]))
                    if int(read(latest / "trainer_state.json")["global_step"]) != rung:
                        raise RuntimeError(f"Trainer did not stop at requested step {rung}: {latest}")
                    partial = frozen.with_name("adapter.copying")
                    shutil.copytree(live, partial, dirs_exist_ok=True)
                    partial.rename(frozen)
                if not adapter_ready(frozen):
                    raise RuntimeError(f"Incomplete frozen adapter: {frozen}")
                hpo.write_json(out / "live_progress.json", {"dataset": dataset, "trial": trial.number,
                    "rung": rung, "phase": "validation", "search_stage": 2,
                    "log": str(frozen.parent / "validation_screen/evaluation.log")})
                em = evaluate(frozen, screen, frozen.parent / "validation_screen", f"{prefix}_trial{trial.number}_step{rung}")
                hpo.write_json(metrics_path, {"trial": trial.number, "step": rung, "screen_em": em,
                    "adapter": str(frozen), "adapter_sha256": fingerprint(frozen), "params": params,
                    "b1_screen_em": baseline_screen, "selected_on": f"validation_screen_{screen_size}"})
            metric = read(metrics_path)
            em = float(metric["screen_em"])
            if em > best_em:
                best_em = em
                trial.set_user_attr("best_checkpoint", str(frozen))
                trial.set_user_attr("best_step", rung)
            new_report = rung not in study.get_trials()[trial.number].intermediate_values
            if new_report:
                trial.report(em, rung)
            hpo.write_study_exports(study, out)
            print(f"[{dataset}-hpo] trial={trial.number} step={rung} screen_em={em:.6f} b1={baseline_screen:.6f}", flush=True)
            if new_report and rung < rungs[-1] and trial.should_prune():
                raise optuna.TrialPruned(f"validation pruning at step {rung}")
        return best_em if resource_cap >= rungs[-1] else None

    # Recover the exact RUNNING trial after a supervisor restart rather than
    # abandoning it and silently spending the budget on another configuration.
    for frozen_trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        trial = optuna.trial.Trial(study, frozen_trial._trial_id)
        try:
            value = execute_trial(trial)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            if value is None:
                continue
            study.tell(trial, value)
    for _ in range(max(0, trial_budget - sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials))):
        # ask/tell leaves an interrupted trial RUNNING for checkpoint recovery.
        trial = study.ask()
        try:
            value = execute_trial(trial)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            if value is None:
                continue
            study.tell(trial, value)
        hpo.write_study_exports(study, out)

    if fingerprint(s1) != stage1_identity:
        raise ValueError("Stage-1 weights changed during Stage-2 execution")
    hpo.write_json(out / "search_progress.json", {"dataset": dataset, "trial_budget": trial_budget,
        "trials_recorded": len(study.trials), "search_stage": 2, "stage1_unchanged": True,
        "resource_cap": resource_cap,
        "optuna_sampling_seed": sampler_seed,
        "seed": 2026, "test_evaluated": False})
    if search_only:
        return

    metrics = [read(p) for p in (out / "trials").glob("trial_*/step_*/metrics.json")]
    finalists = sorted(metrics, key=lambda row: (-row["screen_em"], row["trial"], row["step"]))[:2]
    baseline_full = evaluate(b1, Path(paths["valid"]), out / "b1_full_validation", f"{prefix}_b1_full_validation")
    for candidate in finalists:
        frozen = Path(candidate["adapter"])
        hpo.write_json(out / "live_progress.json", {"dataset": dataset, "phase": "full_validation",
            "trial": candidate["trial"], "rung": candidate["step"],
            "log": str(frozen.parent / "full_validation/evaluation.log")})
        candidate["full_validation_em"] = evaluate(frozen, Path(paths["valid"]),
            frozen.parent / "full_validation", f"{prefix}_trial{candidate['trial']}_step{candidate['step']}_full_validation")
    winner = max(finalists, key=lambda row: row["full_validation_em"])
    selection = {"selected_at": now(), "selection_split": "full_validation", "finalists": finalists,
                 "b1_full_validation_em": baseline_full, "winner": winner,
                 "promote": winner["full_validation_em"] > baseline_full,
                 "test_used_in_this_search_selection": False, "prior_nell_test_exposure": True}
    hpo.write_json(out / "selection.json", selection)
    if not selection["promote"]:
        hpo.write_json(out / "no_go.json", {**selection, "reason": "No full-validation improvement over matched B1"})
        return
    # Freeze the decision before opening either test result. One selected SC-CL
    # checkpoint is evaluated, with exactly the same decoder as the B1 control.
    baseline_test = evaluate(b1, Path(paths["test"]), out / "b1_test", f"{prefix}_matched_b1_test")
    hpo.write_json(out / "live_progress.json", {"dataset": dataset, "phase": "selected_full_test",
        "trial": winner["trial"], "rung": winner["step"], "log": str(out / "selected_test/evaluation.log")})
    winner_test = evaluate(Path(winner["adapter"]), Path(paths["test"]), out / "selected_test", f"{prefix}_selected_checkpoint_test")
    hpo.write_json(out / "final_result.json", {"dataset": dataset, "claim_type": "computed",
        "winner": winner, "b1_test_em": baseline_test, "selected_test_em": winner_test,
        "paper_reference_em": paper_reference, "delta_vs_paper_pp": (winner_test - paper_reference) * 100,
        "exceeds_paper_reference": winner_test > paper_reference,
        "improves_matched_baseline": winner_test > baseline_test,
        "target_achieved": winner_test > max(paper_reference, baseline_test),
        "test_count": len(read(Path(paths["test"]))["questions"]),
        "seed": 2026, "search_stage": 2,
        "scope": "fixed local protocol; single seed; paper-comparability requires protocol audit",
        "prediction_file": str(out / "selected_test/predictions.jsonl"),
        "selection_file": str(out / "selection.json"), "test_used_in_this_search_selection": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--audit-fb-inputs", action="store_true")
    parser.add_argument("--worker-fb-search", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.audit_fb_inputs:
        print(prepare_fb_inputs(root))
        return
    if args.worker_fb_search:
        fb_search(root)
        return
    if args.plan_only:
        print(json.dumps(plan(root), ensure_ascii=False, indent=2))
        return
    supervisor = Supervisor(root)
    if args.snapshot_only:
        supervisor.snapshot()
        return
    hpo.write_json(root / "plan.json", plan(root))
    (root / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    supervisor.event("supervisor_start", pid=os.getpid(), python=sys.executable)
    try:
        import torch
        import optuna
        hpo.write_json(root / "environment.json", {"time": now(), "python": sys.executable,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda, "optuna": optuna.__version__,
            "gpu": probe(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
            "repo_commit": probe(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "baseline_commit": probe(["git", "-C", str(BASELINE), "rev-parse", "HEAD"])})
        if not torch.cuda.is_available():
            raise RuntimeError("Required guardenv cannot see CUDA")
        prepare_fb_inputs(root)
        supervisor.wait_for_nell()
        supervisor.fb_pipeline()
    except BaseException as error:
        supervisor.action = "failed_needs_inspection_resume_with_launcher"
        supervisor.event("failure", error=str(error))
        supervisor.snapshot(error=str(error))
        raise
    supervisor.action = "bounded_campaign_finished"
    supervisor.event("campaign_finished")
    supervisor.snapshot()


if __name__ == "__main__":
    main()
