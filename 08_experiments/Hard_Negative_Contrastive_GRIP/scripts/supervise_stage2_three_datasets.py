#!/usr/bin/env python3
"""Single-seed, Stage-2-only SC-CL search for NELL23K, FB and CoDEx-M.

Existing Stage-1 adapters are pinned. The already running FB prerequisite job
can be adopted without interrupting its GPU process. New baseline launches
explicitly skip Stage 1. Search budgets advance in round-robin batches, with
one validation-selected final test per dataset after the declared search.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

import supervise_nell_fb_hpo as legacy

hpo = legacy.hpo
HNG, REPO, BASELINE, PYTHON = legacy.HNG, legacy.REPO, legacy.BASELINE, legacy.PYTHON
ROOT = HNG / "results/hpo/stage2_three_dataset_20261030"
PHYSICAL = Path("/home/kieran/grip-cu-runs/stage2_three_dataset_20261030")
LEGACY_ROOT = legacy.DEFAULT_ROOT
DATASETS = ("nell23k", "codexm", "fb15k237_2")
PAPER = {"nell23k": 0.8774, "fb15k237_2": 0.8875, "codexm": 0.9442}


def registry(root):
    records = {}
    nell = Path("/mnt/e/GRIP-CU/artifacts/stage2_memory_retention/nell_r1")
    for dataset in DATASETS:
        data = REPO / "04_benchmark/data" / dataset
        work = root / dataset
        if dataset == "nell23k":
            s1 = nell / "stage1_20261018T154406Z/adapter"
            b1 = nell / "control_20261018_C1/adapter"
            baseline_root = None
            epochs = 10
        elif dataset == "codexm":
            baseline_root = BASELINE / "outputs/paper_runs/codexm_taskfile_3090"
            s1, b1 = baseline_root / "stage1_adapter", baseline_root / "stage2_adapter"
            epochs = 5
        else:
            _, s1, b1, _ = legacy.fb_paths(LEGACY_ROOT)
            baseline_root = s1.parent
            epochs = 5
        task = data / "tasks" / f"grip_{dataset}_tasks.json"
        if dataset == "fb15k237_2":
            task = LEGACY_ROOT / "fb15k237_2/assets/grip_fb15k237_2_tasks_train_pair_repaired.json"
        record = {
            "dataset": dataset, "input": str(task), "valid": str(data / "processed/processed_val.json"),
            "test": str(data / "processed/processed_test.json"), "raw": str(data / "raw"),
            "s1_adapter": str(s1), "b1_adapter": str(b1), "epochs": epochs,
            "baseline_root": str(baseline_root) if baseline_root else None,
            "sampler": str(work / "assets/k9"), "search_dir": str(work / "search"),
            "relation_eligibility": str(work / "assets/relation_eligibility.json"),
            "paper_reference": PAPER[dataset], "seed": 2026, "search_stage": 2,
            "screen_size": 1000, "prune_after": 32, "total_trials": 12,
            "stage1_hyperparameters_searched": False,
        }
        if dataset == "codexm":
            record["entity_text_file"] = str(data / "raw/entity2text.txt")
        if dataset == "fb15k237_2":
            record["entity_text_file"] = str(LEGACY_ROOT / "fb15k237_2/assets/entity2text_train.txt")
        records[dataset] = record
    return records


def prepare_task_gate(record):
    """Keep every generation QA unchanged; gate InfoNCE on verified train pairs."""
    sys.path.insert(0, str(HNG / "src"))
    from hard_negative_grip.official_lists import load_train_relation_order
    from hard_negative_grip.offline_scoring import iter_matchable_relation_qa
    from hard_negative_grip.task_file import known_pair_relations, original_question_ids_from_payload, train_relation_alias_index

    task_path, raw = Path(record["input"]), Path(record["raw"])
    task = legacy.read(task_path)
    rel = raw / "rel2text.json" if record["dataset"] != "nell23k" else None
    entity = Path(record["entity_text_file"]) if record.get("entity_text_file") else None
    order = load_train_relation_order(raw, relation_text_file=rel)
    known = known_pair_relations(raw, splits=("train",), relation_text_file=rel, entity_text_file=entity)
    pairs = {(head.casefold(), tail.casefold()): (head, tail) for head, tail in known}
    qids = original_question_ids_from_payload(task, len(task["qa_samples"]))
    items = list(iter_matchable_relation_qa(task["qa_samples"], train_relation_alias_index(order), entity_pairs=pairs))
    eligible, excluded = [], []
    for item in items:
        index = int(item["qa_id"].split(":")[-1])
        qid = qids[index] if qids else item["qa_id"]
        if item["matched_train_relation"] in known.get(item["entity_pair"], set()):
            eligible.append(qid)
        else:
            excluded.append({"question_id": qid, "reason": "entity pair/gold not deterministically verified against train; generation objective retained"})
    if not eligible:
        raise ValueError(f"No verified training relation QA for {record['dataset']}")
    gate = {"train_only": True, "task_sha256": hpo.sha256_file(task_path),
            "train_sha256": hpo.sha256_file(raw / "train.txt"), "eligible_question_ids": sorted(eligible),
            "excluded_relation_qa": excluded, "generation_qa_count": len(task["qa_samples"]),
            "relation_qa_count": len(items), "qa_text_modified": False,
            "context_samples_modified": False, "test_used": False, "validation_used": False}
    destination = Path(record["relation_eligibility"])
    if destination.exists() and legacy.read(destination) != gate:
        raise ValueError(f"Immutable relation eligibility changed: {destination}")
    hpo.write_json(destination, gate)
    record["expected_listed"] = len(eligible)
    full_steps = math.ceil(len(task["qa_samples"]) / 512) * record["epochs"]
    record["rungs"] = sorted({step for step in (8, 16, 32, 64, 128, full_steps) if step <= full_steps})
    record["task_sha256"] = gate["task_sha256"]
    return record


def baseline_command(record):
    """This function cannot request Stage-1 training."""
    root = record.get("baseline_root")
    if not root or not legacy.adapter_ready(Path(record["s1_adapter"])):
        raise ValueError("An existing complete Stage-1 adapter is required")
    config = legacy.read(Path(record["s1_adapter"]) / "adapter_config.json")
    batch = 2 if record["dataset"] == "codexm" else 1
    return [str(PYTHON), str(BASELINE / "scripts/train_task_file_adapter.py"),
        "--task_file", record["input"], "--output_dir", root,
        "--model_name", "qwen-7b", "--model_source", "local",
        "--model_cache_dir", str(hpo.MODEL_CACHE_DEFAULT), "--local_files_only",
        "--dtype", "bfloat16", "--tokenize_max_length", "4096",
        "--stage1_epochs", "0", "--stage2_epochs", str(record["epochs"]),
        "--per_device_train_batch_size", str(batch), "--gradient_accumulation_steps", str(512 // batch),
        "--learning_rate", "1e-3", "--weight_decay", "1e-4", "--adam_epsilon", "1e-8",
        "--lora_r", str(config["r"]), "--lora_alpha", str(config["lora_alpha"]),
        "--target_modules", "down_proj", "up_proj", "gate_proj", "--gradient_checkpointing",
        "--s1_stop_loss_threshold", "0.15", "--s2_stop_loss_threshold", "0.15",
        "--s1_min_epoch", "1", "--s2_min_epoch", "1",
        "--stage1_checkpoint_interval", "0", "--stage2_checkpoint_interval", "10", "--seed", "2026"]


class Controller(legacy.Supervisor):
    def __init__(self, root):
        super().__init__(root)
        self.dataset = None
        self.adopted_pid = None
        self.legacy_lock = None

    def run(self, name, command, command_env=None):
        if any("train_task_file_adapter.py" in part for part in command):
            if "--stage1_epochs" not in command or float(command[command.index("--stage1_epochs") + 1]) != 0:
                raise ValueError("New Stage-1 training is disabled in this campaign")
        return super().run(name, command, command_env)

    def snapshot(self, **values):
        state = super().snapshot(**values)
        state["dataset"] = self.dataset
        state["search_stage"] = 2
        state["training_seed"] = 2026
        state["adopted_pid"] = self.adopted_pid
        results = {}
        for dataset in DATASETS:
            folder = self.root / dataset / "search"
            result = folder / "final_result.json"
            no_go = folder / "no_go.json"
            progress = folder / "search_progress.json"
            failure = self.root / dataset / "failure.json"
            failed = legacy.read(failure) if failure.exists() else {}
            results[dataset] = {"status": "failed", **failed} if failed and not failed.get("resolved") else (legacy.read(result) if result.exists() else (
                {"status": "no_validation_promotion"} if no_go.exists() else (
                    {"status": "searching", **legacy.read(progress)} if progress.exists() else {"status": "queued"})))
        state["dataset_results"] = results
        if self.adopted_pid and Path(f"/proc/{self.adopted_pid}").exists():
            log = LEGACY_ROOT / "logs/fb_matched_b1.log"
            state["latest_log"] = str(log)
            state["child_pid"] = self.adopted_pid
        elif values.get("log"):
            state["latest_log"] = values["log"]
        if self.dataset and "search" in self.action:
            live = self.root / self.dataset / "search/live_progress.json"
            if live.exists():
                state["stage2_trial"] = legacy.read(live)
                state["latest_log"] = state["stage2_trial"].get("log", state.get("latest_log"))
        log = Path(state["latest_log"]) if state.get("latest_log") else None
        state["latest_training_progress"] = None
        if log and log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 12000))
                matches = re.findall(r"\|\s*(\d+)/(\d+)", stream.read().decode(errors="replace"))
            if matches:
                state["latest_training_progress"] = dict(zip(["optimizer_step", "schedule_horizon_steps"], map(int, matches[-1])))
            state["latest_log_age_seconds"] = time.time() - log.stat().st_mtime
        state["all_three_targets_achieved"] = all(results[dataset].get("target_achieved") is True for dataset in DATASETS)
        hpo.write_json(self.root / "status.json", state)
        body = f"# Three-dataset Stage-2 search\n\nUpdated: {state['updated_at']}\n\nAction: `{self.action}`; dataset: `{self.dataset}`\n\n"
        body += "Seed 2026. Stage 1 is fixed. Promotion uses validation free-generation EM; success requires full-test EM above the matched B1 and paper reference.\n\n"
        body += "| Dataset | Paper reference | State / result |\n|---|---:|---|\n"
        for dataset, result in results.items():
            text = (f"test EM {100*result['selected_test_em']:.3f}%; target achieved={result.get('target_achieved')}"
                    if "selected_test_em" in result else result.get("status", "queued"))
            body += f"| {dataset} | {100*PAPER[dataset]:.2f}% | {text} |\n"
        body += f"\nCurrent progress: `{state.get('latest_training_progress')}`\n\nLog: `{state.get('latest_log')}`\n"
        (self.root / "STATUS.md").write_text(body, encoding="utf-8")
        return state

    def adopt(self, pid):
        self.adopted_pid = pid
        self.dataset = "fb15k237_2"
        self.action = "waiting_for_existing_fb_prerequisite_job"
        command_file = Path(f"/proc/{pid}/cmdline") if pid else None
        if command_file and command_file.exists():
            command = command_file.read_bytes().decode(errors="replace")
            if "train_task_file_adapter.py" not in command or "nell_fb_guard_20261029" not in command:
                raise ValueError("Refusing to adopt an unrelated process")
            start_ticks = Path(f"/proc/{pid}/stat").read_text().split()[21]
            hpo.write_json(self.root / "adopted_job.json", {"pid": pid, "start_ticks": start_ticks,
                "role": "existing one-time FB prerequisite restoration; not an HPO trial", "interrupted": False})
            while command_file.exists():
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text().split()
                except FileNotFoundError:
                    break
                if stat[21] != start_ticks or stat[2] == "Z":
                    break
                self.snapshot()
                time.sleep(30)
        self.adopted_pid = None
        # The adopted child inherited this lock from the old controller. Holding
        # it afterwards also prevents the old launcher from duplicating jobs.
        self.legacy_lock = (LEGACY_ROOT / "supervisor.lock").open("a+")
        while True:
            try:
                fcntl.flock(self.legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.snapshot()
                time.sleep(30)
        self.event("legacy_job_finished")
        self.wait_gpu()

    def wait_gpu(self):
        while True:
            state = legacy.probe(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            if state.get("returncode") != 0:
                raise RuntimeError(f"Cannot inspect GPU: {state}")
            if not state.get("stdout", "").strip():
                return
            self.snapshot(waiting_for_gpu=state["stdout"])
            time.sleep(30)

    def prepare(self, record):
        self.dataset = record["dataset"]
        self.action = f"{self.dataset}_fixed_stage1_preflight"
        s1, b1 = Path(record["s1_adapter"]), Path(record["b1_adapter"])
        if not legacy.adapter_ready(s1):
            raise RuntimeError(f"Missing fixed Stage-1 asset; automatic Stage-1 training is disabled: {s1}")
        identity = legacy.fingerprint(s1)
        identity_file = self.root / self.dataset / "fixed_stage1.json"
        if identity_file.exists() and legacy.read(identity_file) != identity:
            raise ValueError("Pinned Stage-1 adapter changed")
        hpo.write_json(identity_file, identity)
        if not legacy.adapter_ready(b1):
            self.run(f"{self.dataset}_resume_baseline_stage2", baseline_command(record))
        if legacy.fingerprint(s1) != identity:
            raise RuntimeError("Stage-1 adapter changed during baseline Stage 2")
        if not legacy.adapter_ready(b1):
            raise RuntimeError("Stage-2 baseline did not finish")
        raw = Path(record["raw"])
        work = self.root / self.dataset
        score = work / "assets/b1_train_scores"
        bank = Path(record["sampler"])
        e = legacy.env()
        e.update(DATASET=self.dataset, STEP="score", NEGATIVE_SOURCE="structure_counterfactual", SKIP_TMUX="1",
            INPUT_FILE=record["input"], EVAL_FILE=record["valid"], RAW_DIR=str(raw), S1_ADAPTER=str(s1), B1_ADAPTER=str(b1),
            MODEL_CACHE_DIR=str(hpo.MODEL_CACHE_DEFAULT), ARTIFACT_ROOT=str(work / "assets"), SCORES_DIR=str(score),
            SAMPLER_DIR=str(bank), MANIFEST=str(bank / "counterfactual_bank.jsonl"),
            COUNTERFACTUAL_METADATA=str(bank / "metadata.json"), EXPECTED_LISTED=str(record["expected_listed"]),
            RELATION_ELIGIBILITY=record["relation_eligibility"], LISTED_NEGATIVE_K="9", CANDIDATE_BATCH_SIZE="4",
            TORCH_DTYPE="bfloat16", SEED="2026")
        if self.dataset != "nell23k": e["RELATION_TEXT_FILE"] = str(raw / "rel2text.json")
        if record.get("entity_text_file"): e["ENTITY_TEXT_FILE"] = record["entity_text_file"]
        for stage, artifact in [("score", score / "run_stats.json"), ("freeze", bank / "metadata.json"), ("wire", bank / "wiring.json")]:
            if artifact.exists(): continue
            e.update(STEP=stage, RUN_ID=f"stage2_only_{self.dataset}_{stage}", RUN_DIR=str(work / stage))
            self.run(f"{self.dataset}_{stage}", ["bash", str(hpo.RUNNER)], e.copy())


def create_plan(root):
    records = registry(root)
    for dataset, record in records.items():
        work = root / dataset
        work.mkdir(parents=True, exist_ok=True)
        search = work / "search"
        target = PHYSICAL / dataset / "search"
        target.mkdir(parents=True, exist_ok=True)
        if not search.exists() and not search.is_symlink(): search.symlink_to(target, target_is_directory=True)
        prepare_task_gate(record)
        hpo.write_json(work / "dataset_contract.json", record)
    result = {"created_at": legacy.now(), "seed": 2026, "search_stage": 2,
        "stage1_search_enabled": False, "new_stage1_training_enabled": False,
        "datasets": records, "order": list(DATASETS), "trials_per_dataset": 12, "trials_per_turn": 2,
        "scheduling": "Finish adopted FB prerequisite job; then round-robin two additional Stage-2 trials per dataset. Freeze full-validation selection after all declared trials, then one selected test.",
        "success": "all three datasets: full-test free-generation EM exceeds matched B1 and paper reference",
        "scope": "single seed; fixed local protocols; FB retains the previously documented 464-QA repair; no further QA changes. Unverified relation pairs keep generation loss and receive no candidate loss.",
        "nell_validation_scope": "Existing fixed NELL task generation used validation few-shot references; validation is developmental. Prior NELL tests have been inspected; this campaign is adaptive exploration.",
        "failure_policy": "Record failed datasets and continue other ready datasets; preserve checkpoints for manual resume."}
    hpo.write_json(root / "campaign_contract.json", result)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--adopt-pid", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        legacy.fb_search(args.root, legacy.read(args.worker_spec))
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = create_plan(root)
    if args.plan_only:
        print(json.dumps({ds: {k:r[k] for k in ["s1_adapter", "b1_adapter", "expected_listed", "rungs", "search_stage"]} for ds,r in records.items()}, indent=2))
        return
    controller = Controller(root)
    (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    controller.event("three_dataset_controller_started", pid=os.getpid(), seed=2026)
    try:
        controller.adopt(args.adopt_pid)
        ready = {}
        for budget in range(2, 13, 2):
            for dataset in DATASETS:
                controller.dataset = dataset
                if dataset not in ready:
                    try:
                        controller.prepare(records[dataset])
                    except Exception as error:
                        controller.event("dataset_prerequisite_failed", dataset=dataset, error=str(error))
                        hpo.write_json(root / dataset / "failure.json", {"error": str(error), "phase": "prerequisite"})
                        ready[dataset] = False
                        continue
                    ready[dataset] = True
                    failure = root / dataset / "failure.json"
                    if failure.exists():
                        hpo.write_json(failure, {**legacy.read(failure), "resolved": True, "resolved_at": legacy.now()})
                if not ready[dataset]: continue
                resource_cap = min(records[dataset]["rungs"][-1], 32 * 2 ** (budget // 2 - 1))
                spec = {**records[dataset], "trial_budget": budget, "search_only": True, "resource_cap": resource_cap}
                path = root / dataset / "active_worker_spec.json"
                hpo.write_json(path, spec)
                try:
                    controller.run(f"{dataset}_search_to_{budget}", [str(PYTHON), "-u", str(Path(__file__).resolve()),
                        "--root", str(root), "--worker-spec", str(path)])
                except Exception as error:
                    controller.event("dataset_search_failed", dataset=dataset, error=str(error))
                    hpo.write_json(root / dataset / "failure.json", {"error": str(error), "phase": "search"})
                    ready[dataset] = False
        for dataset in DATASETS:
            if not ready.get(dataset): continue
            controller.dataset = dataset
            path = root / dataset / "active_worker_spec.json"
            hpo.write_json(path, {**records[dataset], "trial_budget": 12, "search_only": False})
            try:
                controller.run(f"{dataset}_search_final_selection", [str(PYTHON), "-u", str(Path(__file__).resolve()),
                    "--root", str(root), "--worker-spec", str(path)])
            except Exception as error:
                controller.event("dataset_finalization_failed", dataset=dataset, error=str(error))
                hpo.write_json(root / dataset / "failure.json", {"error": str(error), "phase": "finalization"})
    except BaseException as error:
        controller.action = "controller_failed"
        controller.event("failure", error=str(error))
        controller.snapshot(error=str(error))
        raise
    state = controller.snapshot()
    controller.action = "all_three_single_seed_targets_achieved" if state["all_three_targets_achieved"] else "search_budget_finished_targets_not_all_achieved"
    controller.snapshot()


if __name__ == "__main__":
    main()
