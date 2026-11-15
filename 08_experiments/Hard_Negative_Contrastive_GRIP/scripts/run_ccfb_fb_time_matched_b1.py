#!/usr/bin/env python3
"""Run the FB15k-237 generation-only B1 under the seed2026 Soft-Mix S2 time budget."""
import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch
from transformers import TrainerCallback

from analyze_ccfb_seed2026 import analyze
from run_ccfb_top9 import CODE, FB, HNG, RUNS, argument, hpo, sprint

ROOT = RUNS / "ccfb_fb_time_matched_b1_seed2026"
HERE = Path(__file__).resolve()
GIB = 1024**3
QA_COUNT = 19845
QUERY_COUNT = 10000
ANCHOR_STEPS = (117, 234)
SAVE_MARGIN_SECONDS = 12
CHECKPOINT_FILES = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth")


def verify_hashes(files):
    for name, expected in files.items():
        if hpo.sha256(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def receipt(name):
    result = hpo.read_json(FB / "completed" / f"{name}.json")
    verify_hashes(result["outputs"])
    return result["command"]


def prepare():
    original = hpo.read_json(FB / "experiment_plan.json")
    selection = hpo.read_json(FB / "selection.json")
    if (hpo.read_json(FB / "run_status.json")["status"] != "COMPLETE"
            or selection["selected"] != "soft_mix_1" or selection["test_seen"]):
        raise ValueError("The FB Soft-Mix reference must be the validation-selected completed run.")
    baseline, soft_mix, test = (receipt(name) for name in ("b1", "soft_mix_1", "test_soft_mix_1"))
    baseline_test = receipt("test_b1_seed2026")
    receipt("s1")
    receipt("tasks")
    if soft_mix != original["training_commands"]["soft_mix_1"]:
        raise ValueError("The Soft-Mix training command differs from the archived plan.")
    for flag in ("--input_file", "--s1_adapter", "--eval_file", "--seed", "--model_name", "--lora_r",
                 "--lora_alpha", "--qa_loss_mask", "--s2_optimizer", "--learning_rate",
                 "--per_device_train_batch_size", "--gradient_accumulation_steps", "--model_cache_dir"):
        if argument(baseline, flag) != argument(soft_mix, flag):
            raise ValueError(f"Original B1 and Soft-Mix differ at {flag}")
    if (argument(baseline, "--seed"), argument(baseline, "--stage"), argument(baseline, "--lambda_candidate"),
            argument(baseline, "--eval_splits"), argument(test, "--decode"), argument(test, "--splits")) != (
            "2026", "b1", "0", "validation", "generate", "test"):
        raise ValueError("Unexpected historical training or decoding settings.")
    evaluation = Path(argument(test, "--eval_file"))
    if str(evaluation) != argument(baseline, "--eval_file"):
        raise ValueError("Training and test evaluation files differ.")
    soft_adapter = FB / "soft_mix_1/listed/adapter"
    baseline_adapter = FB / "b1_seed2026/b1/adapter"
    if argument(test, "--adapter_dir") != str(soft_adapter):
        raise ValueError("The archived test command does not evaluate the selected Soft-Mix model.")
    meta_files = [adapter / "run_metadata.json" for adapter in (baseline_adapter, soft_adapter)]
    b1, soft = (hpo.read_json(path) for path in meta_files)
    for meta in (b1, soft):
        if (meta["qa_samples"], meta["trainer_global_step"], meta["trainer_epoch"], meta["resumed_from"]) != (
                QA_COUNT, 117, 3.0, None):
            raise ValueError("Expected an uninterrupted original seed2026 three-epoch run.")
    if b1["candidate_forwards"] or not soft["candidate_forwards"]:
        raise ValueError("The original controls differ from generation-only B1 and Soft-Mix.")
    budget = soft["seconds"]
    if not all(math.isfinite(v) and v > 0 for v in (budget, b1["seconds"])):
        raise ValueError("Original training time measurements are invalid.")
    epochs = math.ceil(3 * budget / b1["seconds"])
    s1 = Path(argument(baseline, "--s1_adapter"))
    raw = Path(argument(baseline, "--raw_dir"))
    sprint.verify_data({
        "dataset": "fb15k237", "data": str(evaluation.parent), "raw": str(raw), "evaluation": str(evaluation),
        "spec": {"validation_count": QUERY_COUNT, "test_count": QUERY_COUNT}, "initial": True,
    })
    training_code = [Path(baseline[1]), Path(test[1]),
                     HNG / "src/hard_negative_grip/listed_training.py",
                     HNG / "src/hard_negative_grip/decode_io.py",
                     HNG / "src/hard_negative_grip/task_file.py"]
    verify_hashes({str(path): original["sources"][str(path)] for path in training_code})
    train = sprint.option(sprint.option(baseline, "--output_dir", ROOT), "--involve_qa_epochs", epochs)
    soft_test = Path(argument(test, "--output_dir"))
    original_test = Path(argument(baseline_test, "--output_dir"))
    report_files = [p / name for p in (soft_test, original_test)
                    for name in ("summary.json", "predictions_correct.jsonl")]
    inputs = [HERE, *training_code, *meta_files, *report_files, evaluation,
              Path(argument(baseline, "--input_file")), raw / "train.txt",
              Path(argument(baseline, "--relation_text_file")),
              *(s1 / name for name in ("adapter_model.safetensors", "adapter_config.json", "s1_optimizer.pt"))]
    plan = {
        "dataset": "fb15k237", "seed": 2026, "s2_budget_seconds": budget,
        "original_b1_seconds": b1["seconds"], "scheduler_epochs": epochs,
        "expected_steps_per_epoch": b1["trainer_global_step"] // 3,
        "anchor_steps": list(ANCHOR_STEPS), "save_margin_seconds": SAVE_MARGIN_SECONDS,
        "save_interval_steps": int(argument(train, "--save_steps")),
        "selection_rule": "Maximum validation official_em among steps 117, 234 and the latest checkpoint saved within budget. Ties use the earlier step.",
        "timing_scope": "Active S2 Trainer loop including checkpoint writes. Model loading, validation and test are outside this budget. Resume carries forward durably recorded active time.",
        "reference_b1_train": baseline, "reference_soft_mix_train": soft_mix,
        "train": train, "eval_template": test,
        "inputs": {str(path): hpo.sha256(path) for path in dict.fromkeys(inputs)},
    }
    hpo.freeze_text(ROOT / "plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(f"Soft-Mix S2 {budget:.1f}s; three-epoch B1 {b1['seconds']:.1f}s; "
          f"linear scheduler horizon {epochs} epochs", flush=True)


def complete_checkpoint(path):
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in CHECKPOINT_FILES)


def keep_anchor(path, output_dir):
    keep = output_dir / "selection_checkpoints" / path.name
    keep.parent.mkdir(parents=True, exist_ok=True)
    if keep.exists():
        if not complete_checkpoint(keep) or not (keep / CHECKPOINT_FILES[0]).samefile(path / CHECKPOINT_FILES[0]):
            raise ValueError(f"Existing selection checkpoint differs: {keep}")
    else:
        with tempfile.TemporaryDirectory(dir=keep.parent, prefix=".checkpoint-") as tmp:
            copied = Path(tmp) / path.name
            shutil.copytree(path, copied, copy_function=os.link)
            copied.rename(keep)
    return keep


class Stage2Clock(TrainerCallback):
    """Track active training across restarts and retain predeclared checkpoints."""

    def __init__(self, output_dir, budget, max_steps):
        self.output_dir = output_dir
        self.path = output_dir / "timing.json"
        self.budget = budget
        self.max_steps = max_steps
        self.records = hpo.read_json(self.path) if self.path.is_file() else {
            "active_seconds": 0.0, "latest_step": 0, "latest_step_seconds": 0.0, "saved": {},
        }
        self.prior_seconds = self.records["active_seconds"]
        self.start = None
        self.durations = deque(maxlen=8)
        if self.records["latest_step_seconds"] > 0:
            self.durations.append(self.records["latest_step_seconds"])

    def checkpoints(self):
        return {int(step): record for step, record in self.records["saved"].items()
                if record["seconds"] <= self.budget and complete_checkpoint(Path(record["path"]))}

    def record(self, now):
        if self.start is not None:
            self.records["active_seconds"] = self.prior_seconds + now - self.start
        hpo.write_json(self.path, self.records)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps != self.max_steps:
            raise ValueError(f"Expected {self.max_steps} scheduler steps, got {state.max_steps}")
        self.start = self.last_step = time.monotonic()
        print(f"[S2 budget] {self.budget:.1f}s total, {self.prior_seconds:.1f}s used; "
              f"starting after step {state.global_step}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        now = time.monotonic()
        elapsed = self.prior_seconds + now - self.start
        duration = now - self.last_step
        self.durations.append(duration)
        self.last_step = now
        self.records["latest_step"] = state.global_step
        self.records["latest_step_seconds"] = duration
        remaining = self.budget - elapsed
        at_horizon = state.global_step == self.max_steps
        stop = at_horizon or remaining <= max(self.durations) + SAVE_MARGIN_SECONDS
        if stop:
            control.should_training_stop = True
            self.records["stop_requested"] = "scheduler_horizon" if at_horizon else "s2_time_budget"
        control.should_save = remaining >= 0 and (
            control.should_save or stop or state.global_step in ANCHOR_STEPS
        )
        self.record(now)
        if stop:
            print(f"[S2 budget] stop at step {state.global_step}, {elapsed:.1f}s used before checkpoint save",
                  flush=True)

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not complete_checkpoint(path):
            raise RuntimeError(f"Incomplete checkpoint: {path}")
        if state.global_step in ANCHOR_STEPS:
            path = keep_anchor(path, self.output_dir)
        now = time.monotonic()
        elapsed = self.prior_seconds + now - self.start
        self.records["saved"][str(state.global_step)] = {
            "path": str(path), "epoch": state.epoch, "seconds": elapsed,
        }
        self.record(now)

    def on_train_end(self, args, state, control, **kwargs):
        self.records["stop_reason"] = self.records["stop_requested"]
        self.record(time.monotonic())
        self.start = None


def fit(plan, args):
    import train_listed_contrastive as training

    clock = Stage2Clock(ROOT, plan["s2_budget_seconds"],
                        plan["scheduler_epochs"] * plan["expected_steps_per_epoch"])
    available = clock.checkpoints()
    if "stop_reason" not in clock.records:
        enough_time = clock.budget - clock.prior_seconds > max(clock.durations, default=0) + SAVE_MARGIN_SECONDS
        if not enough_time:
            clock.records["stop_reason"] = "remaining_time_after_interruption"
            clock.record(time.monotonic())
        else:
            training.set_random_seed(args.seed)
            _, _, texts, metas, _ = training.resolve_training_assets(args)
            if len(texts) != QA_COUNT:
                raise ValueError(f"Expected {QA_COUNT} QA, found {len(texts)}")
            adapter = Path(args.s1_adapter)
            training.write_run_config(ROOT, args)
            print(f"[b1] {len(texts)} QA, seed={args.seed}, answer-only generation loss, no InfoNCE",
                  flush=True)
            model, tokenizer = training.load_adapter(args, adapter, trainable=True)
            dataset = training.ListedQADataset(texts, metas, tokenizer, answer_only=args.qa_loss_mask == "answer_only")
            trainer = training.ListedContrastiveTrainer(
                model=model,
                args=training.training_arguments(ROOT / "trainer_b1", args, args.involve_qa_epochs, len(dataset)),
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=training.ListedDataCollator(tokenizer=tokenizer),
                lambda_candidate=0.0,
                temperature=args.temperature,
                memory_size=args.memory_size,
                inherit_optimizer_state=training.load_s1_optimizer_state(args, adapter),
                callbacks=[clock],
            )
            resume = available[max(available)]["path"] if available else None
            print(f"[b1] resume from {resume}" if resume else "[b1] start from frozen S1", flush=True)
            try:
                trainer.train(resume_from_checkpoint=resume)
            except BaseException:
                clock.record(time.monotonic())
                raise
            if trainer.candidate_forwards:
                raise RuntimeError("B1 unexpectedly executed contrastive forwards.")

    available = clock.checkpoints()
    if not available:
        raise RuntimeError("Training produced no completed checkpoint inside the Soft-Mix time budget.")
    candidates = sorted({int(step) for step in plan["anchor_steps"] if step in available} | {max(available)})
    rows = {str(step): {**available[step], "adapter_sha256": hpo.sha256(
        Path(available[step]["path"]) / "adapter_model.safetensors")} for step in candidates}
    hpo.freeze_text(ROOT / "train_complete.json", json.dumps({
        "stop_reason": clock.records["stop_reason"], "active_seconds": clock.records["active_seconds"],
        "last_training_step": clock.records["latest_step"], "candidates": rows,
    }, indent=2) + "\n")


def set_status(stage, **fields):
    hpo.write_json(ROOT / "run_status.json", {"status": "RUNNING", "stage": stage, **fields})
    print(f"[fb-time-matched-b1] {stage} {fields}", flush=True)


def launch(command, log_file, lock):
    with log_file.open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                       check=True, pass_fds=(lock.fileno(),))


def evaluate(plan, checkpoint, split, destination, lock):
    command = list(plan["eval_template"])
    for flag, value in (("--adapter_dir", checkpoint), ("--splits", split), ("--output_dir", destination)):
        command = sprint.option(command, flag, value)
    entry = destination / "summary.json"
    launch(command, ROOT / f"{destination.name}.log", lock)
    summary = hpo.read_json(entry)
    score = summary[split]
    if score["count"] != QUERY_COUNT or (split == "validation" and "test" in summary):
        raise ValueError(f"Incomplete or incorrect {split} split: {entry}")
    return score["official_em"]


def execute(plan, lock):
    trained = ROOT / "train_complete.json"
    if not trained.is_file():
        if shutil.disk_usage(ROOT).free < 25 * GIB:
            raise RuntimeError("Training requires 25 GiB for rolling and retained checkpoints plus reserve.")
        set_status("train")
        launch([plan["train"][0], str(HERE), "--fit", *plan["train"][2:]], ROOT / "train.log", lock)

    training = hpo.read_json(trained)
    candidates = training["candidates"]
    for value in candidates.values():
        verify_hashes({str(Path(value["path"]) / "adapter_model.safetensors"): value["adapter_sha256"]})
    selection_path = ROOT / "selection.json"
    if not selection_path.is_file():
        scores = {}
        for step, candidate in candidates.items():
            set_status("validation", checkpoint_step=int(step))
            scores[step] = evaluate(plan, candidate["path"], "validation",
                                    ROOT / "validation" / f"step-{step}", lock)
        best = max(scores, key=lambda step: (scores[step], -int(step)))
        hpo.freeze_text(selection_path, json.dumps({
            "metric": "validation.official_em", "scores": scores, "selected_step": int(best),
            "selected": candidates[best], "test_seen": False,
        }, indent=2) + "\n")
    selection = hpo.read_json(selection_path)
    chosen = selection["selected"]
    if chosen != candidates[str(selection["selected_step"])]:
        raise ValueError("Selected model differs from the recorded training candidate.")

    test_dir = ROOT / "test"
    set_status("test", checkpoint_step=selection["selected_step"])
    result = evaluate(plan, chosen["path"], "test", test_dir, lock)
    soft_test = Path(argument(plan["eval_template"], "--output_dir"))
    reference = hpo.read_json(soft_test / "summary.json")["test"]["official_em"]
    comparison = analyze(
        "fb15k237", FB, Path(argument(plan["eval_template"], "--eval_file")),
        soft_test, test_dir, QUERY_COUNT, (reference, result),
        (Path(argument(plan["eval_template"], "--adapter_dir")), Path(chosen["path"])),
        methods=("soft_mix_1", "time_matched_b1"),
    )
    comparison["s2_training"]["time_matched_b1"] = {
        "batch_active_seconds": training["active_seconds"], "budget_seconds": plan["s2_budget_seconds"],
        "selected_checkpoint_seconds": chosen["seconds"], "selected_step": selection["selected_step"],
        "selected_epoch": chosen["epoch"], "last_training_step": training["last_training_step"],
    }
    comparison["validation_em"] = {
        "soft_mix_1": hpo.read_json(FB / "selection.json")["validation"]["soft_mix_1"],
        "time_matched_b1": selection["scores"][str(selection["selected_step"])],
    }
    comparison["three_epoch_b1_test_em"] = hpo.read_json(
        FB / "test/b1_seed2026/summary.json")["test"]["official_em"]
    hpo.freeze_text(ROOT / "comparison.json", json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    hpo.write_json(ROOT / "run_status.json", {
        "status": "COMPLETE", "selected_step": selection["selected_step"], "test_official_em": result,
    })


def run(plan):
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run the control inside the dedicated tmux session.")
    status = ROOT / "run_status.json"
    with (RUNS / "ccfb_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sprint.check_idle_gpu()
        verify_hashes(plan["inputs"])
        if status.is_file() and hpo.read_json(status)["status"] == "COMPLETE":
            return
        try:
            execute(plan, lock)
        except (Exception, KeyboardInterrupt) as exc:
            current = hpo.read_json(status) if status.is_file() else {}
            hpo.write_json(status, {**current, "status": "INTERRUPTED", "error": str(exc)})
            raise


def main():
    if sys.argv[1:2] == ["--fit"]:
        import train_listed_contrastive as training

        plan = hpo.read_json(ROOT / "plan.json")
        if not os.environ.get("TMUX") or sys.argv[2:] != plan["train"][2:]:
            raise ValueError("The fit worker requires the frozen tmux command.")
        del sys.argv[1]
        fit(plan, training.parse_args())
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Verify reference artifacts and freeze the run on CPU.")
    if parser.parse_args().prepare:
        prepare()
    else:
        run(hpo.read_json(ROOT / "plan.json"))


if __name__ == "__main__":
    main()
