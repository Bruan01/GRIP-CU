#!/usr/bin/env python3
"""Serial post-WN queue for Llama 3.1 on the four CCF B datasets."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import analyze_ccfb_seed2026 as paired
import ccfb_llama_runtime as llama
import run_ccfb_sprint as sprint
import run_fb_three_configs as fb
from hard_negative_grip.decode_io import load_jsonl
from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
from hard_negative_grip.task_file import load_graph_record

HNG, CODE, hpo = fb.HNG, fb.CODE, fb.hpo
RUNS = HNG / "results/runs"
QUEUE = RUNS / "ccfb_llama3_1_seed2026"
WN = RUNS / "ccfb_wn_independent_s1_seed2027_s2seed2026"
CACHE = CODE / "model_cache"
MODEL = CACHE / llama.MODEL_FOLDER
RUNTIME = HNG / "scripts/ccfb_llama_runtime.py"
ORDER = ("fb15k237", "wn18rr", "nell23k", "codexm")
EXPECTED = {
    "fb15k237": (19845, 3021, 0), "wn18rr": (19782, 2697, 79),
    "nell23k": (19808, 3030, 0), "codexm": (19798, 2758, 202),
}
PAPER = {"fb15k237": 0.7779, "wn18rr": 0.6326, "nell23k": 0.7740, "codexm": 0.9158}
GIB = 1024**3
RESERVE = 3 * GIB
ARMS = {"b1": "b1", "soft_mix_1": "listed"}
TASKS = ("qwen_analysis", "llama_tokenizer_check", "wn_independent_s1", "llama_weights", "llama_gpu_check", *ORDER)
STATUS = {name: {"status": "QUEUED"} for name in TASKS}


def write_status(task, state, **fields):
    current = {"status": state, **fields}
    if STATUS[task] != current:
        print(f"[queue] {task} {state}" + (f" {fields}" if fields else ""), flush=True)
    STATUS[task] = current
    hpo.write_json(QUEUE / "queue_status.json", {
        "tasks": STATUS, "order": list(TASKS), "model": llama.MODEL_ID,
        "model_revision": llama.REVISION, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })


def inputs(dataset):
    spec = hpo.read_json(sprint.MATRIX)["datasets"][dataset]
    primary = RUNS / spec["primary_root"]
    raw, data, evaluation, task, _s1 = sprint.paths(dataset, primary)
    return spec, primary, raw, data, evaluation, task


def qwen_analysis():
    archive = hpo.read_json(HNG / "results/archive/ccfb_seed2026_paired_errors.json")["datasets"]
    rows = {}
    for dataset in ORDER:
        spec, primary, raw, _, evaluation, _ = inputs(dataset)
        record = load_graph_record(evaluation)
        evidence = archive[dataset]
        left, right = [HNG / name for name in evidence["inputs"] if name.endswith(".jsonl")]
        for path in (left, right, evaluation):
            if hpo.sha256(path) != evidence["inputs"][str(path.relative_to(HNG))]:
                raise ValueError(f"Qwen analysis source changed: {path}")
        n = spec["test_count"]
        _, b1 = paired.read_predictions(left.parent, n)
        _, sm = paired.read_predictions(right.parent, n)
        relations = {}
        for q in record["recurrent_questions"]:
            qid = q["question_id"]
            if q["split"] != "test" or qid == evidence["excluded_question_id"]:
                continue
            a, b = b1[qid]["official_correct"], sm[qid]["official_correct"]
            rel = relations.setdefault(q["answer"], Counter())
            rel.update(n=1, b1_correct=int(a), soft_mix_correct=int(b),
                       gains=int(b and not a), regressions=int(a and not b))
        if sum(row["n"] for row in relations.values()) != evidence["n"]:
            raise ValueError(f"Relation counts differ for {dataset}")
        if sum(row["gains"] - row["regressions"] for row in relations.values()) != evidence["net"]:
            raise ValueError(f"Paired gains differ for {dataset}")
        stats = hpo.read_json(primary / "confusion/run_stats.json")
        rows[dataset] = {
            "by_gold_relation": relations, "paired_summary": evidence,
            "scoring": {key: stats.get(key) for key in
                        ("qa_count", "total_candidates", "elapsed_sec", "peak_gpu_gib")},
        }
    hpo.write_json(QUEUE / "qwen_relation_and_cost.json", {"seed": 2026, "datasets": rows})


def sequence_lengths(tokenizer, texts):
    lengths = []
    for start in range(0, len(texts), 512):
        encoded = tokenizer(texts[start:start + 512], truncation=False, return_attention_mask=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def cpu_preflight():
    llama.fetch_files(MODEL, llama.METADATA)
    from hard_negative_grip.task_file import assistant_gold, assistant_answer_prefix
    from hard_negative_grip.listed_training import answer_only_labels
    from hard_negative_grip import listed_training

    tokenizer = llama.get_tokenizer(MODEL)
    offset = listed_training.answer_start_char
    listed_training.answer_start_char = llama.assistant_offset
    report = {}
    for dataset in ORDER:
        spec, _, raw, data, evaluation, task = inputs(dataset)
        sprint.verify_data({"dataset": dataset, "data": str(data), "raw": str(raw),
                            "evaluation": str(evaluation), "spec": spec, "initial": True})
        payload = hpo.read_json(task)
        contexts = [llama.render_context(tokenizer, text) for text in payload["context_samples"]]
        qa = [llama.render_qa(tokenizer, text) for text in payload["qa_samples"]]
        if len(qa) != EXPECTED[dataset][0]:
            raise ValueError(f"Training QA count changed for {dataset}")
        if any(assistant_gold(old) != assistant_gold(new)
               for old, new in zip(payload["qa_samples"], qa, strict=True)):
            raise ValueError(f"Gold answers changed in the Llama template for {dataset}")
        c_lengths = sequence_lengths(tokenizer, contexts)
        q_lengths = sequence_lengths(tokenizer, qa)
        longest_context = max(range(len(c_lengths)), key=c_lengths.__getitem__)
        longest_qa = max(range(len(q_lengths)), key=q_lengths.__getitem__)
        if max(c_lengths[longest_context], q_lengths[longest_qa]) > llama.MAX_LENGTH:
            raise ValueError(f"{dataset}: a training sequence exceeds {llama.MAX_LENGTH} tokens")
        sample = qa[longest_qa]
        ids = tokenizer(sample)["input_ids"]
        labels = answer_only_labels(sample, ids, tokenizer)
        prefix = tokenizer(assistant_answer_prefix(sample))["input_ids"]
        if ids[0] != tokenizer.bos_token_id or ids.count(tokenizer.bos_token_id) != 1:
            raise ValueError("Expected a single Llama BOS")
        if ids[-1] != tokenizer.eos_token_id or prefix[0] != tokenizer.bos_token_id:
            raise ValueError("Llama generation and candidate tokenization differ in boundaries")
        answer = tokenizer.decode([token for token in labels if token != -100])
        if assistant_gold(sample) not in answer or "<answer>" not in answer:
            raise ValueError("Answer-only labels failed to retain the gold answer")
        report[dataset] = {
            "task_sha256": hpo.sha256(task), "eval_sha256": hpo.sha256(evaluation),
            "raw_train_sha256": hpo.sha256(raw / "train.txt"),
            "context_count": len(contexts), "qa_count": len(qa),
            "max_context_tokens": c_lengths[longest_context], "longest_context_index": longest_context,
            "max_qa_tokens": q_lengths[longest_qa], "truncated_count": 0,
        }
        print(f"[llama tokenizer] {dataset} {report[dataset]}", flush=True)
    listed_training.answer_start_char = offset
    hpo.write_json(QUEUE / "tokenizer_check.json", report)
    return report


def wait_for_wn():
    while True:
        path = WN / "run_status.json"
        if path.is_file() and hpo.read_json(path)["status"] == "COMPLETE":
            report = hpo.read_json(WN / "comparison.json")
            if report["n"] != 3134 or report["s1_seed"] != 2027:
                raise ValueError("WN independent pair is incomplete or from another seed")
            hpo.write_json(QUEUE / "wn_independent_s1_comparison.json", report)
            write_status("wn_independent_s1", "COMPLETE", delta_em=report["delta_em"])
            return
        live = subprocess.run(["tmux", "has-session", "-t", "=ccfb-wn-independent-s1-2027"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        reason = "WN training or evaluation is still active" if live else "WN has no completion receipt; waiting for its recovery"
        write_status("wn_independent_s1", "WAITING", reason=reason)
        time.sleep(60)


def disk_needed(dataset, root, *, include_probe=False):
    """Estimate from rank and real teacher-output sizes, with no extra embedding copy."""
    spec, primary, *_ = inputs(dataset)
    cfg = hpo.read_json(MODEL / "config.json")
    lora_params = 3 * (cfg["hidden_size"] + cfg["intermediate_size"]) * cfg["num_hidden_layers"] * spec["lora_r"]
    adapter = lora_params * 4 + 30 * 1024**2
    optimizer = lora_params * 8 + 16 * 1024**2
    scores = sum(p.stat().st_size for p in (primary / "confusion").glob("*.jsonl"))
    expected = 7 * adapter + 5 * optimizer + 2 * scores + 512 * 1024**2
    if include_probe:
        expected += 2 * adapter + optimizer + 64 * 1024**2
    already_allocated = 0 if not root.exists() else int(subprocess.check_output(
        ["du", "-s", "-B1", str(root)], text=True).split()[0])
    return max(0, expected - already_allocated) + RESERVE


def wait_for_disk(task, required):
    while True:
        needed = required()
        free = shutil.disk_usage(QUEUE).free
        if free >= needed:
            return
        write_status(task, "WAITING_DISK", free_gib=round(free / GIB, 2), needed_gib=round(needed / GIB, 2),
                     additional_gib=round((needed - free) / GIB, 2))
        time.sleep(60)


@contextmanager
def gpu(task):
    with (RUNS / "ccfb_gpu.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sprint.check_idle_gpu()
            except (BlockingIOError, RuntimeError):
                fcntl.flock(lock, fcntl.LOCK_UN)
                write_status(task, "WAITING_GPU")
                time.sleep(30)
            else:
                break
        yield lock


def worker(kind, cmd):
    return [cmd[0], str(RUNTIME), kind, *cmd[2:]]


def training(root, dataset, stage, output, manifest=None, *, task=None, evaluation=None):
    spec, _, raw, _, eval_path, task_path = inputs(dataset)
    cmd = fb.training(root, raw, evaluation or eval_path, stage, output, manifest)
    for flag, value in (
            ("--input_file", task or task_path), ("--model_name", llama.MODEL_SHORT), ("--model_cache_dir", CACHE),
            ("--lora_r", spec["lora_r"]), ("--lora_alpha", spec["lora_alpha"]), ("--save_total_limit", 1)):
        cmd = sprint.option(cmd, flag, value)
    if dataset == "nell23k":
        cmd = sprint.option(cmd, "--relation_text_file", remove=True)
        cmd.append("--official_drop_last_line")
    return worker("train", cmd)


def test_command(adapter, evaluation, output, split="test"):
    return worker("test", fb.command(HNG / "scripts/eval_listed_decode.py",
        "--adapter_dir", adapter, "--output_dir", output, "--eval_file", evaluation,
        "--splits", split, "--model_name", llama.MODEL_SHORT, "--model_cache_dir", CACHE,
        "--decode", "generate", "--gen_max_length", 1000, "--progress_every", 100))


def stage_run(root, name, cmd, outputs, lock):
    receipt = root / "completed" / f"{name}.json"
    if receipt.is_file():
        saved = hpo.read_json(receipt)
        if saved["command"] != cmd or saved["outputs"] != {str(p): hpo.sha256(p) for p in outputs}:
            raise ValueError(f"Completed output differs: {root.name}/{name}")
        return
    started = time.monotonic()
    print(f"[llama] {root.name} / {name}", flush=True)
    with (root / f"{name}.log").open("a") as log:
        subprocess.run(cmd, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                       pass_fds=(lock.fileno(),))
    hpo.write_json(receipt, {"command": cmd, "outputs": {str(p): hpo.sha256(p) for p in outputs},
                             "wall_seconds": time.monotonic() - started})


def gpu_probe(report):
    # The longest FB context and longest relation QA from the real inputs
    # exercise the full BF16 model and the contrastive backward path.
    _, primary, _, _, evaluation, task = inputs("fb15k237")
    root = QUEUE / "gpu_check"
    root.mkdir(exist_ok=True)
    payload = hpo.read_json(task)
    manifest = load_jsonl(primary / "soft_mix_1/negatives.jsonl")
    tokenizer = llama.get_tokenizer(MODEL)
    selected = max(manifest, key=lambda row: len(tokenizer(llama.render_qa(
        tokenizer, payload["qa_samples"][int(row["question_id"].split(":")[-1])]))["input_ids"]))
    qid = selected["question_id"]
    qa_index = int(qid.split(":")[-1])
    longest = report["fb15k237"]["longest_context_index"]
    probe_task = root / "tasks.json"
    hpo.write_json(probe_task, {
        "context_samples": [payload["context_samples"][longest]], "qa_samples": [payload["qa_samples"][qa_index]],
        "original_question_ids": [qid],
    })
    probe_manifest = root / "negatives.jsonl"
    hpo.freeze_text(probe_manifest, json.dumps(selected, ensure_ascii=False) + "\n")
    probe_eval = root / "validation.json"
    source = load_graph_record(evaluation)
    example = next(row for row in source["recurrent_questions"] if row["split"] == "validation")
    hpo.write_json(probe_eval, {
        "title": source["title"], "graph": source["graph"], "recurrent_questions": [example],
    })
    s1 = root / "s1_adapter"
    listed = root / "listed/adapter"
    with gpu("llama_gpu_check") as lock:
        write_status("llama_gpu_check", "RUNNING")
        for stage, outputs in (
            ("s1", [s1 / "adapter_model.safetensors", s1 / "s1_optimizer.pt"]),
            ("listed", [listed / "adapter_model.safetensors", root / "listed/summary.json"]),
        ):
            cmd = training(root, "fb15k237", stage, root, probe_manifest if stage == "listed" else None,
                           task=probe_task, evaluation=probe_eval)
            cmd = sprint.option(sprint.option(cmd, "--save_steps", 0), "--involve_qa_epochs", 1)
            if stage == "listed":
                cmd.append("--allow_underfit_listed")
            stage_run(root, stage, cmd, outputs, lock)
        reload_eval = root / "reload"
        stage_run(root, "reload", test_command(listed, probe_eval, reload_eval, "validation"),
                  [reload_eval / "summary.json", reload_eval / "predictions_correct.jsonl"], lock)
    metadata = hpo.read_json(listed / "run_metadata.json")
    if metadata["trainer_global_step"] != 1 or metadata["candidate_forwards"] != 10:
        raise ValueError("GPU preflight did not execute the ten-way contrastive path")
    write_status("llama_gpu_check", "COMPLETE")


def verification_rows(path, expected, split):
    rows = load_jsonl(path / "predictions_correct.jsonl")
    ids = [row["question_id"] for row in rows if row["split"] == split]
    if len(rows) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{path}: incomplete {split} question IDs")
    return rows


def run_dataset(dataset, checked):
    root = QUEUE / dataset
    spec, _, raw, data, evaluation, task = inputs(dataset)
    actual = (hpo.sha256(task), hpo.sha256(evaluation), hpo.sha256(raw / "train.txt"))
    if actual != (checked["task_sha256"], checked["eval_sha256"], checked["raw_train_sha256"]):
        raise ValueError(f"{dataset} inputs changed since the tokenizer check")
    record = load_graph_record(evaluation)
    ids = {split: {q["question_id"] for q in record["recurrent_questions"] if q["split"] == split}
           for split in ("validation", "test")}
    counts = EXPECTED[dataset]
    relation_text = None if dataset == "nell23k" else raw / "rel2text.json"
    manifest = root / "soft_mix_1/negatives.jsonl"
    s1 = root / "s1_adapter"
    score = root / "confusion"
    sampler = root / "sampler_seed2026"
    s1_outputs = [s1 / n for n in ("adapter_model.safetensors", "adapter_config.json", "s1_optimizer.pt")]
    outputs = {arm: root / arm / variant for arm, variant in ARMS.items()}
    train_cmds = {"s1": training(root, dataset, "s1", root)}
    for arm, variant in ARMS.items():
        train_cmds[arm] = training(root, dataset, variant, root / arm, manifest if variant == "listed" else None)
    commands = {name: cmd for name, cmd, _ in fb.steps(root, raw, data) if name in {"score", "freeze"}}
    score_cmd = commands["score"]
    for flag, value in (("--task_file", task), ("--b1_adapter", outputs["b1"] / "adapter"),
                        ("--model_name", llama.MODEL_SHORT), ("--model_cache_dir", CACHE)):
        score_cmd = sprint.option(score_cmd, flag, value)
    if relation_text is None:
        score_cmd = sprint.option(score_cmd, "--relation_text_file", remove=True)
    tests = [root / "test" / arm for arm in ARMS]
    test_cmds = {arm: test_command(outputs[arm] / "adapter", evaluation, test) for arm, test in zip(ARMS, tests)}
    root.mkdir(exist_ok=True)
    hpo.freeze_text(root / "experiment_plan.json", json.dumps({
        "dataset": dataset, "seed": 2026, "inputs": checked, "model_revision": llama.REVISION,
        "raw_train_sha256": hpo.sha256(raw / "train.txt"),
        "source_sha256": {str(path.relative_to(HNG)): hpo.sha256(path) for path in (
            RUNTIME, Path(__file__).resolve(), HNG / "scripts/train_listed_contrastive.py",
            HNG / "scripts/score_relation_candidates_offline.py", HNG / "scripts/freeze_shared_pool_samplers.py",
            HNG / "scripts/eval_listed_decode.py", *sorted((HNG / "src/hard_negative_grip").glob("*.py")))},
        "commands": {**train_cmds, "score": worker("score", score_cmd), "freeze": commands["freeze"],
                     **{f"test_{arm}": cmd for arm, cmd in test_cmds.items()}},
    }, sort_keys=True, indent=2) + "\n")
    validation = {}
    with gpu(dataset) as lock:
        def run(name, cmd, paths):
            write_status(dataset, "RUNNING", stage=name)
            stage_run(root, name, cmd, paths, lock)

        run("s1", train_cmds["s1"], s1_outputs)
        for arm in ARMS:
            directory = outputs[arm]
            if arm == "soft_mix_1":
                run("score", worker("score", score_cmd),
                    [score / n for n in ("candidate_scores.jsonl", "metadata.json", "qa_eligibility.json", "run_stats.json")])
                sprint.verify_teacher({"score": score, "task": task})
                eligibility = hpo.read_json(score / "qa_eligibility.json")
                if (eligibility["eligible_before_sharding"], eligibility["excluded_count"]) != counts[1:]:
                    raise ValueError("Teacher eligibility changed")
                run("freeze", commands["freeze"], [sampler / "calibrated.jsonl", sampler / "policy.json"])
                hpo.freeze_manifest(sampler / "calibrated.jsonl", manifest,
                                    {"method": "soft_mix", "lambda_scale": 1.0})
                wiring = verify_listed_manifest_wiring(task_path=task, manifest_paths={"soft_mix": manifest},
                    raw_dir=raw, expected_n=counts[0], expected_listed=counts[1], relation_text_file=relation_text)
                w = wiring["variants"]["soft_mix"]
                if (w["n_generation_only_excluded"], w["lambda_min"], w["lambda_max"]) != (counts[2], 1.0, 1.0):
                    raise ValueError("Missing zero-weight rows or non-unit Soft-Mix weight")
            run(arm, train_cmds[arm], [directory / n for n in (
                "adapter/adapter_model.safetensors", "adapter/run_metadata.json",
                "summary.json", "predictions_correct.jsonl")])
            rows = verification_rows(directory, ids["validation"], "validation")
            meta = hpo.read_json(directory / "adapter/run_metadata.json")
            if meta["trainer_global_step"] != 117 or meta["trainer_epoch"] != 3.0 or Path(meta["s1_adapter"]) != s1:
                raise ValueError("S2 did not finish the fixed three-epoch pair")
            validation[arm] = (sprint.nell_harness(rows, split="validation", expected=spec["validation_count"])
                               if dataset == "nell23k" else
                               {"official_em": hpo.read_json(directory / "summary.json")["validation"]["official_em"]})
        hpo.freeze_text(root / "selection.json", json.dumps({
            "checkpoint_step": 117, "validation": validation, "both_arms": True,
        }, indent=2) + "\n")
        scores = []
        for arm, output in zip(ARMS, tests):
            run(f"test_{arm}", test_cmds[arm], [output / "summary.json", output / "predictions_correct.jsonl"])
            rows = verification_rows(output, ids["test"], "test")
            if dataset == "nell23k":
                scores.append(sprint.nell_harness(rows, split="test", expected=spec["test_count"])["official_harness_em"])
            else:
                scores.append(hpo.read_json(output / "summary.json")["test"]["official_em"])
    result = paired.analyze(dataset, root, evaluation, *tests, spec["test_count"], scores,
                            [p / "adapter" for p in outputs.values()])
    timings = {p.stem: hpo.read_json(p)["wall_seconds"] for p in (root / "completed").glob("*.json")}
    result.update(delta_em=scores[1] - scores[0], paper_llama_em=PAPER[dataset], validation=validation,
                  seed=2026, stage_wall_seconds=timings)
    hpo.write_json(root / "comparison.json", result)
    write_status(dataset, "COMPLETE", delta_em=result["delta_em"])
    return result


def final_report(comparisons):
    import numpy as np

    ordered = sorted(comparisons, key=lambda dataset: comparisons[dataset]["mcnemar"]["p_value"])
    adjusted = 0.0
    for rank, dataset in enumerate(ordered):
        row = comparisons[dataset]
        p = row["mcnemar"]["p_value"]
        adjusted = max(adjusted, min(1.0, p * (len(ordered) - rank)))
        row["mcnemar"]["holm_p_value"] = adjusted
        # Each question contributes -1, 0, or 1 to the paired accuracy change.
        # Counts drawn from these three values give the paired item bootstrap.
        n, gains, losses = row["n"], row["gains"], row["regressions"]
        draw = np.random.default_rng(2026).multinomial(n, [losses / n, (n - gains - losses) / n, gains / n], 10000)
        interval = np.quantile((draw[:, 2] - draw[:, 0]) / n, [0.025, 0.975]).tolist()
        row["delta_em_item_bootstrap95"] = interval
    hpo.write_json(QUEUE / "comparison.json", {
        "model": llama.MODEL_ID, "model_revision": llama.REVISION, "datasets": comparisons,
        "delta_interval_unit": "test question", "bootstrap_seed": 2026, "bootstrap_repetitions": 10000,
    })


def main():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Launch this queue inside tmux")
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = "qwen_analysis"
        try:
            write_status(active, "RUNNING")
            try:
                qwen_analysis()
            except (ValueError, OSError, KeyError) as exc:
                write_status(active, "INTERRUPTED", reason=str(exc))
            else:
                write_status(active, "COMPLETE")
            active = "llama_tokenizer_check"
            write_status(active, "RUNNING")
            report = cpu_preflight()
            write_status(active, "COMPLETE")
            active = "wn_independent_s1"
            wait_for_wn()
            active = "llama_weights"
            wait_for_disk(active, lambda: disk_needed("fb15k237", QUEUE / "fb15k237", include_probe=True)
                          + llama.remaining_weight_bytes(MODEL))
            write_status(active, "RUNNING")
            llama.fetch_files(MODEL, llama.WEIGHTS)
            write_status(active, "COMPLETE")
            active = "llama_gpu_check"
            gpu_probe(report)
            comparisons = {}
            for dataset in ORDER:
                active = dataset
                wait_for_disk(active, lambda: disk_needed(dataset, QUEUE / dataset))
                comparisons[dataset] = run_dataset(dataset, report[dataset])
            final_report(comparisons)
            print("[queue] all four Llama comparisons complete", flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            write_status(active, "INTERRUPTED", reason=str(exc))
            raise


if __name__ == "__main__":
    main()
