#!/usr/bin/env python3
"""One bounded continuation: audit Violation-only, run Profile-only, then stop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

HNG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HNG/"scripts"), str(HNG/"src")]
import run_fb_violation_only_control as component
from hard_negative_grip.matched_control import audit_predictions, paired_result, read_json, sha256, write_csv, write_json

base = component.base


def require(condition, message):
    base.require(condition, message)


def session_alive(name):
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def gpu_idle():
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
        memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20)
        if apps.returncode != 0 or memory.returncode != 0:
            return False
        return not apps.stdout.strip() and max(float(v) for v in memory.stdout.splitlines()) < 6000
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def progress(run):
    status = read_json(run/"status.json")
    result = {key: status.get(key) for key in ("stage", "status", "at", "elapsed_seconds", "error")}
    log = Path(status["log"]) if status.get("log") else run/"controller.log"
    if log.is_file():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size-12000))
            tail = stream.read().decode(errors="replace")
        steps = re.findall(r"(\d+)/120\s*\[", tail)
        generated = re.findall(r"\[generate\]\s+(\d+)/(\d+)", tail)
        if steps: result["optimizer_steps_observed"] = int(steps[-1])
        if generated: result["generated_observed"] = list(map(int, generated[-1]))
        result["log_mtime"] = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat()
        if status.get("status") == "running" and time.time()-log.stat().st_mtime > 1800:
            result["warning"] = "No log change for over 30 minutes; inspect the run. No automatic retry or next launch."
    return result


def audit_completed(run, expected_id, expected_contract, variant):
    """Read-only remeasurement and archive validation; never reruns inference."""
    require(read_json(run/"status.json")["status"] == "completed", "experiment not complete")
    require(sha256(run/"contract.json") == expected_contract, "run contract differs from queue plan")
    contract = base.verify(run)
    require(contract["experiment_id"] == expected_id and contract["design"]["variant"] == variant, "wrong component run")
    require(contract["design"]["dataset"] == "fb15k237_2" and contract["design"]["seed"] == 2026, "wrong dataset or seed")
    actual = read_json(run/"training/run_config.json")
    component.validate_mode(run)
    require(actual["counterfactual_sampling"] == component.settings(variant)["sampler"], "wrong sampler at completion")
    state = read_json(run/"training/trainer_listed/checkpoint-32/trainer_state.json")
    meta = read_json(run/"training/listed/adapter/run_metadata.json")
    require(state["global_step"] == meta["trainer_global_step"] == 32 and
            state["max_steps"] == meta["trainer_max_steps"] == 120, "training budget differs")
    samples = base.eval_samples(component.load_graph_record(run/"inputs/test.json"))
    normalize = base.normalization(run/"code_snapshot")
    paths = [run/"reference/predictions.jsonl", run/"reference/random9_predictions.jsonl", run/"evaluation/test/predictions.jsonl"]
    indexed = [audit_predictions(base.jsonl(path), samples, normalize) for path in paths]
    measured, _, _ = component.three_way_summary(*indexed, variant=variant)
    result = read_json(run/"paper/result.json")
    require(result["experiment_id"] == expected_id and result["status"] == "completed", "wrong result identity")
    for key, value in measured.items():
        require(result[key] == value, "recomputed comparison differs: "+key)
    require(measured["n"] == 10000 and measured["full_correct"] == 8990 and measured["random9_correct"] == 8648,
            "reference cohort differs")
    weights = {name: sha256(run/"training/listed/adapter"/name) for name in ("adapter_config.json", "adapter_model.safetensors")}
    require(read_json(run/"selection.json")["adapter_sha256"] == weights and
            read_json(run/"evaluation/test/reuse_identity.json")["adapter"] == weights, "test weights differ from selection")
    summary = read_json(run/"evaluation/test/summary.json")["all"]
    require(summary["count"] == 10000 and abs(summary["em"]-measured[variant+"_em"]) < 1e-12, "test summary mismatch")
    manifest = read_json(run/"artifact_manifest.json")
    for entry in manifest["artifacts"]:
        path = run/entry["path"]
        require(path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                "archive mismatch: "+entry["path"])
    return {"status": "pass", "at": base.now(), "experiment_id": expected_id, "variant": variant,
            "run_dir": str(run), "contract_sha256": expected_contract,
            "archive_sha256": sha256(run/"artifact_manifest.json"), "archive_files_verified": len(manifest["artifacts"]),
            "correctness_flags_recomputed": 30000, "adapter_sha256": weights, "result": measured}


def launch_next(queue, plan):
    """At most one launch; a persisted intent is never silently retried."""
    receipt_path = queue/"launch_receipt.json"
    session = plan["next_session"]
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt["next_contract_sha256"] == plan["next_contract_sha256"], "launch receipt belongs to another run")
        require(receipt["status"] == "launched", "prior launch intent is incomplete; inspect before retry")
        return receipt
    run = Path(plan["next_run"])
    require(not session_alive(session), "unowned next-run session exists; refusing duplicate launch")
    state = read_json(run/"status.json")
    cpu_ready = ((state.get("stage") == "prepared" and state.get("status") == "ready") or
                 (state.get("stage") == "cli_preflight" and state.get("status") == "finished"))
    require(cpu_ready and not list((run/"training/trainer_listed").glob("checkpoint-*")) and
            not (run/"training/listed/adapter").exists(), "next run already started or failed")
    require(sha256(run/"contract.json") == plan["next_contract_sha256"], "prepared next contract changed")
    contract = base.verify(run)
    require(contract["design"]["variant"] == "profile_only", "only Profile-only is authorized as continuation")
    require((run/"checks/cli.json").is_file(), "Profile-only CLI was not preflighted")
    cli = read_json(run/"checks/cli.json")
    require(cli.get("model_loaded") is False and cli.get("n_qa") == 12136 and cli.get("listed_qa") == 3109,
            "Profile-only CPU wiring check is incomplete")
    component.validate_mode(run)
    argv = [str(base.PYTHON), str(run/"code_snapshot"/base.REL_HNG/"scripts/run_fb_violation_only_control.py"),
            "run", "--config", str(run/"inputs/design.json"), "--run-dir", str(run)]
    receipt = {"at": base.now(), "status": "intent", "next_contract_sha256": plan["next_contract_sha256"],
               "session": session, "argv": argv}
    write_json(receipt_path, receipt)
    command = "exec "+shlex.join(argv)+" >> "+shlex.quote(str(run/"controller.log"))+" 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", plan["repository_root"], command], check=True)
    receipt.update(status="launched", launched_at=base.now())
    write_json(receipt_path, receipt)
    base.append_event(queue, {"stage": "launch_profile_only", **receipt})
    return receipt


def four_way_export(queue, plan, violation_audit, profile_audit):
    violation = Path(plan["prerequisite_run"])
    profile = Path(plan["next_run"])
    vc, pc = read_json(violation/"contract.json"), read_json(profile/"contract.json")
    for relative in ("inputs/task.json", "inputs/test.json", "inputs/stage1_adapter/adapter_model.safetensors"):
        require(vc["immutable_artifacts"][relative]["sha256"] == pc["immutable_artifacts"][relative]["sha256"],
                "four-way inputs differ: "+relative)
    samples = base.eval_samples(component.load_graph_record(violation/"inputs/test.json"))
    require(len(samples) == 10000, "four-way table requires the complete 10,000-question cohort")
    normalize = base.normalization(violation/"code_snapshot")
    paths = {"random9": violation/"reference/random9_predictions.jsonl",
             "profile_only": profile/"evaluation/test/predictions.jsonl",
             "violation_only": violation/"evaluation/test/predictions.jsonl",
             "full": violation/"reference/predictions.jsonl"}
    predictions = {name: audit_predictions(base.jsonl(path), samples, normalize) for name,path in paths.items()}
    labels = {"random9": "Random-9", "profile_only": "Profile-only-9", "violation_only": "Violation-only-9", "full": "Full SC-CL-9"}
    mask = {"random9": (0,0), "profile_only": (1,0), "violation_only": (0,1), "full": (1,1)}
    metrics = [{"method": labels[name], "seed": 2026, "structure_signal": mask[name][0], "b1_signal": mask[name][1],
                "n": len(rows), "correct": sum(r["correct"] for r in rows.values()),
                "em_percent": 100*sum(r["correct"] for r in rows.values())/len(rows)} for name,rows in predictions.items()]
    paired = [{"question_id": sample["question_id"],
               **{name+"_correct": int(rows[sample["question_id"]]["correct"]) for name,rows in predictions.items()}}
              for sample in samples]
    comparisons = {}
    for left,right in (("full","violation_only"),("full","profile_only"),("violation_only","random9"),("profile_only","random9")):
        summary,_ = paired_result(predictions[left], predictions[right])
        comparisons[left+"_minus_"+right] = {"delta_pp": summary["delta_sccl_minus_random9_pp"],
            "net_correct": summary["net_correct"], "paired_counts": summary["paired_counts"]}
    output = queue/"paper"
    write_csv(output/"metrics.csv", metrics)
    write_csv(output/"paired_four_methods.csv", paired)
    result = {"status": "completed", "at": base.now(), "dataset": "fb15k237_2", "seed": 2026,
              "exp_id": "SC_COMPONENT_FOUR_ROW_FB_SEED2026", "section_id": "experiments",
              "paper_role": "claim-carrying", "claim_links": ["C2", "C4", "C5"],
              "methods": metrics, "comparisons": comparisons, "violation_audit": violation_audit,
              "profile_audit": profile_audit, "raw_predictions": {k:str(v) for k,v in paths.items()},
              "scope": "single-seed component effects at the previously selected Full configuration",
              "limitations": ["no seed-level stability or statistical synergy claim", "no equal-HPO comparison",
                              "464 QA repairs and prior test exposure retained"], "next_action": "write and stop; no further training is authorized"}
    write_json(output/"result.json", result)
    end = chr(92)*2
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method & Structure & B1 & Correct & EM (\\%) "+end, "\\midrule"]
    tex += [f"{r['method']} & {r['structure_signal']} & {r['b1_signal']} & {r['correct']} & {r['em_percent']:.4f} "+end for r in metrics]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (output/"table.tex").write_text("\n".join(tex)+"\n")
    (output/"RESULTS.md").write_text(
        "# FB single-seed component ablations\n\n"+
        "\n".join(f"- {r['method']}: {r['correct']}/10000, EM={r['em_percent']:.4f}%" for r in metrics)+
        "\n\nFull-minus-component comparisons and rescue/regression counts are in result.json. "
        "All four rows describe one fixed configuration and one seed; they do not establish population-level interaction or stability.\n")
    dest = Path(plan["repository_root"])/"11_paper/artifacts/fb_component_ablation_seed2026"
    dest.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir(): shutil.copy2(path, dest/path.name)
    (dest/".gitattributes").write_text("* -text\n")
    return result


def tick(queue, plan):
    pre, nxt = Path(plan["prerequisite_run"]), Path(plan["next_run"])
    observed = progress(pre)
    require(observed["status"] != "failed", "Violation-only failed; continuation stopped")
    if observed["status"] != "completed":
        require(session_alive(plan["prerequisite_session"]), "Violation-only session exited before completion")
        return {"stage": "waiting_violation_only", "prerequisite": observed}
    if session_alive(plan["prerequisite_session"]):
        return {"stage": "waiting_violation_controller_exit", "prerequisite": observed}
    audit_path = queue/"violation_completion_audit.json"
    if audit_path.exists():
        va = read_json(audit_path)
        require(va["contract_sha256"] == plan["prerequisite_contract_sha256"] and
                va["archive_sha256"] == sha256(pre/"artifact_manifest.json"), "prerequisite archive changed after audit")
    else:
        va = audit_completed(pre, plan["prerequisite_id"], plan["prerequisite_contract_sha256"], "violation_only")
        write_json(audit_path, va)
    if not (queue/"launch_receipt.json").exists():
        if not gpu_idle(): return {"stage": "waiting_gpu", "prerequisite_audit": "pass"}
    launch_next(queue, plan)
    current = progress(nxt)
    require(current["status"] != "failed", "Profile-only failed; no automatic retry")
    if current["status"] == "completed":
        if session_alive(plan["next_session"]): return {"stage": "waiting_profile_controller_exit", "next": current}
        pa = audit_completed(nxt, plan["next_id"], plan["next_contract_sha256"], "profile_only")
        write_json(queue/"profile_completion_audit.json", pa)
        result = four_way_export(queue, plan, va, pa)
        return {"stage": "completed", "result": result}
    require(session_alive(plan["next_session"]), "Profile-only session disappeared; inspect launch receipt before any retry")
    return {"stage": "monitoring_profile_only", "next": current, "prerequisite_audit": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    queue = args.plan.resolve().parent
    require(Path(sys.executable).resolve() == base.PYTHON.resolve(), "use guardenv Python")
    require(plan["max_additional_gpu_runs"] == 1 and 1 <= plan["poll_seconds"] <= 60, "bounded continuation policy changed")
    for relative, digest in plan["watcher_code_sha256"].items():
        require(sha256(queue/relative) == digest, "frozen watcher code changed: "+relative)
    with base.locked(queue):
        if (queue/"status.json").exists() and read_json(queue/"status.json").get("stage") == "completed":
            print("already completed; no additional experiment", flush=True)
            return
        while True:
            try:
                state = {"at": base.now(), **tick(queue, plan)}
                write_json(queue/"status.json", state)
                base.append_event(queue, state)
                print(json.dumps(state, ensure_ascii=False), flush=True)
                if state["stage"] == "completed": return
            except BaseException as error:
                state = {"at": base.now(), "stage": "blocked", "error": repr(error), "next_run_retried": False}
                write_json(queue/"status.json", state)
                base.append_event(queue, state)
                print(json.dumps(state, ensure_ascii=False), flush=True)
                raise
            time.sleep(plan["poll_seconds"])


if __name__ == "__main__":
    main()
