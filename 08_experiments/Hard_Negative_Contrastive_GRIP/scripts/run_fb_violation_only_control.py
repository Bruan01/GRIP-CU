#!/usr/bin/env python3
"""Matched single-component ablations; preparation never starts a GPU run.

The historical entry name is retained. Explicit designs select violation_only
or profile_only; an already running frozen controller is never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

HNG = Path(__file__).resolve().parents[1]
ROOT = HNG.parents[1]
sys.path[:0] = [str(HNG/"scripts"), str(HNG/"src")]
import run_fb_random9_control as base
from hard_negative_grip.counterfactuals import (
    STRUCTURE_COUNTERFACTUAL_SOURCE, load_train_relation_profiles,
    sample_structure_counterfactual_group,
)
from hard_negative_grip.frozen_random import score_groups, pool_digest
from hard_negative_grip.matched_control import (
    audit_predictions, paired_result, parameter_diff, read_json, sha256,
    training_command, write_csv, write_json,
)
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import (
    build_qa_assets_from_task_texts, load_counterfactual_manifest, load_graph_record,
    original_question_ids_from_payload, validate_structure_counterfactual_manifest,
)

EXPECTED_SAMPLER = {"uniform_mix": 1/3, "structure_weight": 0.0,
                    "violation_weight": 1.0, "temperature": .25}
VARIANTS = {
    "violation_only": {"sampler": EXPECTED_SAMPLER, "short": "violation", "label": "Violation-only-9", "item": "SC_CTRL_VIOLATION", "claim_links": ["C2", "C4"]},
    "profile_only": {"sampler": dict(EXPECTED_SAMPLER, structure_weight=1.0, violation_weight=0.0),
                     "short": "profile", "label": "Profile-only-9", "item": "SC_CTRL_PROFILE", "claim_links": ["C2", "C5"]},
}


def require(condition, message):
    base.require(condition, message)


def settings(variant):
    require(variant in VARIANTS, "unregistered component ablation")
    return VARIANTS[variant]


def run_variant(run):
    contract = read_json(run/"contract.json")
    return contract.get("design", {}).get("variant", "violation_only")


def make_training_argv(reference, run, model_cache, variant="violation_only"):
    sampler = settings(variant)["sampler"]
    cmd = training_command(str(base.PYTHON), run/"code_snapshot"/base.REL_HNG/"scripts/train_listed_contrastive.py",
            reference, inputs=run/"inputs", bank=run/"bank"/(variant+".jsonl"),
            output=run/"training", model_cache=Path(model_cache))
    for option in ("--listed_negative_source", "--negative_sampling_mode"):
        cmd[cmd.index(option)+1] = "structure_counterfactual"
    cmd[cmd.index("--random_manifest")] = "--counterfactual_manifest"
    cmd += ["--counterfactual_metadata", str(run/"bank/metadata.json")]
    for name, value in sampler.items():
        cmd += ["--counterfactual_"+name, str(value)]
    return cmd


def ablate_group(group, profiles, vocabulary, parent_row, pool_row, parent_parameters, variant="violation_only"):
    """Replay the original bank before removing exactly one registered signal."""
    qid = parent_row["question_id"]
    require(group and all(r.get("qa_id") == qid and r.get("split") == "train" for r in group),
            "score group identity/split mismatch")
    names = [r["candidate_relation"] for r in group]
    require(len(names) == len(set(names)) and set(names) == set(vocabulary), "incomplete score vocabulary")
    legal = pool_row["legal_relations"]
    require(pool_row["question_id"] == qid and pool_digest(legal) == pool_row["pool_sha256"],
            "pool audit identity mismatch")
    require({r["candidate_relation"] for r in group if r.get("is_valid_negative") is True} == set(legal),
            "score legality differs from the frozen parent pool")
    known = set(pool_row["excluded_train_relations"])
    gold = parent_row["matched_train_relation"]
    require(set(legal) == set(vocabulary)-known-{gold}, "parent pool is not the train-filtered pool")
    require(all(r.get("matched_train_relation") == gold and r.get("gold_relation") == parent_row["positive_relation"]
                and r.get("is_gold") is (r["candidate_relation"] == gold)
                and r.get("is_valid_negative") is (r["candidate_relation"] in legal) for r in group),
            "gold or train-known filtering changed")
    common = dict(total_k=9, seed=2026, known_relations=known)
    replay, full_audit = sample_structure_counterfactual_group(group, profiles, **common, **parent_parameters)
    require(replay["negative_relations"] == parent_row["negative_relations"], "original Full bank did not replay")
    control, audit = sample_structure_counterfactual_group(group, profiles, **common, **settings(variant)["sampler"])
    control["ablation_variant"] = variant
    control["parent_full_negative_relations"] = parent_row["negative_relations"]
    require(control["positive_relation"] == parent_row["positive_relation"], "ablation gold mismatch")
    require(set(control["negative_relations"]).issubset(legal), "ablation left the legal pool")
    overlap = len(set(control["negative_relations"]) & set(parent_row["negative_relations"]))
    diagnostic = {"question_id": qid, "pool_size": len(legal), "overlap_k": overlap,
        "same_set": overlap == 9, "same_order": control["negative_relations"] == parent_row["negative_relations"],
        "full_structure_mean": full_audit["selected_mean_structure_similarity"],
        variant+"_structure_mean": audit["selected_mean_structure_similarity"],
        "full_violation_mean": full_audit["selected_mean_score_violation"],
        variant+"_violation_mean": audit["selected_mean_score_violation"],
        "original_full_replayed": True, "train_only_pool_verified": True}
    return control, audit, diagnostic


def require_intervention(diagnostics):
    require(diagnostics and any(not row["same_set"] for row in diagnostics),
            "removing a signal changed no candidate sets; do not spend GPU on an identical realization")


def validate_mode(run):
    variant = run_variant(run)
    actual = read_json(run/"training/run_config.json")
    require(actual["listed_negative_source"] == "structure_counterfactual"
            and actual["negative_sampling_mode"] == "structure_counterfactual", "wrong bank loader")
    require(actual["counterfactual_sampling"] == settings(variant)["sampler"], "sampler does not match the registered variant")
    require(Path(actual["counterfactual_manifest"]).resolve() == (run/"bank"/(variant+".jsonl")).resolve(),
            "training points at a different candidate bank")
    parameter_diff(read_json(run/"reference/run_config.json"), actual)


def prepare(run, config):
    if (run/"contract.json").exists():
        contract = base.verify(run)
        require(contract["design_sha256"] == sha256(config), "design changed after preparation")
        return contract
    require(not (run/"inputs").exists(), "partial preparation retained; use a fresh run directory")
    design = read_json(config)
    variant = design["variant"]
    sampler = settings(variant)["sampler"]
    require(design["sampler"] == sampler and design["seed"] == 2026 and design["optimizer_steps"] == 32,
            "only registered single-seed component experiments are supported")
    source = Path(design["source_control_run"])
    require(sha256(source/"contract.json") == design["source_contract_sha256"], "parent contract changed")
    require(sha256(source/"artifact_manifest.json") == design["source_archive_sha256"], "parent archive changed")
    require(read_json(source/"status.json")["status"] == "completed", "parent run is not complete")
    source_contract = base.verify(source)
    # Preparation and the frozen training execution must use identical sampler/data code.
    for relative in ("src/hard_negative_grip/counterfactuals.py", "src/hard_negative_grip/task_file.py"):
        require(sha256(HNG/relative) == sha256(source/"code_snapshot"/base.REL_HNG/relative),
                "sampler/data implementation changed since the completed control")
    base.append_event(run, {"stage": "prepare", "status": "running"})
    artifacts = {}
    base.freeze(run, config, "inputs/design.json", artifacts)
    for relative, record in source_contract["immutable_artifacts"].items():
        if relative == "inputs/design.json" or relative.startswith("bank/"):
            continue
        if not relative.startswith(("inputs/", "reference/", "code_snapshot/")):
            continue
        target = run/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith("code_snapshot/"):
            shutil.copy2(source/relative, target)
        else:
            target.symlink_to((source/relative).resolve())
        artifacts[relative] = {"sha256": record["sha256"], "bytes": record["bytes"],
                               "source": str(source/relative), "borrowed": not relative.startswith("code_snapshot/")}
    driver_rel = Path("code_snapshot")/base.REL_HNG/"scripts"/Path(__file__).name
    base.freeze(run, Path(__file__), driver_rel, artifacts)
    archive = {r["path"]: r for r in read_json(source/"artifact_manifest.json")["artifacts"]}
    for src_rel, dst in (("evaluation/test/predictions.jsonl", "reference/random9_predictions.jsonl"),
                         ("paper/result.json", "reference/random9_result.json"),
                         ("bank/pool_audit.jsonl", "reference/pool_audit.jsonl")):
        target = run/dst
        target.symlink_to((source/src_rel).resolve())
        require(sha256(target) == archive[src_rel]["sha256"], "completed comparator artifact changed")
        artifacts[dst] = {"sha256": archive[src_rel]["sha256"], "bytes": target.stat().st_size,
                          "source": str(source/src_rel), "borrowed": True}
    parent_bank = load_counterfactual_manifest(run/"reference/counterfactual_bank.jsonl")
    pools = {row["question_id"]: row for row in base.jsonl(run/"reference/pool_audit.jsonl")}
    require(set(pools) == set(parent_bank) and len(pools) == design["listed_qa_count"], "parent coverage changed")
    params = read_json(run/"reference/metadata.json")["parameters"].copy()
    params.pop("seed")
    require(params == dict(EXPECTED_SAMPLER, structure_weight=1.0), "parent sampler parameters differ")
    wiring = read_json(source/"bank/wiring.json")
    scores = Path(wiring["source_scores"])
    require(sha256(scores) == wiring["source_scores_sha256"], "B1 score dump changed")
    vocabulary = load_train_relation_order(run/"inputs/raw", relation_text_file=run/"inputs/raw/rel2text.json")
    profiles = load_train_relation_profiles(run/"inputs/raw", relation_text_file=run/"inputs/raw/rel2text.json")
    bank, diagnostics = {}, []
    (run/"bank").mkdir()
    with (run/"bank"/(variant+".jsonl")).open("w", encoding="utf-8") as out, \
         (run/"bank/sampler_audit.jsonl").open("w", encoding="utf-8") as audit_out:
        for group in score_groups(scores):
            qid = group[0]["qa_id"]
            require(qid in parent_bank and qid not in bank, "duplicate or unexpected score group")
            row, audit, diagnostic = ablate_group(group, profiles, vocabulary, parent_bank[qid], pools[qid], params, variant)
            bank[qid] = row
            diagnostics.append(diagnostic)
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
            audit_out.write(json.dumps(audit, ensure_ascii=False)+"\n")
            if len(bank) % 100 == 0:
                write_json(run/"status.json", {"stage": "prepare_bank", "status": "running", "rows": len(bank), "at": base.now()})
    require(set(bank) == set(parent_bank), "ablation bank is incomplete")
    require_intervention(diagnostics)
    validate_structure_counterfactual_manifest(bank, k=9)
    payload = read_json(run/"inputs/task.json")
    texts = payload["qa_samples"]
    ids = original_question_ids_from_payload(payload, len(texts))
    common = dict(seed=2026, relation_order=vocabulary, listed_negative_source="structure_counterfactual",
                  question_ids=ids, eligible_question_ids=set(bank))
    _, before = build_qa_assets_from_task_texts(texts, counterfactual_manifest=parent_bank, **common)
    _, after = build_qa_assets_from_task_texts(texts, counterfactual_manifest=bank, **common)
    for left, right in zip(before, after):
        require({k:v for k,v in left.items() if k != "listed_relations"} ==
                {k:v for k,v in right.items() if k != "listed_relations"}, "non-candidate QA metadata changed")
    require(len(texts) == design["qa_count"], "full generation QA count changed")
    metadata = {"source": STRUCTURE_COUNTERFACTUAL_SOURCE, "variant": variant, "requested_k": 9,
                "parameters": dict(sampler, seed=2026), "rows_written": len(bank), "rows_failed": 0,
                "source_scores_sha256": wiring["source_scores_sha256"], "parent_full_replayed_rows": len(bank)}
    write_json(run/"bank/metadata.json", metadata)
    write_csv(run/"bank/diagnostics.csv", diagnostics)
    changed = sum(not d["same_set"] for d in diagnostics)
    prep = {"n_qa": len(texts), "listed_qa": len(bank), "k": 9, "seed": 2026,
            "parent_full_bank_replayed": True, "train_only_pool_match": True,
            "qa_text_and_non_candidate_metadata_unchanged": True,
            "changed_candidate_set_questions": changed, "changed_candidate_set_fraction": changed/len(bank),
            "mean_shared_negatives_of_9": sum(d["overlap_k"] for d in diagnostics)/len(bank),
            "performance_results_available": False, "new_training_started": False}
    write_json(run/"bank/preparation_audit.json", prep)
    for path in (run/"bank").iterdir():
        artifacts[str(path.relative_to(run))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    reference = read_json(run/"reference/run_config.json")
    cmd = make_training_argv(reference, run, source_contract["model_cache"], variant)
    contract = {"schema": "fb-component-control-v1", "experiment_id": design["experiment_id"],
                "created_at": base.now(), "repository_root": str(ROOT), "design": design,
                "design_sha256": sha256(config), "immutable_artifacts": artifacts, "training_argv": cmd,
                "model_cache": source_contract["model_cache"], "expected_optimizer_steps": 32,
                "expected_scheduler_max_steps": 120, "selection": "fixed_step32_before_one_full_test",
                "validation_policy": "not_run_by_registered_budget_plan", "sampler": sampler,
                "borrowed_source_run": str(source), "borrowed_inputs_must_be_retained": True}
    write_json(run/"contract.json", contract)
    write_json(run/"execution_plan.json", {"training_argv": cmd, "stages": ["cli_check", "train32", "one_full_test", "report"],
                                         "full_validation": "not_requested"})
    write_json(run/"environment_prepare.json", base.environment(run))
    write_json(run/"status.json", {"stage": "prepared", "status": "ready", "at": base.now(),
                                   "training_complete": False, "test_complete": False})
    base.append_event(run, {"stage": "prepare", "status": "completed", "audit": prep})
    return contract


def validate_training(run, contract):
    validate_mode(run)
    state = read_json(run/"training/trainer_listed/checkpoint-32/trainer_state.json")
    meta = read_json(run/"training/listed/adapter/run_metadata.json")
    require(state["global_step"] == meta["trainer_global_step"] == contract["expected_optimizer_steps"] == 32
            and state["max_steps"] == meta["trainer_max_steps"] == contract["expected_scheduler_max_steps"] == 120,
            "training exceeded or changed the fixed optimization budget")
    rows = parameter_diff(read_json(run/"reference/run_config.json"), read_json(run/"training/run_config.json"))
    variant = contract.get("design", {}).get("variant", "violation_only")
    rows = [{"parameter": r["parameter"], "full_reference": r["reference"],
             variant: r["random_control"], "equal": r["equal"]} for r in rows]
    write_csv(run/"paper/config_match.csv", rows)
    return state


def train(run, contract):
    base.cli_check(run, contract)
    validate_mode(run)
    if (run/"training/listed/adapter/run_metadata.json").exists():
        validate_training(run, contract)
        return
    checkpoints = list((run/"training/trainer_listed").glob("checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        require(read_json(latest/"trainer_state.json")["global_step"] <= 32, "resume exceeds registered budget")
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
            require((latest/name).exists(), "incomplete resume checkpoint: "+str(latest/name))
    base.execute(run, "train", contract["training_argv"])
    validate_training(run, contract)


def evaluate(run, contract):
    validate_training(run, contract)
    adapter = run/"training/listed/adapter"
    weights = {name: sha256(adapter/name) for name in ("adapter_config.json", "adapter_model.safetensors")}
    selected = {"selected_at": base.now(), "selection_rule": contract["selection"], "step": 32,
                "adapter_sha256": weights, "validation_run": False, "search_budget": 0, "reopen_after_test": False}
    if (run/"selection.json").exists():
        require(read_json(run/"selection.json")["adapter_sha256"] == weights, "frozen weights changed")
    else:
        write_json(run/"selection.json", selected)
    base.execute(run, "eval_test", [str(base.PYTHON), str(run/"code_snapshot"/base.REL_HNG/"scripts/evaluate_adapter.py"),
        "--adapter", str(adapter), "--eval_file", str(run/"inputs/test.json"), "--output_dir", str(run/"evaluation/test"),
        "--model_name", "qwen-7b", "--model_cache_dir", contract["model_cache"], "--dtype", "bfloat16",
        "--gen_max_length", "32", "--progress_every", "100", "--resume", "--label",
        "fb_"+contract.get("design", {}).get("variant", "violation_only")+"_seed2026_fixed32_test"])


def three_way_summary(full, random, component, variant="violation_only"):
    short = settings(variant)["short"]
    full_v, pairs = paired_result(full, component)
    v_random, _ = paired_result(component, random)
    result = {"n": full_v["n"], "full_correct": full_v["sccl_correct"],
        variant+"_correct": full_v["random9_correct"], "random9_correct": v_random["random9_correct"],
        "full_em": full_v["sccl_em"], variant+"_em": full_v["random9_em"], "random9_em": v_random["random9_em"],
        "delta_full_minus_"+short+"_pp": full_v["delta_sccl_minus_random9_pp"],
        "delta_"+short+"_minus_random_pp": v_random["delta_sccl_minus_random9_pp"],
        "net_correct_full_minus_"+short: full_v["net_correct"],
        "paired_full_vs_"+short: full_v["paired_counts"], "paired_"+short+"_vs_random": v_random["paired_counts"]}
    converted = [{"question_id": row["question_id"], "full_correct": row["sccl_correct"],
                  variant+"_correct": row["random9_correct"],
                  "outcome_full_vs_"+short: row["outcome_sccl_vs_random9"], "question_sha256": row["question_sha256"]}
                 for row in pairs]
    _, second = paired_result(component, random)
    second = [{"question_id": row["question_id"], variant+"_correct": row["sccl_correct"],
               "random9_correct": row["random9_correct"], "outcome_"+short+"_vs_random9": row["outcome_sccl_vs_random9"],
               "question_sha256": row["question_sha256"]} for row in second]
    return result, converted, second


def report(run, contract):
    variant = contract.get("design", {}).get("variant", "violation_only")
    spec = settings(variant)
    short = spec["short"]
    state = validate_training(run, contract)
    samples = base.eval_samples(load_graph_record(run/"inputs/test.json"))
    normalize = base.normalization(run/"code_snapshot")
    paths = (run/"reference/predictions.jsonl", run/"reference/random9_predictions.jsonl", run/"evaluation/test/predictions.jsonl")
    indexed = [audit_predictions(base.jsonl(path), samples, normalize) for path in paths]
    weights = {name: sha256(run/"training/listed/adapter"/name) for name in ("adapter_config.json", "adapter_model.safetensors")}
    require(read_json(run/"selection.json")["adapter_sha256"] == weights
            and read_json(run/"evaluation/test/reuse_identity.json")["adapter"] == weights, "prediction/model identity mismatch")
    result, pairs, second = three_way_summary(*indexed, variant=variant)
    require(result["n"] == 10000 and result["full_correct"] == 8990 and result["random9_correct"] == 8648,
            "completed reference counts changed")
    summary = read_json(run/"evaluation/test/summary.json")["all"]
    require(summary["count"] == result["n"] and abs(summary["em"]-result[variant+"_em"]) < 1e-12,
            "test summary differs from recomputed predictions")
    result.update({"experiment_id": contract["experiment_id"], "at": base.now(), "status": "completed", "seed": 2026,
        "variant": variant, "paper_role": "claim-carrying", "section_id": "experiments", "item_id": spec["item"], "claim_links": spec["claim_links"],
        "scope": contract["design"]["scope"], "validation_status": contract["validation_policy"],
        "limitations": ["single seed, previously SC-CL-selected hyperparameters", "no equal-HPO or statistical stability claim",
                        "this three-row report alone establishes no interaction/synergy claim", "historical test exposure and 464 QA repairs retained"],
        "next_action": "write the observed positive, null or negative component effect without reopening this experiment"})
    write_json(run/"paper/result.json", result)
    write_csv(run/("paper/paired_full_vs_"+short+".csv"), pairs)
    write_csv(run/("paper/paired_"+short+"_vs_random.csv"), second)
    metrics = [{"model": label, "seed": 2026, "n": result["n"], "correct": result[key], "em_percent": 100*result[key]/result["n"]}
               for label, key in (("Random-9", "random9_correct"), (spec["label"], variant+"_correct"), ("Full SC-CL-9", "full_correct"))]
    write_csv(run/"paper/metrics.csv", metrics)
    fields = sorted({key for row in state["log_history"] for key in row})
    write_csv(run/"paper/training_history.csv", [{key:row.get(key, "") for key in fields} for row in state["log_history"]])
    end_row = chr(92)*2
    rows = ["\\begin{tabular}{lrr}", "\\toprule", "Method & Correct / n & EM (\\%) "+end_row, "\\midrule"]
    rows += [f"{row['model']} & {row['correct']}/{row['n']} & {row['em_percent']:.4f} "+end_row for row in metrics]
    rows += ["\\bottomrule", "\\end{tabular}"]
    (run/"paper/table.tex").write_text("\n".join(rows)+"\n")
    (run/"paper/RESULTS.md").write_text(
        "# Fixed-configuration component ablation: "+variant+"\n\n"
        + f"Full − {spec['label']} = {result['delta_full_minus_'+short+'_pp']:+.4f} pp; "
        + f"{spec['label']} − Random = {result['delta_'+short+'_minus_random_pp']:+.4f} pp.\n\n"
        + "One seed at the existing SC-CL-selected configuration; no standalone structure or synergy/stability claim. "
        + "The fixed step32 model was tested once. Full validation was omitted by the predeclared budget plan.\n")
    base.inventory(run)
    destination = Path(contract["repository_root"])/("11_paper/artifacts/fb_"+variant+"_control_seed2026")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"result.json").exists():
        require(read_json(destination/"result.json")["experiment_id"] == contract["experiment_id"], "another experiment owns the paper exports")
    for path in (run/"paper").iterdir():
        if path.is_file(): shutil.copy2(path, destination/path.name)
    (destination/".gitattributes").write_text("* -text\n")
    write_json(destination/"evidence_index.json", {"run_dir": str(run), "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256(run/"contract.json"), "archive_sha256": sha256(run/"artifact_manifest.json"),
        "borrowed_source_run": contract["borrowed_source_run"], "borrowed_inputs_must_be_retained": True})
    write_json(run/"status.json", {"stage": "completed", "status": "completed", "at": base.now(), "result": result})
    base.append_event(run, {"stage": "report", "status": "completed", "result": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "train", "evaluate", "report", "run"), nargs="?", default="prepare")
    parser.add_argument("--config", type=Path, default=HNG/"configs/fb_violation_only_matched.json")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    require(Path(sys.executable).resolve() == base.PYTHON.resolve(), "use guardenv Python")
    design = read_json(args.config)
    run = (args.run_dir or Path(design["default_run_dir"])).resolve()
    with base.locked(run):
        try:
            c = prepare(run, args.config) if args.action in ("prepare", "run") else base.verify(run)
            print(json.dumps({"action": args.action, "run_dir": str(run), "experiment_id": c["experiment_id"]}), flush=True)
            if args.action in ("check", "run"):
                base.cli_check(run, c)
                validate_mode(run)
            if args.action in ("train", "run"): train(run, c)
            if args.action in ("evaluate", "run"): evaluate(run, c)
            if args.action in ("report", "run"): print(json.dumps(report(run, c), ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            write_json(run/"status.json", {"stage": args.action, "status": "failed", "at": base.now(), "error": repr(error)})
            base.append_event(run, {"stage": args.action, "status": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
