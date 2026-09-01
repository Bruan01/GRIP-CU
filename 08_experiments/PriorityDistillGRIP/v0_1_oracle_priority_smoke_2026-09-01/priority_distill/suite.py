"""Aggregate seeds and apply the registered oracle-priority gate."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .io_utils import write_csv, write_json
from .supervision import METHODS


def discover_runs(run_root: Path) -> list[dict]:
    runs = []
    for path in sorted(run_root.glob("*/seed_*/run_summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def aggregate_runs(runs: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    aggregate: dict[str, dict] = {}
    for method, method_runs in grouped.items():
        depth_values = {
            str(depth): [run["metrics"]["test"]["accuracy_by_depth"][str(depth)] for run in method_runs]
            for depth in range(1, 5)
        }
        aggregate[method] = {
            "seed_count": len(method_runs),
            "seeds": sorted(run["seed"] for run in method_runs),
            "test_accuracy": sum(run["metrics"]["test"]["accuracy"] for run in method_runs) / len(method_runs),
            "test_macro_depth_accuracy": sum(run["metrics"]["test"]["macro_depth_accuracy"] for run in method_runs) / len(method_runs),
            "test_worst_depth_accuracy": sum(run["metrics"]["test"]["worst_depth_accuracy"] for run in method_runs) / len(method_runs),
            "test_deep_3_4_accuracy": sum(run["metrics"]["test"]["deep_3_4_accuracy"] for run in method_runs) / len(method_runs),
            "test_accuracy_by_depth": {depth: sum(values) / len(values) for depth, values in depth_values.items()},
            "stage1_input_tokens": sum(run["training"]["stage1"]["input_tokens"] for run in method_runs) / len(method_runs),
            "stage1_optimizer_steps": sum(run["training"]["stage1"]["optimizer_steps"] for run in method_runs) / len(method_runs),
            "stage1_truncated_example_rate": sum(
                run["training"]["stage1"]["truncated_examples"] / max(run["training"]["stage1"]["examples"], 1)
                for run in method_runs
            ) / len(method_runs),
        }
    return aggregate


def apply_gate(aggregate: dict[str, dict], config: dict) -> dict:
    missing = set(METHODS) - set(aggregate)
    if missing:
        return {"status": "INCOMPLETE_SUITE", "missing_methods": sorted(missing), "checks": {}}
    gate = config["gate"]
    oracle = aggregate["oracle_priority_equal_token"]
    answer = aggregate["answer_only"]
    more_qa = aggregate["more_qa_equal_token"]
    random_path = aggregate["random_path_equal_token"]
    all_paths = aggregate["all_paths_equal_token"]
    stage1_methods = ("more_qa_equal_token", "random_path_equal_token", "all_paths_equal_token", "oracle_priority_equal_token")
    stage1_tokens = [float(aggregate[name]["stage1_input_tokens"]) for name in stage1_methods]
    token_reference = max(stage1_tokens)
    token_relative_gap = (max(stage1_tokens) - min(stage1_tokens)) / token_reference if token_reference else 1.0
    checks = {
        "oracle_over_answer_only": {"value": oracle["test_accuracy"] - answer["test_accuracy"], "threshold": float(gate["oracle_over_answer_only"]), "comparison": "greater_equal"},
        "oracle_over_more_qa": {"value": oracle["test_accuracy"] - more_qa["test_accuracy"], "threshold": float(gate["oracle_over_more_qa"]), "comparison": "greater_equal"},
        "oracle_over_random_path": {"value": oracle["test_accuracy"] - random_path["test_accuracy"], "threshold": float(gate["oracle_over_random_path"]), "comparison": "greater"},
        "oracle_over_all_paths": {"value": oracle["test_accuracy"] - all_paths["test_accuracy"], "threshold": float(gate["oracle_over_all_paths"]), "comparison": "greater"},
        "deep_3_4_improvement": {"value": oracle["test_deep_3_4_accuracy"] - answer["test_deep_3_4_accuracy"], "threshold": float(gate["deep_3_4_improvement"]), "comparison": "greater"},
        "stage1_token_budget_match": {"value": token_relative_gap, "threshold": float(gate.get("max_stage1_token_relative_gap", 0.01)), "comparison": "less_equal"},
        "stage1_no_prompt_truncation": {"value": max(float(aggregate[name].get("stage1_truncated_example_rate", 0.0)) for name in stage1_methods), "threshold": float(gate.get("max_stage1_truncated_example_rate", 0.0)), "comparison": "less_equal"},
    }
    for check in checks.values():
        if check["comparison"] == "less_equal":
            check["pass"] = check["value"] <= check["threshold"]
        elif check["comparison"] == "greater":
            check["pass"] = check["value"] > check["threshold"]
        else:
            check["pass"] = check["value"] >= check["threshold"]
    minimum_seed_count = min(aggregate[method]["seed_count"] for method in METHODS)
    enough_seeds = minimum_seed_count >= int(gate["minimum_seeds_for_final"])
    all_pass = all(check["pass"] for check in checks.values())
    if enough_seeds:
        status = "GO_LEARNED_PRIORITIZER" if all_pass else "STOP_PRIORITY_DISTILL"
    else:
        status = "PRELIMINARY_GO" if all_pass else "PRELIMINARY_STOP"
    return {"status": status, "all_checks_pass": all_pass, "minimum_seed_count": minimum_seed_count, "required_seed_count": int(gate["minimum_seeds_for_final"]), "checks": checks}


def render_report(aggregate: dict[str, dict], gate: dict) -> str:
    lines = [
        "# PriorityDistill-GRIP v0.1 Suite Report",
        "",
        f"Decision: **{gate['status']}**",
        "",
        "| Method | Seeds | Accuracy | Deep 3/4 | d1 | d2 | d3 | d4 | Stage-1 tokens | Stage-1 steps | Truncated examples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if method not in aggregate:
            continue
        values = aggregate[method]
        depths = values["test_accuracy_by_depth"]
        lines.append(
            f"| {method} | {values['seed_count']} | {values['test_accuracy']:.4f} | {values['test_deep_3_4_accuracy']:.4f} | "
            f"{depths['1']:.4f} | {depths['2']:.4f} | {depths['3']:.4f} | {depths['4']:.4f} | "
            f"{values['stage1_input_tokens']:.0f} | {values['stage1_optimizer_steps']:.1f} | "
            f"{values.get('stage1_truncated_example_rate', 0.0):.2%} |"
        )
    lines.extend(["", "## Registered gate", ""])
    for name, check in gate.get("checks", {}).items():
        lines.append(f"- `{name}`: value `{check['value']:.6f}`, comparison `{check['comparison']}`, threshold `{check['threshold']:.6f}`, pass `{check['pass']}`")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Only `GO_LEARNED_PRIORITIZER` authorizes v0.2 learned path scoring. A one-seed result is preliminary. "
        "If token budgets diverge materially, the suite remains a systems result rather than a mechanism claim.",
        "",
    ])
    return "\n".join(lines)


def write_suite_outputs(run_root: Path, config: dict) -> dict:
    runs = discover_runs(run_root)
    if not runs:
        raise FileNotFoundError(f"no run_summary.json under {run_root}")
    aggregate = aggregate_runs(runs)
    gate = apply_gate(aggregate, config)
    payload = {"run_count": len(runs), "aggregate": aggregate, "gate": gate}
    write_json(run_root / "suite_summary.json", payload)
    rows = [{"method": method, **values} for method, values in aggregate.items()]
    write_csv(run_root / "suite_metrics.csv", rows)
    (run_root / "REPORT.md").write_text(render_report(aggregate, gate), encoding="utf-8")
    return payload
