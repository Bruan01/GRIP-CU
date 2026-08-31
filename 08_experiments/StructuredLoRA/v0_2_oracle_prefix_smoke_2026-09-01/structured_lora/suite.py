"""Aggregate method/seed runs and apply the pre-registered go/no-go gate."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .io_utils import write_csv, write_json
from .metrics import mean


def discover_runs(run_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for path in sorted(run_root.glob("**/run_summary.json")):
        import json

        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        summary["_path"] = str(path)
        summaries.append(summary)
    return summaries


def aggregate_runs(runs: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    aggregate: dict[str, dict] = {}
    for method, method_runs in sorted(grouped.items()):
        test_metrics = [run["metrics"]["test"] for run in method_runs]
        validation_metrics = [run["metrics"]["validation"] for run in method_runs]
        aggregate[method] = {
            "seeds": sorted(run["seed"] for run in method_runs),
            "seed_count": len(method_runs),
            "test_accuracy": mean(metric["accuracy"] for metric in test_metrics),
            "test_macro_depth_accuracy": mean(metric["macro_depth_accuracy"] for metric in test_metrics),
            "test_worst_depth_accuracy": mean(metric["worst_depth_accuracy"] for metric in test_metrics),
            "validation_accuracy": mean(metric["accuracy"] for metric in validation_metrics),
            "test_accuracy_by_depth": {
                str(depth): mean(metric["accuracy_by_depth"][str(depth)] for metric in test_metrics)
                for depth in range(1, 5)
            },
            "trainable_parameters": sorted({run["injection"]["trainable_parameters"] for run in method_runs}),
            "mean_training_seconds": mean(run["training"]["elapsed_seconds"] for run in method_runs),
            "mean_peak_gpu_memory_bytes": mean(run["training"]["peak_gpu_memory_bytes"] for run in method_runs),
        }
    return aggregate


def apply_gate(aggregate: dict[str, dict], config: dict) -> dict:
    required = {
        "monolithic",
        "flat_oracle",
        "ordered_prefix",
        "permuted_depth_prefix",
        "non_nested_random_masks",
    }
    missing = sorted(required - set(aggregate))
    if missing:
        return {"status": "INCOMPLETE", "missing_methods": missing, "checks": {}}
    ordered = aggregate["ordered_prefix"]
    monolithic = aggregate["monolithic"]
    flat = aggregate["flat_oracle"]
    permuted = aggregate["permuted_depth_prefix"]
    non_nested = aggregate["non_nested_random_masks"]
    gate = config["gate"]

    deep_delta = mean(
        ordered["test_accuracy_by_depth"][str(depth)]
        - monolithic["test_accuracy_by_depth"][str(depth)]
        for depth in (3, 4)
    )
    one_hop_delta = (
        ordered["test_accuracy_by_depth"]["1"] - monolithic["test_accuracy_by_depth"]["1"]
    )
    checks = {
        "ordered_over_monolithic": {
            "value": ordered["test_accuracy"] - monolithic["test_accuracy"],
            "threshold": float(gate["ordered_over_monolithic"]),
        },
        "ordered_over_flat": {
            "value": ordered["test_accuracy"] - flat["test_accuracy"],
            "threshold": float(gate["ordered_over_flat"]),
        },
        "deep_3_4_improvement": {
            "value": deep_delta,
            "threshold": float(gate["deep_3_4_improvement"]),
        },
        "one_hop_not_degraded": {
            "value": one_hop_delta,
            "threshold": -float(gate["maximum_1hop_drop"]),
        },
        "ordered_over_permuted": {
            "value": ordered["test_accuracy"] - permuted["test_accuracy"],
            "threshold": float(gate["ordered_over_permuted"]),
        },
        "ordered_over_non_nested": {
            "value": ordered["test_accuracy"] - non_nested["test_accuracy"],
            "threshold": float(gate["ordered_over_non_nested"]),
        },
    }
    for check in checks.values():
        check["pass"] = check["value"] > check["threshold"] if check["threshold"] == 0 else check["value"] >= check["threshold"]

    seed_counts = [aggregate[method]["seed_count"] for method in required]
    enough_seeds = min(seed_counts) >= int(gate["minimum_seeds_for_final"])
    all_pass = all(check["pass"] for check in checks.values())
    if enough_seeds:
        status = "GO_LEARNED_ROUTER" if all_pass else "STOP_STRUCTURED_LORA"
    else:
        status = "PRELIMINARY_GO" if all_pass else "PRELIMINARY_STOP"
    return {
        "status": status,
        "all_checks_pass": all_pass,
        "minimum_seed_count": min(seed_counts),
        "required_seed_count": int(gate["minimum_seeds_for_final"]),
        "checks": checks,
        "random_group_order_note": (
            "A fixed permutation of group identities preserves the nested prefix function class. "
            "It is a symmetry control, not a valid expected-degradation control; the non-nested mask is decisive."
        ),
    }


def render_report(aggregate: dict[str, dict], gate: dict) -> str:
    lines = [
        "# StructuredLoRA v0.2 Suite Report",
        "",
        f"Decision: **{gate['status']}**",
        "",
        "## Test metrics",
        "",
        "| Method | Seeds | Accuracy | Macro-hop | Worst-hop | d1 | d2 | d3 | d4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in sorted(aggregate.items()):
        depths = values["test_accuracy_by_depth"]
        lines.append(
            f"| {method} | {values['seed_count']} | {values['test_accuracy']:.4f} | "
            f"{values['test_macro_depth_accuracy']:.4f} | {values['test_worst_depth_accuracy']:.4f} | "
            f"{depths['1']:.4f} | {depths['2']:.4f} | {depths['3']:.4f} | {depths['4']:.4f} |"
        )
    lines.extend(["", "## Registered gate", ""])
    for name, check in gate.get("checks", {}).items():
        lines.append(
            f"- `{name}`: value `{check['value']:.6f}`, threshold `{check['threshold']:.6f}`, "
            f"pass `{check['pass']}`"
        )
    if gate.get("random_group_order_note"):
        lines.extend(["", "## Control interpretation", "", gate["random_group_order_note"]])
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "Only `GO_LEARNED_ROUTER` permits implementation of a learned ordinal router. "
            "`PRELIMINARY_GO` requires the remaining registered seeds first. Any stop decision is recorded without adding rank.",
            "",
        ]
    )
    return "\n".join(lines)


def write_suite_outputs(run_root: Path, config: dict) -> dict:
    runs = discover_runs(run_root)
    if not runs:
        raise FileNotFoundError(f"no run_summary.json under {run_root}")
    aggregate = aggregate_runs(runs)
    gate = apply_gate(aggregate, config)
    rows = []
    for method, values in sorted(aggregate.items()):
        row = {
            "method": method,
            "seed_count": values["seed_count"],
            "seeds": ";".join(str(seed) for seed in values["seeds"]),
            "test_accuracy": values["test_accuracy"],
            "test_macro_depth_accuracy": values["test_macro_depth_accuracy"],
            "test_worst_depth_accuracy": values["test_worst_depth_accuracy"],
        }
        row.update({f"test_d{depth}": values["test_accuracy_by_depth"][str(depth)] for depth in range(1, 5)})
        rows.append(row)
    payload = {"run_count": len(runs), "aggregate": aggregate, "gate": gate}
    write_json(run_root / "suite_summary.json", payload)
    write_csv(run_root / "suite_metrics.csv", rows)
    (run_root / "REPORT.md").write_text(render_report(aggregate, gate), encoding="utf-8")
    return payload
