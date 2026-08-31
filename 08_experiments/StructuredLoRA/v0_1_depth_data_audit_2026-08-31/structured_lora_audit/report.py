"""Decision summary and Markdown report rendering for the audit."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def evaluate_gates(
    distribution: Iterable[dict],
    prediction_metrics: Iterable[dict],
    *,
    exact_hop_total: int,
    expected_exact_hop_total: int,
    relation_depth_nmi: float,
) -> dict:
    distribution = list(distribution)
    prediction_metrics = list(prediction_metrics)
    test_buckets = {
        row["depth_bucket"]: row["count"]
        for row in distribution
        if row["split"] == "test" and row["mode"] == "undirected"
    }
    supported_buckets = [bucket for bucket in ("2", "3", "4") if test_buckets.get(bucket, 0) >= 100]

    accuracy_by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in prediction_metrics:
        if row["n"] >= 8:
            accuracy_by_condition[(row["split"], row["condition"])].append(float(row["accuracy"]))
    spreads = [max(values) - min(values) for values in accuracy_by_condition.values() if len(values) >= 2]
    maximum_accuracy_spread = max(spreads, default=None)

    gates = {
        "exact_hop_generation_complete": exact_hop_total == expected_exact_hop_total,
        "at_least_two_nontrivial_test_buckets": len(supported_buckets) >= 2,
        "depth_not_relation_identity": relation_depth_nmi < 0.80,
        "existing_baseline_has_depth_variation": (
            maximum_accuracy_spread is not None and maximum_accuracy_spread >= 0.05
        ),
    }
    required = [
        gates["exact_hop_generation_complete"],
        gates["at_least_two_nontrivial_test_buckets"],
        gates["depth_not_relation_identity"],
    ]
    decision = "GO_ORACLE_PREFIX" if all(required) else "REVISE_DEPTH_AXIS"
    return {
        "decision": decision,
        "gates": gates,
        "supported_nontrivial_test_buckets": supported_buckets,
        "maximum_observed_baseline_accuracy_spread": maximum_accuracy_spread,
        "relation_depth_nmi_test": relation_depth_nmi,
        "interpretation": (
            "Proceed only to an oracle-prefix, equal-rank pilot. Learned routing remains blocked until the "
            "oracle structure beats monolithic LoRA."
            if decision == "GO_ORACLE_PREFIX"
            else "The current depth labels do not yet justify StructuredLoRA; revise the structural axis before training."
        ),
    }


def render_markdown(summary: dict) -> str:
    dataset = summary["dataset"]
    decision = summary["decision"]
    lines = [
        "# StructuredLoRA v0.1 Depth Data Audit Report",
        "",
        f"- Dataset: `{dataset['name']}`",
        f"- Train/valid/test: `{dataset['split_sizes']['train']}/{dataset['split_sizes']['valid']}/{dataset['split_sizes']['test']}`",
        f"- Train graph nodes/edges/relations: `{dataset['train_nodes']}/{dataset['train_edges']}/{dataset['train_relations']}`",
        f"- Decision: **{decision['decision']}**",
        "",
        "## What was measured",
        "",
        "1. Train triples use leave-one-edge-out support depth.",
        "2. Validation and test triples use shortest support paths in the train graph.",
        "3. Directed and undirected depths are both exported; undirected is the primary NELL23K structural proxy.",
        "4. Exact-hop tasks require a simple directed path, no shorter directed path, and a unique answer for the relation chain.",
        "5. Existing diagnostic predictions are joined by split, endpoint pair, and target relation; their previous `true_hop` field is not trusted.",
        "",
        "## Decision gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in decision["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Key measurements",
            "",
            f"- Test nontrivial support buckets with at least 100 examples: `{', '.join(decision['supported_nontrivial_test_buckets']) or 'none'}`",
            f"- Test relation-depth normalized mutual information: `{decision['relation_depth_nmi_test']:.6f}`",
            f"- Largest observed baseline accuracy spread across usable depth buckets: `{decision['maximum_observed_baseline_accuracy_spread']}`",
            f"- Exact-hop tasks: `{summary['exact_hop']['total_tasks']}`",
            f"- Prediction join rate: `{summary['predictions']['join_rate']}`",
            "",
            "## Interpretation",
            "",
            decision["interpretation"],
            "",
            "The NELL23K label is a **support-depth proxy**, not a controlled query-hop label. It can support the performance table and stratified analysis, but exact-hop data is still required for causal depth-specialization claims.",
            "",
            "The imported diagnostic run contains only 32 validation and 64 test questions per condition. Its depth-wise accuracy variation is a weak feasibility signal, not paper-level statistical evidence.",
            "",
            "A bounded `>4_or_unreachable` bucket means no path was found within four steps; it must not be described as globally unreachable.",
            "",
            "## Next experiment",
            "",
            "Run `v0_2_oracle_prefix_smoke` with identical data, token budget, target modules, and total LoRA rank for every baseline:",
            "",
            "1. equal-rank monolithic LoRA;",
            "2. static split with all groups active;",
            "3. flat routed experts;",
            "4. ordered oracle prefix;",
            "5. ordered oracle prefix plus depth-local credit assignment;",
            "6. permuted depth labels and random group order controls.",
            "",
            "Do not implement the learned router unless oracle prefix beats equal-rank monolithic LoRA.",
            "",
        ]
    )
    return "\n".join(lines)
