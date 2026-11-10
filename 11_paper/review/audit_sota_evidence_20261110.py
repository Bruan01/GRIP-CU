"""CPU-only reconciliation of the supplied PDF's local experiment evidence.

Run with guardenv Python from any directory. This checks exported per-question
data against result JSON; it does not rerun inference or authenticate weights.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "11_paper/artifacts"
SOURCES = {}


def source(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def rows(path, expected_n):
    with source(path).open(newline="") as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == expected_n, path
    if "question_id" in data[0]:
        assert len({r["question_id"] for r in data}) == expected_n, path
    return data


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), (actual, expected)


def holm(values):
    ordered = sorted(values, key=values.get)
    result, previous = {}, 0.0
    for i, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(values) - i) * values[name]))
        result[name] = previous
    return result


def paired(left, right):
    b = sum(a and not c for a, c in zip(left, right))
    c = sum(not a and d for a, d in zip(left, right))
    return {
        "rescue": b, "regression": c,
        "delta_pp": 100 * (b - c) / len(left),
        "p_raw_two_sided_exact": float(binomtest(b, b + c).pvalue) if b + c else 1.0,
    }


def reconcile_sources(folder, expected_n):
    result = read_json(folder / "result.json")
    data = rows(folder / "paired_all_methods.csv", expected_n)
    metrics, flags = {}, {}
    for item in result["metrics"]:
        method = item["method"]
        ranks = [int(r[f"{method}_gold_rank"]) for r in data]
        assert all(1 <= rank <= item["candidate_count"] for rank in ranks)
        direct = [int(r[f"{method}_direct_correct"]) for r in data]
        assert set(direct) <= {0, 1}
        observed = {
            "direct_em_percent": 100 * sum(direct) / len(data),
            "h1_percent": 100 * sum(r == 1 for r in ranks) / len(data),
            "mrr_percent": 100 * sum(1 / r for r in ranks) / len(data),
            "h3_percent": 100 * sum(r <= 3 for r in ranks) / len(data),
        }
        for key, value in observed.items():
            close(value, item[key])
        metrics[method] = observed
        flags[method] = {"direct": direct, "ranking": [r == 1 for r in ranks]}
    comparisons = {}
    for name, expected in result["comparisons"].items():
        left, right = name.split("_minus_")
        comparisons[name] = {}
        for protocol, result_key in (("direct", "direct_left_minus_right"), ("ranking", "ranking")):
            observed = paired(flags[left][protocol], flags[right][protocol])
            for key in ("rescue", "regression"):
                assert observed[key] == expected[result_key]["paired_counts"][key]
            comparisons[name][protocol] = observed
    # Explicitly exploratory: these families reproduce the NELL table's
    # within-dataset six-comparison convention, not a preregistration claim.
    family = {f"{name}:{p}": value["p_raw_two_sided_exact"]
              for name, pair in comparisons.items() for p, value in pair.items()}
    return {"n": len(data), "metrics": metrics, "comparisons": comparisons,
            "exploratory_holm_family_six_within_dataset": holm(family)}


def main():
    output = {"scope": "Export-level CPU audit; no GPU rerun, weight rehash, or causal validation."}
    source(ROOT / "11_paper/SC-CL_sota_paper.pdf")
    output["nell"] = reconcile_sources(ART / "nell23k_source_controls_seed2026", 4943)
    output["wn"] = reconcile_sources(ART / "wn18rr_source_controls_seed2026", 3134)

    direct_files = {"nell": "nell23k_single_seed_test_20261102.json",
                    "codexm": "codexm_single_seed_test_20261103.json",
                    "fb": "fb15k237_2_single_seed_test_20261104.json"}
    direct_p = {}
    for name, filename in direct_files.items():
        item = read_json(ART / filename)
        counts = item["paired_counts"]
        assert counts["both_correct"] + counts["rescue"] == item["selected_correct"]
        assert counts["both_correct"] + counts["regression"] == item["baseline_correct"]
        assert sum(counts.values()) == item["test_count"]
        direct_p[name] = float(binomtest(counts["rescue"], counts["rescue"] + counts["regression"]).pvalue)
    output["table2_p_raw_from_recorded_counts"] = direct_p
    output["table2_p_holm_m3"] = holm(direct_p)

    data = rows(ART / "fb_frequency_slices_seed2026/question_slices.csv", 10000)
    per_relation = rows(ART / "fb_frequency_slices_seed2026/per_relation.csv", 215)
    assert len({r["relation"] for r in per_relation}) == 215
    for rel in per_relation:
        question_rows = [r for r in data if r["relation"] == rel["relation"]]
        assert len(question_rows) == int(rel["n"])
        assert all(r["frequency_group"] == rel["frequency_group"] for r in question_rows)
        for method in ("random9", "profile_only", "violation_only", "frequency_matched", "full"):
            assert sum(int(r[f"{method}_correct"]) for r in question_rows) == int(rel[f"{method}_correct"])
    expected = read_json(ART / "fb_frequency_slices_seed2026/result.json")
    output["fb_slices"] = []
    for item in expected["metrics"]:
        group, method = item["frequency_group"], item["method"]
        selected = [r for r in data if group == "all" or r["frequency_group"] == group]
        rels = [r for r in per_relation if group == "all" or r["frequency_group"] == group]
        count = sum(int(r[f"{method}_correct"]) for r in selected)
        macro = 100 * sum(int(r[f"{method}_correct"]) / int(r["n"]) for r in rels) / len(rels)
        assert count == item["correct"] and len(selected) == item["n"]
        close(100 * count / len(selected), item["em_percent"])
        close(macro, item["macro_relation_em_percent"])
        output["fb_slices"].append(item)
    output["fb_frequency_full_minus_quota"] = {
        group: paired([int(r["full_correct"]) for r in data if group == "all" or r["frequency_group"] == group],
                      [int(r["frequency_matched_correct"]) for r in data if group == "all" or r["frequency_group"] == group])
        for group in ("low", "middle", "high", "all")}
    pairs = (("full", "random9"), ("full", "profile_only"), ("full", "violation_only"),
             ("frequency_matched", "random9"), ("full", "frequency_matched"))
    output["table3_pairs"] = {
        f"{a}_minus_{b}": paired([int(r[f"{a}_correct"]) for r in data],
                                [int(r[f"{b}_correct"]) for r in data]) for a, b in pairs}
    output["table3_holm_m5"] = holm({k: v["p_raw_two_sided_exact"] for k, v in output["table3_pairs"].items()})

    rank_data = rows(ART / "fb_matched_rerank_seed2026/paired_ranks_and_direct.csv", 10000)
    rank_result = read_json(ART / "fb_matched_rerank_seed2026/result.json")
    for item in rank_result["metrics"]:
        method = item["method"]
        ranks = [int(r[f"{method}_gold_rank"]) for r in rank_data]
        for key, value in (("h1_percent", 100 * sum(r == 1 for r in ranks) / len(ranks)),
                           ("mrr_percent", 100 * sum(1 / r for r in ranks) / len(ranks)),
                           ("h3_percent", 100 * sum(r <= 3 for r in ranks) / len(ranks))):
            close(value, item[key])
    output["fb_ranking_metrics_verified"] = rank_result["metrics"]

    k_data = rows(ART / "fb_k_sensitivity_seed2026/paired_three_k.csv", 10000)
    k_result = read_json(ART / "fb_k_sensitivity_seed2026/result.json")
    for item in k_result["metrics"]:
        ranks = [int(r[f"k{item['k']}_gold_rank"]) for r in k_data]
        close(100 * sum(r == 1 for r in ranks) / len(ranks), item["h1_percent"])
        close(100 * sum(1 / r for r in ranks) / len(ranks), item["mrr_percent"])
        close(100 * sum(r <= 3 for r in ranks) / len(ranks), item["h3_percent"])
    output["k_metrics_verified"] = k_result["metrics"]
    output["k_pairs"] = {f"k{k}_minus_k9": paired([int(r[f"k{k}_gold_rank"]) == 1 for r in k_data],
                                                    [int(r["k9_gold_rank"]) == 1 for r in k_data]) for k in (3, 15)}
    output["arithmetic_checks"] = {
        "k3_process_training_time_saving_vs_k15_percent": 100 * (1 - 9944.0 / 14280.6),
        "historical_nell_rerank_seconds_per_question": 3567 / 4943,
        "historical_nell_scored_candidates_per_bank_question": 644094 / 3253,
        "historical_nell_scored_candidates_div_current_bank_questions": 644094 / 2974,
        "fb_micro_gain_recovery_ratio_percent": 100 * (89.83 - 86.48) / (89.90 - 86.48),
        "note": "Ratios describe this run; they are not causal mediation estimates or total-cost savings.",
    }
    output["input_sha256"] = SOURCES
    output["status"] = "pass"
    target = Path(__file__).with_name("sota_evidence_audit_20261110.json")
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "pass", "source_files": len(SOURCES), "output": str(target)}, ensure_ascii=False))
    for name in ("nell", "wn"):
        print(name, json.dumps(output[name]["comparisons"], ensure_ascii=False))
    print("k_pairs", json.dumps(output["k_pairs"]))
    print("table2", json.dumps(output["table2_p_holm_m3"]))


if __name__ == "__main__":
    main()
