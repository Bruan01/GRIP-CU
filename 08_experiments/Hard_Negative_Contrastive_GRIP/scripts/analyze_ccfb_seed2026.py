#!/usr/bin/env python3
"""Pair the archived seed2026 test predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from hard_negative_grip.decode_io import eval_samples
from hard_negative_grip.official_lists import listed_relations_from_sample
from hard_negative_grip.paired_stats import mcnemar_exact, normalize_answer
from run_ccfb_sprint import HNG, hpo

RUNS = HNG / "results/runs"
DATA = HNG / "data"
NELL = RUNS / "paper_v2_nell23k"
NELL_HPO = NELL / "hpo_budget60_seed2026"
FB = RUNS / "fb15k237_three_configs_seed2026"
WN = RUNS / "wn18rr_three_configs_seed2026"
CODEX = RUNS / "codexm_three_configs_seed2026"
NELL_B1 = RUNS / "ccfb/nell23k/seed2026"


def read_predictions(directory, count):
    path = directory / "predictions_correct.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {row["question_id"]: row for row in rows}
    if len(rows) != count or len(indexed) != count or any(row["split"] != "test" for row in rows):
        raise ValueError(f"Incomplete or non-unique test predictions: {path}")
    return path, indexed


def timing(adapter):
    path = adapter / "run_metadata.json"
    if path.is_file():
        data = hpo.read_json(path)
        return {k: data[k] for k in ("seconds", "trainer_global_step", "trainer_epoch") if k in data}
    return {}


def analyze(dataset, primary, evaluation, left, right, expected_n, expected_scores, adapters,
            methods=("b1", "soft_mix_1")):
    left_path, left_rows = read_predictions(left, expected_n)
    right_path, right_rows = read_predictions(right, expected_n)
    samples = eval_samples(hpo.read_json(evaluation), ("test",))
    by_id = {sample["question_id"]: sample for sample in samples}
    if set(by_id) != set(left_rows) or set(left_rows) != set(right_rows) or len(samples) != expected_n:
        raise ValueError(f"{dataset}: test IDs differ from evaluation inputs")
    excluded = "nell23k:test:4943" if dataset == "nell23k" else None
    counts = {method: Counter() for method in methods}
    transitions = Counter()
    confusions = {method: Counter() for method in methods}
    changes_by_gold = Counter()
    for qid, sample in by_id.items():
        a, b = left_rows[qid], right_rows[qid]
        if a["target"] != b["target"] or a["question"] != b["question"]:
            raise ValueError(f"Mismatch in paired prompt or target: {qid}")
        gold = sample["answer"]
        if normalize_answer(gold) not in {normalize_answer(x) for x in a["target"]}:
            raise ValueError(f"Mismatch in source gold: {qid}")
        if qid == excluded:
            continue
        candidates = {normalize_answer(x) for x in (*listed_relations_from_sample(sample), gold)}
        for method, row in zip(methods, (a, b), strict=True):
            predicted = row["official_response"]
            normalized = normalize_answer(predicted or "")
            kind = ("correct" if row["official_correct"] else
                    "wrong_in_list" if normalized in candidates else "wrong_out_of_list")
            counts[method][kind] += 1
            if kind == "wrong_in_list":
                confusions[method][(gold, normalized)] += 1
        both = bool(a["official_correct"]), bool(b["official_correct"])
        transitions[both] += 1
        if both[0] != both[1]:
            changes_by_gold[gold] += 1 if both[1] else -1

    n = expected_n - (excluded is not None)
    scores = {method: values["correct"] / n for method, values in counts.items()}
    if any(abs(scores[method] - expected) > 1e-10
           for method, expected in zip(counts, expected_scores, strict=True)):
        raise ValueError(f"{dataset}: recomputed EM does not match recorded test EM: {scores}")
    gains, losses = transitions[False, True], transitions[True, False]
    return {
        "n": n, "excluded_question_id": excluded, "test_em": scores,
        "errors": {method: {k: v[k] for k in ("correct", "wrong_in_list", "wrong_out_of_list")}
                   for method, v in counts.items()},
        "gains": gains, "regressions": losses, "net": gains - losses,
        "both_correct": transitions[True, True], "both_wrong": transitions[False, False],
        "mcnemar": mcnemar_exact(gains, losses),
        "common_in_list_confusions": {
            method: [{"gold": a, "predicted_normalized": b, "count": count}
                     for (a, b), count in values.most_common(5)]
            for method, values in confusions.items()
        },
        "largest_net_gains_by_gold": changes_by_gold.most_common(5),
        "largest_net_losses_by_gold": [(key, value) for key, value in
                                       sorted(changes_by_gold.items(), key=lambda x: x[1]) if value < 0][:5],
        "s2_training": {key: timing(value) for key, value in zip(counts, adapters, strict=True)},
        "shared_s1_training": timing(primary / "s1_adapter"),
        "inputs": {str(path.relative_to(HNG)): hpo.sha256(path)
                   for path in (left_path, right_path, evaluation)},
    }


def result_rows(primary):
    return {row["arm"]: row for row in hpo.read_json(primary / "comparison.json")["rows"]}


def top9_report():
    runs = RUNS / "ccfb_top9_seed2026"
    if hpo.read_json(runs / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Both Top-9 controls must be complete.")
    comparisons = {}
    for dataset, primary, eval_name, soft, adapter, n in (
        ("nell23k", NELL, "recurrent_relation_prediction_full.aligned.json",
         NELL_HPO / "selected_test", NELL_HPO / "trial_001/listed/adapter", 4944),
        ("fb15k237", FB, "recurrent_relation_prediction_full.json",
         FB / "test/soft_mix_1", FB / "soft_mix_1/listed/adapter", 10000),
    ):
        root = runs / dataset
        metric = "official_harness_em" if dataset == "nell23k" else "official_em"
        comparisons[dataset] = analyze(
            dataset, primary, DATA / dataset / eval_name, soft, root / "test", n,
            (hpo.read_json(soft / "summary.json")["test"][metric],
             hpo.read_json(root / "test_result.json")[metric]),
            (adapter, root / "listed/adapter"), methods=("soft_mix_1", "top_k_hard"),
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top9", action="store_true", help="Compare Soft-Mix with Top-9 Hard.")
    if parser.parse_args().top9:
        datasets = top9_report()
        output = HNG / "results/archive/ccfb_seed2026_top9_paired_errors.json"
        hpo.freeze_text(output, json.dumps({
            "seed": 2026, "left": "soft_mix_1", "right": "top_k_hard",
            "error_definition": "Wrong strict official answer within versus outside the normalized candidate list.",
            "datasets": datasets,
        }, ensure_ascii=False, indent=2) + "\n")
        for dataset, row in datasets.items():
            print(dataset, row["gains"], row["regressions"], row["errors"])
        return
    configs = [
        ("nell23k", NELL, DATA / "nell23k/recurrent_relation_prediction_full.aligned.json",
         NELL_B1 / "test/b1", NELL_HPO / "selected_test", 4944,
         (result_rows(NELL_B1)["b1"]["official_harness_em"],
          hpo.read_json(NELL_HPO / "selected_test/summary.json")["test"]["official_harness_em"]),
         (NELL_B1 / "b1/b1/adapter", NELL_HPO / "trial_001/listed/adapter")),
    ]
    for dataset, primary, b1_name, count in (
        ("fb15k237", FB, "b1_seed2026", 10000),
        ("wn18rr", WN, "b1_seed2026", 3134),
        ("codexm", CODEX, "b1", 10311),
    ):
        rows = result_rows(primary)
        configs.append((
            dataset, primary, DATA / dataset / "recurrent_relation_prediction_full.json",
            primary / "test" / b1_name, primary / "test/soft_mix_1", count,
            (rows[b1_name]["test_official_em"], rows["soft_mix_1"]["test_official_em"]),
            (primary / b1_name / "b1/adapter", primary / "soft_mix_1/listed/adapter"),
        ))
    datasets = {config[0]: analyze(*config) for config in configs}
    report = {
        "seed": 2026, "method": "soft_mix_1",
        "scope": "Descriptive comparison of archived fixed adapter pairs. The selected seed results do not estimate training variability.",
        "error_definition": "Wrong strict official answer within versus outside the normalized candidate list.",
        "datasets": datasets,
    }
    output = HNG / "results/archive/ccfb_seed2026_paired_errors.json"
    hpo.freeze_text(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for dataset, row in datasets.items():
        print(dataset, row["n"], row["gains"], row["regressions"], row["errors"], row["s2_training"])
    print(output)


if __name__ == "__main__":
    main()
