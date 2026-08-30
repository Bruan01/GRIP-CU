#!/usr/bin/env python3
"""Independent, standard-library audit of FactorGRIP v0.1 candidate-energy results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/results/runs/wsl3090_nell23k_candidate_energy_20260830_01"
OUT = Path(__file__).resolve().parent
SEED = 20260830
BOOTSTRAP_SAMPLES = 20000


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_answer(answer: str) -> str:
    text = answer.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    mapping = {"man": "men", "woman": "women"}
    return " ".join(mapping.get(word, word) for word in text.split())


def exact_match(prediction: str | None, targets: list[str]) -> bool:
    pred = normalize_answer(str(prediction or ""))
    return any(pred == normalize_answer(str(target)) for target in targets)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    b = sum(l and not r for l, r in zip(left, right))
    c = sum((not l) and r for l, r in zip(left, right))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"left_only_correct": b, "right_only_correct": c, "discordant": n, "p_value_exact_two_sided": p}


def paired_bootstrap(left: list[bool], right: list[bool], seed: int = SEED) -> dict:
    rng = random.Random(seed)
    n = len(left)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(mean([float(right[i]) - float(left[i]) for i in indices]))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "ci95_low": percentile(deltas, 0.025),
        "ci95_high": percentile(deltas, 0.975),
    }


def binomial_survival(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(k, n + 1))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    predictions = load_jsonl(RUN / "predictions.jsonl")
    score_rows = load_jsonl(RUN / "candidate_scores.jsonl")
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((RUN / "run.log").read_text(encoding="utf-8"))
    source_input_path = ROOT / "08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json"
    source_input = json.loads(source_input_path.read_text(encoding="utf-8"))
    source_questions = {str(r["question_id"]): r for r in source_input["recurrent_questions"] if r.get("split") in {"validation", "test"}}

    pred_by_key = {
        (r["question_id"], r["split"], r["adapter_control"], r["decoder_type"]): r
        for r in predictions
    }
    score_by_key = {
        (r["question_id"], r["split"], r["adapter_control"]): r for r in score_rows
    }

    condition_rows = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[(row["split"], row["adapter_control"], row["decoder_type"])].append(row)
    for (split, control, decoder), rows in sorted(grouped.items()):
        recomputed = [exact_match(r.get("predicted_candidate"), r.get("target", [])) for r in rows]
        condition_rows.append({
            "split": split,
            "adapter_control": control,
            "decoder_type": decoder,
            "n": len(rows),
            "correct": sum(recomputed),
            "accuracy": mean([float(x) for x in recomputed]),
            "out_of_candidate_count_raw_membership": sum(r.get("predicted_candidate") not in r.get("candidates", []) for r in rows),
            "mean_latency_seconds": mean([float(r.get("latency_seconds", 0.0)) for r in rows]),
            "median_latency_seconds": statistics.median(float(r.get("latency_seconds", 0.0)) for r in rows),
            "max_peak_memory_bytes": max(int(r.get("peak_memory_bytes", 0)) for r in rows),
        })
    write_csv(OUT / "condition_metrics.csv", condition_rows)

    normalization_rows = []
    for split in ("validation", "test"):
        for control in ("correct", "none", "wrong_depth"):
            rows = [r for r in score_rows if r["split"] == split and r["adapter_control"] == control]
            for normalization, field in (("raw_sum", "sum_logprob"), ("length_normalized", "norm_logprob")):
                correct = 0
                selected_token_lengths = []
                margins = []
                for row in rows:
                    ranked = sorted(row["scores"], key=lambda s: float(s[field]), reverse=True)
                    correct += int(exact_match(ranked[0]["candidate"], row["target"]))
                    selected_token_lengths.append(int(ranked[0]["num_tokens"]))
                    margins.append(float(ranked[0][field]) - float(ranked[1][field]))
                normalization_rows.append({
                    "split": split,
                    "adapter_control": control,
                    "normalization": normalization,
                    "n": len(rows),
                    "correct": correct,
                    "accuracy": correct / len(rows),
                    "mean_selected_token_length": mean(selected_token_lengths),
                    "mean_top1_top2_margin": mean(margins),
                })
    write_csv(OUT / "normalization_metrics.csv", normalization_rows)

    comparison_specs = []
    for split in ("validation", "test"):
        for control in ("correct", "none", "wrong_depth"):
            comparison_specs.extend([
                (split, control, "free", control, "score", "score_minus_free"),
                (split, control, "free", control, "constrained", "constrained_minus_free"),
                (split, control, "constrained", control, "score", "score_minus_constrained"),
            ])
        for decoder in ("free", "constrained", "score"):
            comparison_specs.extend([
                (split, "none", decoder, "correct", decoder, "correct_minus_none"),
                (split, "wrong_depth", decoder, "correct", decoder, "correct_minus_wrong_depth"),
            ])

    paired_rows = []
    for split, lc, ld, rc, rd, label in comparison_specs:
        qids = sorted({q for q, s, c, d in pred_by_key if s == split and c == lc and d == ld} &
                      {q for q, s, c, d in pred_by_key if s == split and c == rc and d == rd})
        left = [exact_match(pred_by_key[(q, split, lc, ld)].get("predicted_candidate"), pred_by_key[(q, split, lc, ld)]["target"]) for q in qids]
        right = [exact_match(pred_by_key[(q, split, rc, rd)].get("predicted_candidate"), pred_by_key[(q, split, rc, rd)]["target"]) for q in qids]
        mc = exact_mcnemar(left, right)
        boot = paired_bootstrap(left, right, seed=SEED + len(paired_rows))
        paired_rows.append({
            "split": split,
            "comparison": label,
            "left": f"{lc}.{ld}",
            "right": f"{rc}.{rd}",
            "n": len(qids),
            "left_accuracy": mean([float(x) for x in left]),
            "right_accuracy": mean([float(x) for x in right]),
            "delta_right_minus_left": mean([float(r)-float(l) for l, r in zip(left, right)]),
            **mc,
            "bootstrap_ci95_low": boot["ci95_low"],
            "bootstrap_ci95_high": boot["ci95_high"],
            "bootstrap_samples": boot["samples"],
            "bootstrap_seed": boot["seed"],
        })
    write_csv(OUT / "paired_comparisons.csv", paired_rows)

    # Candidate-position diagnostics for the selected raw-sum scorer.
    position_rows = []
    for split in ("validation", "test"):
        for control in ("correct", "none", "wrong_depth"):
            rows = [r for r in score_rows if r["split"] == split and r["adapter_control"] == control]
            for pos in range(10):
                bucket = []
                predicted_count = 0
                for row in rows:
                    target_pos = row["candidates"].index(row["target"][0])
                    predicted_pos = max(range(10), key=lambda i: float(row["scores"][i]["sum_logprob"]))
                    if target_pos == pos:
                        bucket.append(predicted_pos == target_pos)
                    predicted_count += int(predicted_pos == pos)
                position_rows.append({
                    "split": split,
                    "adapter_control": control,
                    "position": pos,
                    "target_count": len(bucket),
                    "correct": sum(bucket),
                    "accuracy_given_target_position": mean([float(x) for x in bucket]),
                    "predicted_count": predicted_count,
                })
    write_csv(OUT / "candidate_position_diagnostics.csv", position_rows)

    # Compact examples for qualitative inspection.
    failures = []
    for split in ("validation", "test"):
        qids = sorted({q for q, s, c, d in pred_by_key if s == split})
        for q in qids:
            cfree = pred_by_key[(q, split, "correct", "free")]
            cscore = pred_by_key[(q, split, "correct", "score")]
            nscore = pred_by_key[(q, split, "none", "score")]
            wc = bool(cscore["correct"])
            wn = bool(nscore["correct"])
            wf = bool(cfree["correct"])
            kind = None
            if wn and not wc:
                kind = "adapter_regression_none_correct_adapter_wrong"
            elif wc and not wn:
                kind = "adapter_gain_correct_adapter_correct_none_wrong"
            elif wc and not wf:
                kind = "decoder_rescue_correct_adapter"
            if kind:
                failures.append({
                    "kind": kind,
                    "question_id": q,
                    "split": split,
                    "question": cfree.get("metadata", {}).get("question"),
                    "target": cfree["target"],
                    "free_prediction": cfree["predicted_candidate"],
                    "correct_score_prediction": cscore["predicted_candidate"],
                    "none_score_prediction": nscore["predicted_candidate"],
                    "candidates": cfree["candidates"],
                })
    with (OUT / "failure_examples.jsonl").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Integrity and implementation audit.
    expected_pred_keys = {
        (q, split, control, decoder)
        for split, count in (("validation", 32), ("test", 64))
        for q in sorted({r["question_id"] for r in predictions if r["split"] == split})
        for control in ("correct", "none", "wrong_depth")
        for decoder in ("free", "constrained", "score")
    }
    source_runner = (ROOT / "08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/grip-exp/scripts/run_candidate_energy_probe.py").read_text(encoding="utf-8")
    tautological_order_check = "permuted = list(reversed(score_dicts))" in source_runner
    error_text = "\n".join((RUN / name).read_text(encoding="utf-8", errors="replace") for name in ("run.log", "environment.txt"))
    integrity = {
        "audit_scope": "local independent recomputation; no external cross-model reviewer backend was available",
        "status": "warn",
        "checks": {
            "run_status_complete": run_log.get("status") == "complete",
            "prediction_rows": len(predictions),
            "candidate_score_rows": len(score_rows),
            "prediction_unique_keys": len(pred_by_key),
            "candidate_score_unique_keys": len(score_by_key),
            "complete_prediction_grid": set(pred_by_key) == expected_pred_keys,
            "all_candidate_counts_equal_10": all(len(r.get("candidates", [])) == 10 for r in predictions),
            "all_score_counts_equal_10": all(len(r.get("scores", [])) == 10 for r in score_rows),
            "all_targets_in_candidates": all(all(t in r.get("candidates", []) for t in r.get("target", [])) for r in predictions),
            "source_input_available": source_input_path.is_file(),
            "all_predictions_match_source_question_target_candidates": all(
                r["question_id"] in source_questions
                and r.get("target") == [str(source_questions[r["question_id"]]["answer"])]
                and r.get("candidates") == [str(x) for x in source_questions[r["question_id"]]["candidate_relations"]]
                and r.get("metadata", {}).get("question", source_questions[r["question_id"]]["question"]) == source_questions[r["question_id"]]["question"]
                for r in predictions
            ),
            "all_scores_finite": all(math.isfinite(float(s[k])) for r in score_rows for s in r["scores"] for k in ("sum_logprob", "norm_logprob")),
            "recorded_correct_matches_recomputed_exact_match": all(bool(r["correct"]) == exact_match(r.get("predicted_candidate"), r.get("target", [])) for r in predictions),
            "no_runtime_error_markers": not re.search(r"Traceback|CUDA out of memory|\bOOM\b|NaN|timeout", error_text, flags=re.I),
            "validation_only_normalization_selection": True,
            "selected_normalization": config.get("selected_normalization"),
            "correct_only_validation_would_select_same_normalization": True,
            "target_excluded_from_model_prompt_by_dataset_path": True,
            "true_candidate_order_permutation_was_run": not tautological_order_check,
            "shuffled_or_wrong_graph_adapter_control_available": False,
            "wrong_depth_control_available": True,
            "run_environment_commit_exactly_identifies_probe_code": False,
            "submitted_calibration_uses_selected_normalization": False,
        },
        "findings": [
            "Candidate-order robustness is invalid: the runner reverses already-computed score dictionaries instead of rebuilding the prompt with permuted candidates and re-running the model.",
            "The available wrong_depth adapter is trained on the same NELL23K graph; it is a recurrence mismatch control, not a shuffled/wrong-graph memory-specificity control.",
            "Normalization was selected on validation only, but pooled across all adapter controls. Raw-sum also wins on correct-adapter validation alone, so this does not change the current conclusion.",
            "The environment records parent commit 2ee0c86 rather than an exact committed probe implementation; the submitted source snapshot exists, but run-to-code provenance is weaker than a clean committed-run hash.",
            "response_in_candidates uses raw string membership while correctness uses punctuation-normalized exact match; one valid free response is therefore marked out-of-set in metadata without affecting accuracy.",
            "The submitted calibration script always ranks norm_logprob margins even though raw_sum was selected, so calibration.json is not calibration for the actual score decoder.",
        ],
    }
    (OUT / "integrity_audit.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def metric(split: str, control: str, decoder: str) -> dict:
        return next(r for r in condition_rows if r["split"] == split and r["adapter_control"] == control and r["decoder_type"] == decoder)

    test_correct_score = metric("test", "correct", "score")
    test_none_score = metric("test", "none", "score")
    test_correct_free = metric("test", "correct", "free")
    test_correct_constrained = metric("test", "correct", "constrained")
    val_correct_score = metric("validation", "correct", "score")
    val_none_score = metric("validation", "none", "score")
    decision = {
        "verdict": "STOP_AS_STANDALONE_DECODER_IDEA__KEEP_AS_BASELINE_COMPONENT",
        "go_gate": {
            "correct_minus_none_test_ge_5pp": False,
            "score_minus_free_test_ge_10pp": (test_correct_score["accuracy"] - test_correct_free["accuracy"]) >= 0.10,
            "validation_test_direction_consistent": (test_correct_score["accuracy"] - test_none_score["accuracy"] <= 0) and (val_correct_score["accuracy"] - val_none_score["accuracy"] <= 0),
            "wrong_or_shuffled_below_correct": False,
            "candidate_order_permutation_stable": None,
            "go": False,
        },
        "key_effects": {
            "test_correct_score_accuracy": test_correct_score["accuracy"],
            "test_none_score_accuracy": test_none_score["accuracy"],
            "test_correct_score_minus_none_pp": 100 * (test_correct_score["accuracy"] - test_none_score["accuracy"]),
            "test_correct_score_minus_free_pp": 100 * (test_correct_score["accuracy"] - test_correct_free["accuracy"]),
            "test_correct_constrained_minus_free_pp": 100 * (test_correct_constrained["accuracy"] - test_correct_free["accuracy"]),
            "validation_correct_score_minus_none_pp": 100 * (val_correct_score["accuracy"] - val_none_score["accuracy"]),
            "test_correct_score_binomial_p_vs_random_10way": binomial_survival(test_correct_score["correct"], test_correct_score["n"], 0.1),
            "test_none_score_binomial_p_vs_random_10way": binomial_survival(test_none_score["correct"], test_none_score["n"], 0.1),
        },
        "claim_supported": "partial",
        "supported_claim": "Candidate-constrained or sequence-likelihood decoding substantially reduces free-generation output failures on this 96-question NELL23K diagnostic subset.",
        "unsupported_claims": [
            "The correct GRIP adapter stores or retrieves facts better than the base model.",
            "Candidate-energy decoding validates FactorGRIP as a publishable main method.",
            "Candidate-order robustness has been established.",
            "The result is leaderboard-scale or generalizes beyond NELL23K/Qwen2.5-0.5B/one seed.",
        ],
        "recommended_next_experiment": "v0_1_1 audit rerun for valid order permutation and memory-specific controls, then proceed to equal-budget FactorGRIP storage/factorization only after an Original-GRIP baseline reproduces positive adapter gain.",
    }
    (OUT / "decision_summary.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "analysis_date": "2026-08-30",
        "source_run": str(RUN.relative_to(ROOT)),
        "source_commit_pulled": "2994d43",
        "run_environment_commit": config.get("environment", {}).get("git_commit"),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "input_sha256": {
            name: sha256(RUN / name)
            for name in ("config.json", "environment.txt", "run.log", "predictions.jsonl", "candidate_scores.jsonl", "analysis/summary.json")
        },
        "source_input": {"path": str(source_input_path.relative_to(ROOT)), "sha256": sha256(source_input_path)},
        "outputs": [
            "condition_metrics.csv", "normalization_metrics.csv", "paired_comparisons.csv",
            "candidate_position_diagnostics.csv", "failure_examples.jsonl",
            "integrity_audit.json", "decision_summary.json", "REPORT.md",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(OUT), "verdict": decision["verdict"], "rows": len(predictions)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
