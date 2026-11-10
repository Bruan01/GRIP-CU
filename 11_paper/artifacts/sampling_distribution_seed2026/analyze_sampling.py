"""Audit actual frozen candidate banks; export portable distribution source data.

Run from any directory using guardenv Python. No candidate resampling or GPU work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HNG = ROOT / "08_experiments/Hard_Negative_Contrastive_GRIP"
sys.path.insert(0, str(HNG / "src"))
from hard_negative_grip.counterfactuals import load_train_relation_profiles
from hard_negative_grip.official_lists import load_triples

RUNS = Path("/home/kieran/grip-cu-runs")
HPO = HNG / "results/hpo/stage2_three_dataset_20261030"
LABELS = {"nell23k": "NELL23K", "codexm": "CoDEx-M", "fb15k237_2": "FB15K237-2", "wn18rr": "WN18RR"}
EXPECTED = {"nell23k": 2974, "codexm": 2715, "fb15k237_2": 3109, "wn18rr": 2709}
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register(path):
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if key not in SOURCES:
        SOURCES[key] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def stream(path):
    with register(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def bank(path):
    data = {}
    for row in stream(path):
        qid = row["question_id"]
        assert qid not in data, (path, qid)
        assert row["split"] == "train" and row["seed"] == 2026
        neg = row["negative_relations"]
        assert len(neg) == len(set(neg)) == 9
        assert row["gold_relation"] not in neg
        data[qid] = row
    return data


def write_csv(name, data):
    assert data
    with (OUT / "source_data" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def distances(p, ref, log_counts):
    jsd = float(jensenshannon(p, ref, base=2) ** 2)
    if abs(jsd) < 1e-15:
        jsd = 0.0
    return {"js_divergence_bits": jsd,
            "total_variation": float(np.abs(p - ref).sum() / 2),
            "log2_frequency_wasserstein": float(wasserstein_distance(log_counts, log_counts, u_weights=p, v_weights=ref))}


def main():
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    relation_rows, bin_rows, conditional_rows, metric_rows, query_rows = [], [], [], [], []
    summaries, validations = {}, {}
    for dataset, label in LABELS.items():
        print("Reading", label, flush=True)
        raw = ROOT / "04_benchmark/data" / dataset / "raw"
        register(raw / "train.txt")
        mapping = raw / "rel2text.json"
        if mapping.exists():
            register(mapping)
        profiles = load_train_relation_profiles(raw, relation_text_file=mapping if mapping.exists() else None,
                                                strip_first_underscore=dataset == "wn18rr")
        raw_relation_ids = {relation for _, relation, _ in load_triples(raw / "train.txt")}
        asset = (HNG / "results/runs/wn18rr_sccl_3090_20261023/artifacts/structure_counterfactual_bank"
                 if dataset == "wn18rr" else HPO / dataset / "assets/k9")
        saved_profiles = read_json(asset / "relation_profiles.json")
        assert set(profiles) == set(saved_profiles)
        assert all(p.as_dict() == saved_profiles[r] for r, p in profiles.items())
        relations = sorted(profiles, key=lambda r: (-profiles[r].edge_count, r))
        idx = {r: i for i, r in enumerate(relations)}
        counts = np.array([profiles[r].edge_count for r in relations], dtype=float)
        bins = np.floor(np.log2(counts)).astype(int)
        log_counts = np.log2(counts)
        methods = {}
        if dataset in ("nell23k", "wn18rr"):
            run = RUNS / f"{dataset}_source_controls_20261107_seed2026"
            methods["full"] = bank(run / "reference/full_bank.jsonl")
            methods["uniform_observed"] = bank(run / "bank/random9.jsonl")
            methods["quota_observed"] = bank(run / "bank/frequency_matched.jsonl")
            pool_path = run / "bank/pool_audit.jsonl"
        elif dataset == "fb15k237_2":
            run = RUNS / "fb_random9_control_20261105_seed2026"
            methods["full"] = bank(run / "reference/counterfactual_bank.jsonl")
            methods["uniform_observed"] = bank(run / "bank/random9.jsonl")
            methods["quota_observed"] = bank(RUNS / "fb_frequency_matched_control_20261106_seed2026/bank/frequency_matched.jsonl")
            methods["profile_only_observed"] = bank(RUNS / "fb_profile_only_control_20261106_seed2026/bank/profile_only.jsonl")
            methods["violation_only_observed"] = bank(RUNS / "fb_violation_only_control_20261106_seed2026/bank/violation_only.jsonl")
            pool_path = run / "bank/pool_audit.jsonl"
        else:
            methods["full"] = bank(asset / "counterfactual_bank.jsonl")
            pool_path = None
        full = methods["full"]
        original = bank(asset / "counterfactual_bank.jsonl")
        assert set(full) == set(original) and len(full) == EXPECTED[dataset]
        assert all(full[q]["negative_relations"] == original[q]["negative_relations"] and
                   full[q]["gold_relation"] == original[q]["gold_relation"] for q in full)
        for method, data in methods.items():
            assert set(data) == set(full), method
            assert all(data[q]["gold_relation"] == full[q]["gold_relation"] for q in full)
        pools = {}
        unverified = set()
        if pool_path is not None:
            for row in stream(pool_path):
                qid = row["question_id"]
                assert qid not in pools and row["parent_pool_verified"]
                pool = row["legal_relations"]
                assert len(pool) == len(set(pool)) == row["pool_size"]
                pools[qid] = pool
                if row.get("entity_pair_verification") == "unverified_pair_original_parent_mask":
                    unverified.add(qid)
            if dataset == "wn18rr":
                unverified = {q for q, r in methods["uniform_observed"].items() if not r["train_pair_verified"]}
                assert len(unverified) == 55
        else:
            metadata = read_json(asset / "metadata.json")
            score_path = Path(metadata["score_path"])
            grouped = defaultdict(list)
            for row in stream(score_path):
                assert row["split"] == "train"
                qid = row["qa_id"]
                assert qid in full
                if row["is_valid_negative"] and not row["is_gold"]:
                    grouped[qid].append(row["candidate_relation"])
            pools = dict(grouped)
            audit = {r["question_id"]: r for r in stream(asset / "sampler_audit.jsonl")}
            assert all(len(pool) == len(set(pool)) == audit[q]["pool_size"] for q, pool in pools.items())
        assert set(pools) == set(full)
        nq, nr = len(full), len(relations)
        mass = {"train_edges": counts / counts.sum(), "eligible_gold": np.zeros(nr), "legal_uniform_expected": np.zeros(nr)}
        counts_by_method = {m: np.zeros(nr) for m in methods}
        conditional = {m: Counter() for m in ["full", "legal_uniform_expected"]}
        gap = defaultdict(float)
        per_q_exact_quota = 0
        verified_full_counts = np.zeros(nr)
        verified_uniform_expected = np.zeros(nr)
        for qid, row in full.items():
            gold = row["gold_relation"]
            gi = idx[gold]
            gb = int(bins[gi])
            mass["eligible_gold"][gi] += 1 / nq
            pool = pools[qid]
            assert gold not in pool and len(pool) >= 9 and all(r in idx for r in pool)
            poolset = set(pool)
            for r in pool:
                ri = idx[r]
                w = 1 / (nq * len(pool))
                mass["legal_uniform_expected"][ri] += w
                conditional["legal_uniform_expected"][(gb, int(bins[ri]))] += 9 / len(pool)
                gap["legal_uniform_expected"] += abs(log_counts[ri] - log_counts[gi]) / (nq * len(pool))
                if qid not in unverified:
                    verified_uniform_expected[ri] += 1 / ((nq - len(unverified)) * len(pool))
            for method, data in methods.items():
                negs = data[qid]["negative_relations"]
                assert set(negs) <= poolset
                for r in negs:
                    counts_by_method[method][idx[r]] += 1
                    gap[method] += abs(log_counts[idx[r]] - log_counts[gi]) / (9 * nq)
                    if method == "full":
                        conditional["full"][(gb, int(bins[idx[r]]))] += 1
                        if qid not in unverified:
                            verified_full_counts[idx[r]] += 1
            if "quota_observed" in methods:
                a = Counter(int(bins[idx[r]]) for r in row["negative_relations"])
                b = Counter(int(bins[idx[r]]) for r in methods["quota_observed"][qid]["negative_relations"])
                assert a == b, (dataset, qid)
                per_q_exact_quota += 1
            query_rows.append({"dataset": dataset, "question_id": qid, "gold_relation": gold,
                               "gold_train_edges": int(counts[gi]), "gold_frequency_bin": gb,
                               "legal_pool_size": len(pool), "full_k": 9,
                               "full_mean_log2_frequency": float(np.mean([log_counts[idx[r]] for r in row["negative_relations"]])),
                               "full_mean_abs_log2_gap_to_gold": float(np.mean([abs(log_counts[idx[r]] - log_counts[gi]) for r in row["negative_relations"]])),
                               "train_pair_scope": "unverified_historical_mask" if qid in unverified else ("stored_score_pool" if dataset == "codexm" else "verified_parent_pool")})
        for method, values in counts_by_method.items():
            assert values.sum() == 9 * nq
            mass[method] = values / values.sum()
        assert all(np.isclose(v.sum(), 1, atol=1e-10) for v in mass.values())
        all_bins = range(int(bins.min()), int(bins.max()) + 1)
        for method, values in mass.items():
            for i, r in enumerate(relations):
                relation_rows.append({"dataset": dataset, "relation": r, "train_rank": i + 1,
                                      "train_edge_count": int(counts[i]), "frequency_bin": int(bins[i]),
                                      "method": method, "probability": float(values[i])})
            for b in all_bins:
                bin_rows.append({"dataset": dataset, "method": method, "frequency_bin": b,
                                 "probability": float(values[bins == b].sum()), "relations_in_bin": int((bins == b).sum())})
            if method in ("train_edges", "eligible_gold"):
                continue
            for ref in ("train_edges", "eligible_gold", "legal_uniform_expected"):
                vals = distances(values, mass[ref], log_counts)
                assert -1e-12 <= vals["js_divergence_bits"] <= 1 + 1e-12
                metric_rows.append({"dataset": dataset, "method": method, "reference": ref, **vals,
                                    "entropy_bits": float(-sum(v * math.log2(v) for v in values if v)),
                                    "relations_with_mass": int((values > 0).sum()),
                                    "spearman_vs_train_edges": float(spearmanr(values, counts).statistic),
                                    "mean_abs_log2_frequency_gap_to_gold": gap[method]})
        for method, joint in conditional.items():
            for gb in all_bins:
                denominator = sum(joint[(gb, nb)] for nb in all_bins)
                for nb in all_bins:
                    value = joint[(gb, nb)]
                    conditional_rows.append({"dataset": dataset, "method": method,
                                             "gold_frequency_bin": gb, "negative_frequency_bin": nb,
                                             "effective_slots": float(value), "row_slots": float(denominator),
                                             "conditional_probability": float(value / denominator) if denominator else ""})
                if denominator:
                    assert np.isclose(sum(joint[(gb, nb)] / denominator for nb in all_bins), 1)
        if "quota_observed" in mass:
            assert all(np.isclose(mass["full"][bins == b].sum(), mass["quota_observed"][bins == b].sum(), atol=1e-12) for b in all_bins)
        summaries[dataset] = {"label": label, "train_relations": nr, "raw_train_relation_ids": len(raw_relation_ids), "train_edges": int(counts.sum()),
                              "eligible_questions": nq, "negative_slots": nq * 9, "k": 9, "seed": 2026,
                              "observed_methods": list(methods), "quota_matched_questions": per_q_exact_quota,
                              "unverified_historical_pairs": len(unverified),
                              "controls_missing": ["trained Uniform", "trained quota control"] if dataset == "codexm" else [],
                              "raw_head10pct_edge_share": float(mass["train_edges"][:math.ceil(nr / 10)].sum()),
                              "full_head10pct_relation_share": float(mass["full"][:math.ceil(nr / 10)].sum())}
        summaries[dataset]["eligible_gold_to_train_edges"] = distances(mass["eligible_gold"], mass["train_edges"], log_counts)
        if unverified:
            verified = verified_full_counts / verified_full_counts.sum()
            summaries[dataset]["verified_subset_sensitivity"] = {
                "n": nq - len(unverified), "full_to_train_edges": distances(verified, mass["train_edges"], log_counts),
                "full_all_vs_verified_tv": float(np.abs(verified - mass["full"]).sum() / 2),
                "uniform_expected_to_train_edges": distances(verified_uniform_expected, mass["train_edges"], log_counts)}
        validations[dataset] = {"status": "pass", "raw_train_profiles_match_frozen_profiles": True,
                                "cohort_and_gold_alignment": True, "all_candidates_unique_and_legal": True,
                                "probabilities_and_conditional_rows_normalized": True,
                                "exact_quota_rows": per_q_exact_quota, "independently_verified_all_train_pairs": not bool(unverified)}
        print(label, nq, nr, "passed", flush=True)
    write_csv("relation_distribution.csv", relation_rows)
    write_csv("frequency_bin_distribution.csv", bin_rows)
    write_csv("conditional_frequency_distribution.csv", conditional_rows)
    write_csv("distribution_metrics.csv", metric_rows)
    write_csv("eligible_question_summary.csv", query_rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    (OUT / "validation.json").write_text(json.dumps({"status": "pass", "datasets": validations,
        "claim_type": "computed", "source_scope": "train-only raw edges and frozen seed2026 banks; no new sampling/training/inference",
        "statistics": "descriptive marginals and conditional frequency distributions; no independent seed uncertainty or hypothesis tests"}, ensure_ascii=False, indent=2) + "\n")
    for path in [HNG / "src/hard_negative_grip/counterfactuals.py", HNG / "src/hard_negative_grip/official_lists.py"]:
        register(path)
    (OUT / "provenance.json").write_text(json.dumps({"inputs": SOURCES, "analysis_script_sha256": sha(Path(__file__)),
        "outputs": {p.name: sha(p) for p in sorted((OUT / "source_data").glob("*.csv"))}}, ensure_ascii=False, indent=2) + "\n")
    print("All four datasets exported to", OUT, flush=True)


if __name__ == "__main__":
    main()
