"""Independent aggregate identities and entropy-form JS checks; portable CPU QA."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / "source_data" / name).open() as handle:
        return list(csv.DictReader(handle))


def js(p, q):
    out = 0.0
    for a, b in zip(p, q):
        m = (a + b) / 2
        if a:
            out += a * math.log2(a / m) / 2
        if b:
            out += b * math.log2(b / m) / 2
    return out


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    rel = defaultdict(list)
    for r in read("relation_distribution.csv"):
        rel[(r["dataset"], r["method"])].append(r)
    for key in rel:
        rel[key].sort(key=lambda r: int(r["train_rank"]))
        assert math.isclose(sum(float(r["probability"]) for r in rel[key]), 1, abs_tol=1e-10)
    queries = read("eligible_question_summary.csv")
    assert len({(r["dataset"], r["question_id"]) for r in queries}) == len(queries)
    for dataset, s in summary.items():
        assert len([q for q in queries if q["dataset"] == dataset]) == s["eligible_questions"]
        edges = rel[(dataset, "train_edges")]
        assert sum(int(r["train_edge_count"]) for r in edges) == s["train_edges"]
        assert len(edges) == s["train_relations"]
        assert all(math.isclose(float(r["probability"]), int(r["train_edge_count"]) / s["train_edges"], abs_tol=1e-12) for r in edges)
        full = rel[(dataset, "full")]
        assert all(math.isclose(float(r["probability"]) * s["negative_slots"], round(float(r["probability"]) * s["negative_slots"]), abs_tol=1e-8) for r in full)
        if (dataset, "quota_observed") in rel:
            a, b = defaultdict(float), defaultdict(float)
            for r in full:
                a[int(r["frequency_bin"])] += float(r["probability"])
            for r in rel[(dataset, "quota_observed")]:
                b[int(r["frequency_bin"])] += float(r["probability"])
            assert all(math.isclose(a[k], b[k], abs_tol=1e-12) for k in set(a) | set(b))
    metric_count = 0
    for r in read("distribution_metrics.csv"):
        p = [float(x["probability"]) for x in rel[(r["dataset"], r["method"])]]
        q = [float(x["probability"]) for x in rel[(r["dataset"], r["reference"])]]
        assert math.isclose(js(p, q), float(r["js_divergence_bits"]), abs_tol=1e-12), r
        assert math.isclose(sum(abs(a - b) for a, b in zip(p, q)) / 2, float(r["total_variation"]), abs_tol=1e-12)
        assert all(math.isfinite(float(r[k])) for k in ("js_divergence_bits", "total_variation", "entropy_bits", "spearman_vs_train_edges", "log2_frequency_wasserstein"))
        metric_count += 1
    conditional = defaultdict(list)
    for r in read("conditional_frequency_distribution.csv"):
        conditional[(r["dataset"], r["method"], int(r["gold_frequency_bin"]))].append(r)
    for (dataset, method, gb), group in conditional.items():
        nq = sum(r["dataset"] == dataset and int(r["gold_frequency_bin"]) == gb for r in queries)
        assert math.isclose(sum(float(r["effective_slots"]) for r in group), 9 * nq, abs_tol=1e-7)
        if nq:
            assert math.isclose(sum(float(r["conditional_probability"]) for r in group), 1, abs_tol=1e-10)
        else:
            assert all(r["conditional_probability"] == "" for r in group)
    source_hashes = json.loads((HERE / "provenance.json").read_text())["outputs"]
    for name, digest in source_hashes.items():
        assert hashlib.sha256((HERE / "source_data" / name).read_bytes()).hexdigest() == digest
    out = {"status": "pass", "claim_type": "computed", "eligible_questions": len(queries),
           "marginal_distributions": len(rel), "metric_rows_checked": metric_count,
           "conditional_rows_checked": len(conditional),
           "independent_checks": ["entropy-form base2 JS divergence", "total variation", "raw edge normalization",
                                  "negative-slot integer counts", "exact quota bin identities", "gold row to conditional-slot identities",
                                  "unique question IDs", "finite metrics", "portable source CSV hashes"],
           "scope": "One frozen seed2026 bank per dataset; checks do not establish unknown fact truth or cross-seed uncertainty."}
    (HERE / "independent_validation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
