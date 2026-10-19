"""Paired McNemar tests, question-bootstrap CIs, and error decomposition.

CPU only. Reads frozen prediction JSONL files for one (B1, Random-K) pair per
dataset; no model is loaded. Writes JSON + Markdown and, when matplotlib is
importable, the side-by-side error decomposition figure.

The uncertainty here is over question sampling for one trained pair. It does
not measure training-seed variance.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.paired_stats import (  # noqa: E402
    ERROR_CATEGORIES,
    categorize,
    category_counts,
    mcnemar_exact,
    normalize_answer,
    ranks_from_scores,
    stratified_bootstrap,
    transition_matrix,
)

RUNS = HNG / "results/runs"
NELL_RAW = HNG / "../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/data/raw_datasets/nell23k"

NELL_EVAL = HNG / "data/nell23k/recurrent_relation_prediction_full.aligned.json"
FB_EVAL = HNG / "data/fb15k237/recurrent_relation_prediction_full.json"
NELL_RK_2026 = RUNS / "20261011_accumfix_full_decode/random_k/listed"

# Post-fix pairs only. NELL: 20261013_accumfix_b1_full decoded the *new* B1
# adapter into full_decode/listed (full_decode/b1 is the pre-fix historical B1).
# Pairs marked optional are skipped until their prediction files exist.
DEFAULT_PAIRS = {
    "nell23k": {
        "dataset": "nell23k", "seed": 2026, "kind": "main", "figure": True,
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": RUNS / "20261013_accumfix_b1_full/full_decode/listed", "left_label": "B1",
        "right": NELL_RK_2026, "right_label": "Random-K",
    },
    "fb15k237": {
        "dataset": "fb15k237", "seed": 2026, "kind": "main", "figure": True,
        "eval_file": FB_EVAL, "vocab": "fb_manifest",
        "left": RUNS / "20261015_fb15k237_qwen7b_b1_randomk/full/b1", "left_label": "B1",
        "right": RUNS / "20261015_fb15k237_qwen7b_b1_randomk/full/listed",
        "right_label": "Random-K",
    },
    "nell23k_compute_matched": {
        "dataset": "nell23k", "seed": 2026, "kind": "compute_matched", "optional": True,
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": RUNS / "20261018_nell_b1_compute_matched_37ep/full_decode/b1",
        "left_label": "B1 (37 QA epochs, compute-matched)",
        "right": NELL_RK_2026, "right_label": "Random-K",
    },
    # Sampler ablation on NELL23K (seed 2026, same Stage 1, same post-fix trainer).
    # kind="ablation" keeps these out of the seed summary.
    "nell23k_calibrated": {
        "dataset": "nell23k", "seed": 2026, "kind": "ablation",
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": RUNS / "20261013_accumfix_b1_full/full_decode/listed", "left_label": "B1",
        "right": RUNS / "20261011_accumfix_full_decode/calibrated/listed",
        "right_label": "Calibrated",
    },
    "nell23k_soft_mix": {
        "dataset": "nell23k", "seed": 2026, "kind": "ablation",
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": RUNS / "20261013_accumfix_b1_full/full_decode/listed", "left_label": "B1",
        "right": RUNS / "20261014_accumfix_soft_mix_full_decode/listed",
        "right_label": "Soft-Mix",
    },
    "nell23k_calibrated_vs_random_k": {
        "dataset": "nell23k", "seed": 2026, "kind": "ablation",
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": NELL_RK_2026, "left_label": "Random-K",
        "right": RUNS / "20261011_accumfix_full_decode/calibrated/listed",
        "right_label": "Calibrated",
    },
    "nell23k_soft_mix_vs_random_k": {
        "dataset": "nell23k", "seed": 2026, "kind": "ablation",
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": NELL_RK_2026, "left_label": "Random-K",
        "right": RUNS / "20261014_accumfix_soft_mix_full_decode/listed",
        "right_label": "Soft-Mix",
    },
    "nell23k_seed2027": {
        "dataset": "nell23k", "seed": 2027, "kind": "main", "optional": True,
        "eval_file": NELL_EVAL, "vocab": "nell_train",
        "left": RUNS / "20261018_nell_seed2027/full/b1", "left_label": "B1",
        "right": RUNS / "20261018_nell_seed2027/full/listed", "right_label": "Random-K",
    },
    "fb15k237_seed2027": {
        "dataset": "fb15k237", "seed": 2027, "kind": "main", "optional": True,
        "eval_file": FB_EVAL, "vocab": "fb_manifest",
        "left": RUNS / "20261018_fb15k237_seed2027/full/b1", "left_label": "B1",
        "right": RUNS / "20261018_fb15k237_seed2027/full/listed", "right_label": "Random-K",
    },
}
SPLITS = ("test", "validation", "all")
PRED_FILES = ("predictions_correct.jsonl", "predictions_closed_set.jsonl")


def pair_ready(cfg: dict, n_expected: int | None = None) -> bool:
    """True when both arms have complete generation + closed-set predictions."""
    for side in ("left", "right"):
        for name in PRED_FILES:
            path = Path(cfg[side]) / name
            if not path.is_file():
                return False
            if n_expected is not None:
                with path.open(encoding="utf-8") as handle:
                    if sum(1 for line in handle if line.strip()) != n_expected:
                        return False
    return True


def load_eval(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload[0] if isinstance(payload, list) else payload
    out = {}
    for sample in record["recurrent_questions"]:
        if sample["split"] not in ("test", "validation"):
            continue
        candidates = sample["candidate_relations"]
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        out[sample["question_id"]] = {"split": sample["split"], "answer": sample["answer"],
                                      "candidates": list(candidates)}
    return out


def load_vocab(kind: str) -> set[str]:
    if kind == "nell_train":
        lines = (NELL_RAW / "train.txt").read_text(encoding="utf-8").splitlines()
        relations = {line.split("\t")[1] for line in lines if line.strip()}
    elif kind == "fb_manifest":
        manifest = json.loads((HNG / "data/fb15k237/preparation_manifest.json").read_text())
        relations = set(manifest["relation_order"])
    else:
        raise ValueError(kind)
    return {normalize_answer(rel) for rel in relations}


def load_predictions(path: Path, expected: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["question_id"] in rows:
                raise ValueError(f"duplicate question_id in {path}: {row['question_id']}")
            rows[row["question_id"]] = row
    if rows.keys() != expected.keys():
        raise ValueError(f"{path}: {len(rows)} rows do not match {len(expected)} eval questions")
    for qid, row in rows.items():
        if row["split"] != expected[qid]["split"]:
            raise ValueError(f"split mismatch for {qid} in {path}")
    return rows


def compare_metric(left: np.ndarray, right: np.ndarray, split_labels: np.ndarray,
                   *, binary: bool, seed: int, draws: int) -> dict:
    out = {}
    for split in SPLITS:
        mask = np.ones(len(left), bool) if split == "all" else split_labels == split
        a, b, strata = left[mask], right[mask], split_labels[mask]
        entry = {"count": int(mask.sum()), "left": float(a.mean()), "right": float(b.mean()),
                 "delta": float((b - a).mean())}
        entry.update(stratified_bootstrap(a, b, strata, draws=draws, seed=seed))
        if binary:
            gains = int(((a == 0) & (b == 1)).sum())
            regressions = int(((a == 1) & (b == 0)).sum())
            entry.update({"gains": gains, "regressions": regressions,
                          "mcnemar": mcnemar_exact(gains, regressions)})
        out[split] = entry
    return out


METRIC_NAMES = {"gen": "generate_em", "closed": "closed_set_em", "rr": "closed_set_mrr",
                "hits3": "closed_set_hits@3"}


def load_arm(base: Path, expected: dict, ids: list[str], vocab: set[str]) -> dict:
    gen = load_predictions(base / "predictions_correct.jsonl", expected)
    closed = load_predictions(base / "predictions_closed_set.jsonl", expected)
    ranks = []
    for q in ids:
        gold = closed[q]["target"][0]
        if gold not in closed[q]["scores"]:
            raise ValueError(f"gold missing from scores: {q}")
        ranks.append(ranks_from_scores(closed[q]["scores"], gold))
    ranks = np.array(ranks)
    return {
        "gen": np.array([gen[q]["correct"] for q in ids], dtype=float),
        "closed": np.array([closed[q]["correct"] for q in ids], dtype=float),
        "rr": 1.0 / ranks,
        "hits3": (ranks <= 3).astype(float),
        "categories": [categorize({**gen[q], "candidates": expected[q]["candidates"]}, vocab)
                       for q in ids],
        "source": str(base),
    }


def analyze(name: str, cfg: dict, *, draws: int, seed: int) -> dict:
    expected = load_eval(Path(cfg["eval_file"]))
    vocab = load_vocab(cfg["vocab"])
    ids = sorted(expected)
    split_labels = np.array([expected[q]["split"] for q in ids])
    L = load_arm(Path(cfg["left"]), expected, ids, vocab)
    R = load_arm(Path(cfg["right"]), expected, ids, vocab)
    report = {"pair": name, "dataset": cfg["dataset"], "training_seed": cfg["seed"],
              "kind": cfg["kind"], "left": cfg["left_label"], "right": cfg["right_label"],
              "left_source": L["source"], "right_source": R["source"],
              "eval_file": str(cfg["eval_file"]), "n_questions": len(ids),
              "relation_vocab_size": len(vocab), "metrics": {}}
    for key, binary in (("gen", True), ("closed", True), ("rr", False), ("hits3", True)):
        report["metrics"][METRIC_NAMES[key]] = compare_metric(
            L[key], R[key], split_labels, binary=binary, seed=seed, draws=draws)
    decomposition = {}
    for split in SPLITS:
        mask = [split == "all" or s == split for s in split_labels]
        lc = [c for c, m in zip(L["categories"], mask) if m]
        rc = [c for c, m in zip(R["categories"], mask) if m]
        decomposition[split] = {"left": category_counts(lc), "right": category_counts(rc),
                                "transitions_left_to_right": transition_matrix(lc, rc)}
    report["error_decomposition"] = decomposition
    return report


def seed_summary(reports: list[dict]) -> list[dict]:
    """Mean ± sample std over training seeds of main B1-vs-Random-K pairs."""
    out = []
    for dataset in sorted({r["dataset"] for r in reports if r["kind"] == "main"}):
        runs = sorted((r for r in reports if r["kind"] == "main" and r["dataset"] == dataset),
                      key=lambda r: r["training_seed"])
        for metric in ("generate_em", "closed_set_em"):
            for split in ("test", "all"):
                rows = [r["metrics"][metric][split] for r in runs]
                entry = {"dataset": dataset, "metric": metric, "split": split,
                         "seeds": [r["training_seed"] for r in runs]}
                for field in ("left", "right", "delta"):
                    values = np.array([row[field] for row in rows])
                    entry[f"{field}_mean"] = float(values.mean())
                    entry[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
                    entry[f"{field}_values"] = values.tolist()
                out.append(entry)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def p_text(mc: dict) -> str:
    if mc["p_value"] > 1e-4:
        return f"{mc['p_value']:.4f}"
    return f"1e{mc['log10_p_value']:.0f}"


def _metric_row(rep: dict, metric: str, split: str) -> str:
    e = rep["metrics"][metric][split]
    if metric == "closed_set_mrr":
        fmt, delta = (lambda v: f"{v:.3f}"), f"{e['delta']:+.3f}"
        ci = "[{:+.3f}, {:+.3f}]".format(*e["delta_ci95"])
    else:
        fmt, delta = pct, f"{100 * e['delta']:+.2f}"
        ci = "[{:+.2f}, {:+.2f}]".format(*(100 * v for v in e["delta_ci95"]))
    gl = f"{e['gains']} / {e['regressions']}" if "gains" in e else "—"
    p = p_text(e["mcnemar"]) if "mcnemar" in e else "—"
    return (f"| {rep['pair']} | {rep['left']} vs {rep['right']} | {metric} | {split} | "
            f"{e['count']} | {fmt(e['left'])} | {fmt(e['right'])} | {delta} | {ci} | {gl} | {p} |")


def markdown(reports: list[dict], seeds: list[dict], skipped: list[str]) -> str:
    lines = ["# Paired significance and error decomposition", "",
             "Question-level uncertainty for each trained pair. Bootstrap resamples questions "
             "within each split (10,000 draws); McNemar is the two-sided exact test on "
             "discordant questions. Neither covers training-seed variance; see the seed table "
             "for that.", ""]
    if skipped:
        lines += ["Pairs not yet available (predictions missing or incomplete): "
                  + ", ".join(f"`{s}`" for s in skipped), ""]
    lines += ["## Paired comparison (left vs right; Δ = right − left)", "",
              "| Pair | Arms | Metric | Split | n | Left | Right | Δ (pp) | 95% CI of Δ "
              "| gains / losses | McNemar p |",
              "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for rep in reports:
        for metric in rep["metrics"]:
            for split in SPLITS:
                lines.append(_metric_row(rep, metric, split))
    if seeds:
        lines += ["", "## Across training seeds (main B1 vs Random-K pairs)", "",
                  "Mean ± sample std over seeds; std is blank with one seed.", "",
                  "| Dataset | Metric | Split | Seeds | B1 | Random-K | Δ (pp) |",
                  "|---|---|---|---|---:|---:|---:|"]
        for e in seeds:
            def ms(field: str) -> str:
                mean, std = 100 * e[f"{field}_mean"], e[f"{field}_std"]
                return f"{mean:.2f}" + (f" ± {100 * std:.2f}" if std is not None else "")
            lines.append(f"| {e['dataset']} | {e['metric']} | {e['split']} | "
                         f"{','.join(map(str, e['seeds']))} | {ms('left')} | {ms('right')} | "
                         f"{ms('delta')} |")
    lines += ["", "## Free-generation error decomposition (all val+test)", "",
              "`wrong_in_list`: picked one of the 9 printed distractors. "
              "`wrong_off_list_in_vocab`: a real train relation that is not among the 10. "
              "`wrong_abstain`: a refusal such as \"I don't know\". "
              "`wrong_oov`: any other string that is not a train relation. "
              "Membership uses the EM normalization.", "",
              "| Pair | Arm | n | " + " | ".join(ERROR_CATEGORIES) + " |",
              "|---|---|---:|" + "---:|" * len(ERROR_CATEGORIES)]
    for rep in reports:
        n = rep["error_decomposition"]["all"]
        for side in ("left", "right"):
            c = n[side]
            total = sum(c.values())
            cells = " | ".join(f"{c[k]} ({100 * c[k] / total:.1f}%)" for k in ERROR_CATEGORIES)
            lines.append(f"| {rep['pair']} | {rep[side]} | {total} | {cells} |")
    lines += ["", "## Transitions left → right (all val+test)", ""]
    for rep in reports:
        m = rep["error_decomposition"]["all"]["transitions_left_to_right"]
        lines += [f"**{rep['pair']}** (rows {rep['left']}, columns {rep['right']})", "",
                  "| left \\ right | " + " | ".join(ERROR_CATEGORIES) + " |",
                  "|---|" + "---:|" * len(ERROR_CATEGORIES)]
        for a in ERROR_CATEGORIES:
            lines.append(f"| {a} | " + " | ".join(str(m[a][b]) for b in ERROR_CATEGORIES) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(reports: list[dict], path: Path) -> bool:
    reports = [r for r in reports if DEFAULT_PAIRS.get(r["pair"], {}).get("figure")]
    if not reports:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = {"wrong_in_list": "Wrong, in candidate list",
              "wrong_off_list_in_vocab": "Wrong, real relation off list",
              "wrong_abstain": "Abstained (\"I don't know\")",
              "wrong_oov": "Wrong, other non-relation string"}
    colors = {"wrong_in_list": "#d95f02", "wrong_off_list_in_vocab": "#7570b3",
              "wrong_abstain": "#999999", "wrong_oov": "#1b9e77"}
    names = {"nell23k": "NELL23K", "fb15k237": "FB15K-237"}
    fig, axes = plt.subplots(1, len(reports), figsize=(3.6 * len(reports), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reports):
        counts = rep["error_decomposition"]["all"]
        sides, ticks = ("left", "right"), [rep["left"], rep["right"]]
        bottom = np.zeros(2)
        for key in labels:
            vals = np.array([100 * counts[s][key] / sum(counts[s].values()) for s in sides])
            ax.bar(ticks, vals, bottom=bottom, color=colors[key], label=labels[key], width=0.55)
            for i, v in enumerate(vals):
                if v >= 1.5:
                    ax.text(i, bottom[i] + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=8, color="white")
            bottom += vals
        ax.set_title(f"{names.get(rep['dataset'], rep['dataset'])} "
                     f"(n={rep['n_questions']:,})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error rate, % of questions (free generation)")
    handles, legend_names = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], legend_names[::-1], fontsize=7, frameon=False, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=HNG / "results/paired_stats")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, skipped = [], []
    for name in args.pairs:
        cfg = DEFAULT_PAIRS[name]
        n_expected = len(load_eval(Path(cfg["eval_file"])))
        if cfg.get("optional") and not pair_ready(cfg, n_expected):
            skipped.append(name)
            continue
        reports.append(analyze(name, cfg, draws=args.draws, seed=args.seed))
    seeds = seed_summary(reports)
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "uncertainty_scope": "question resampling within split for one trained pair; "
                                    "seed_summary covers training seeds",
               "bootstrap_draws": args.draws, "bootstrap_seed": args.seed,
               "skipped_pairs": skipped, "reports": reports, "seed_summary": seeds}
    (args.output_dir / "paired_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = markdown(reports, seeds, skipped)
    (args.output_dir / "paired_stats.md").write_text(text, encoding="utf-8")
    drew = plot(reports, args.output_dir / "error_decomposition.png")
    print(text)
    print(f"[paired_stats] wrote {args.output_dir} figure={'yes' if drew else 'no'} "
          f"skipped={skipped}")


if __name__ == "__main__":
    main()
