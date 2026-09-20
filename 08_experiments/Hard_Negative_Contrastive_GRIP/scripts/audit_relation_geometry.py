"""Audit which linear transform, if any, makes relation-embedding negatives useful.

Two questions, both CPU-only:

1. Do cosine neighbours look like related relations? Ground truth from the train
   graph: relations sharing a >=4-char name stem, or sharing a (head_type,
   tail_type) signature. Chance is the base rate over all ordered pairs.
2. Do cosine neighbours look like the *official* 10-way distractors? Uses only
   train-split questions from the aligned record, so nothing is selected on
   val/test, and reports the distractor percentile, the share of distractors that
   even reach the sampling pool, and the sampler probability mass landing on the
   real distractors.

Writes a JSON report; ``--markdown`` also writes the summary tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.embed_negatives import (  # noqa: E402
    DEFAULT_EMBED_POOL_SIZE,
    DEFAULT_EMBED_TEMPERATURE,
    apply_linear_transform,
    geometry_summary,
    l2_normalize,
    load_relation_embeddings,
    softmax,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    load_train_relation_order,
)

CANDIDATE_SPACES: list[dict] = [
    {"mode": "raw"},
    {"mode": "center"},
    {"mode": "allbuttop", "remove": 1},
    {"mode": "allbuttop", "remove": 2},
    {"mode": "allbuttop", "remove": 5},
]
for _k in (32, 64, 128, 197):
    for _alpha in (0.5, 0.75, 1.0):
        CANDIDATE_SPACES.append({"mode": "pca_whiten", "k": _k, "alpha": _alpha})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=HNG / "results/relation_embeddings/s1_qwen7b_20260913.npz",
    )
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_NELL23K)
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=HNG / "data/nell23k/recurrent_relation_prediction.aligned.json",
    )
    parser.add_argument("--pool_size", type=int, default=DEFAULT_EMBED_POOL_SIZE)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5]
    )
    parser.add_argument(
        "--output", type=Path, default=HNG / "results/relation_geometry_audit.json"
    )
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def space_label(spec: dict) -> str:
    if spec["mode"] == "pca_whiten":
        return f"pca_whiten k={spec['k']} a={spec['alpha']}"
    if spec["mode"] == "allbuttop":
        return f"allbuttop m={spec['remove']}"
    return spec["mode"]


def ent_type(name: str) -> str:
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else name


def stem_of(relation: str) -> str:
    return re.sub(r"^concept:", "", relation)


def shared_stem(a: str, b: str) -> int:
    best = 0
    for left in re.split(r"[^a-z]+", stem_of(a)):
        if len(left) < 4:
            continue
        for right in re.split(r"[^a-z]+", stem_of(b)):
            if len(right) < 4:
                continue
            run = 0
            while run < min(len(left), len(right)) and left[run] == right[run]:
                run += 1
            best = max(best, run)
    return best


def build_graph_ground_truth(relations: list[str], raw_dir: Path):
    signatures: dict[str, set] = defaultdict(set)
    frequency: Counter = Counter()
    with (raw_dir / "train.txt").open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            signatures[parts[1]].add((ent_type(parts[0]), ent_type(parts[2])))
            frequency[parts[1]] += 1
    n = len(relations)
    lexical = np.zeros((n, n), dtype=bool)
    structural = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            lexical[i, j] = lexical[j, i] = shared_stem(relations[i], relations[j]) >= 4
            structural[i, j] = structural[j, i] = bool(
                signatures.get(relations[i], set()) & signatures.get(relations[j], set())
            )
    return lexical, structural, frequency


def build_distractor_pairs(eval_file: Path, relations: list[str]):
    index = {rel: i for i, rel in enumerate(relations)}
    record = json.loads(eval_file.read_text(encoding="utf-8"))
    by_gold: dict[str, set] = defaultdict(set)
    for question in record["recurrent_questions"]:
        if question["split"] != "train":
            continue
        gold = question["answer"]
        if gold not in index:
            continue
        for candidate in question["candidate_relations"]:
            if candidate != gold and candidate in index:
                by_gold[gold].add(candidate)
    return index, by_gold


def evaluate_space(
    spec: dict,
    embeddings: np.ndarray,
    relations: list[str],
    lexical: np.ndarray,
    structural: np.ndarray,
    distractor_index: dict[str, int],
    distractor_pairs: dict[str, set],
    pool_size: int,
    temperatures: list[float],
) -> dict:
    transformed = l2_normalize(
        apply_linear_transform(
            embeddings,
            mode=spec["mode"],
            k=spec.get("k"),
            alpha=spec.get("alpha", 1.0),
            remove=spec.get("remove", 1),
        )
    )
    n = len(relations)
    gram = transformed @ transformed.T
    off = ~np.eye(n, dtype=bool)
    related = lexical | structural

    row: dict = {
        "space": space_label(spec),
        "spec": spec,
        "geometry": geometry_summary(
            transformed,
            pool_size=pool_size,
            temperature=temperatures[0],
        ),
        "offdiag_std": float(gram[off].std()),
        "chance": {
            "lexical": float(lexical[off].mean()),
            "structural": float(structural[off].mean()),
            "related": float(related[off].mean()),
        },
        "neighbor_precision": {},
        "distractor": {},
    }
    for name, mask in (("lexical", lexical), ("structural", structural), ("related", related)):
        for k in (1, 5, 9, pool_size):
            hit = 0
            for i in range(n):
                scores = gram[i].copy()
                scores[i] = -np.inf
                hit += int(mask[i, np.argsort(-scores)[:k]].sum())
            precision = hit / (n * k)
            row["neighbor_precision"][f"{name}@{k}"] = precision
            row["neighbor_precision"][f"{name}@{k}_lift"] = precision / max(
                row["chance"][name], 1e-12
            )

    if distractor_pairs:
        percentiles: list[float] = []
        in_pool: list[float] = []
        mass = {tau: [] for tau in temperatures}
        for gold, distractors in distractor_pairs.items():
            i = distractor_index[gold]
            scores = gram[i].copy()
            scores[i] = -np.inf
            order = np.argsort(-scores)
            rank_of = {int(j): r for r, j in enumerate(order)}
            rows = [distractor_index[d] for d in distractors]
            percentiles.extend(1.0 - rank_of[j] / (n - 2) for j in rows)
            in_pool.append(sum(1 for j in rows if rank_of[j] < pool_size) / len(rows))
            top = order[:pool_size]
            position = {int(j): t for t, j in enumerate(top)}
            for tau in temperatures:
                weights = softmax(scores[top], tau)
                share = sum(weights[position[j]] for j in rows if int(j) in position)
                mass[tau].append(share / len(rows))
        row["distractor"] = {
            "n_golds": len(distractor_pairs),
            "n_pairs": int(sum(len(v) for v in distractor_pairs.values())),
            "mean_percentile": float(np.mean(percentiles)),
            "in_pool_rate": float(np.mean(in_pool)),
            "sampler_mass_on_true_distractors": {
                str(tau): float(np.mean(values)) for tau, values in mass.items()
            },
            "uniform_over_pool_mass": float(np.mean(in_pool) / pool_size),
            "uniform_over_vocab_mass": float(9.0 / (n - 1)),
        }
    return row


def main() -> None:
    args = parse_args()
    relations, embeddings = load_relation_embeddings(args.embeddings)
    lexical, structural, frequency = build_graph_ground_truth(relations, args.raw_dir)
    distractor_index, distractor_pairs = build_distractor_pairs(args.eval_file, relations)
    print(
        f"[audit] {len(relations)} 关系  训练切分金关系 {len(distractor_pairs)} 个",
        flush=True,
    )

    rows = [
        evaluate_space(
            spec,
            embeddings,
            relations,
            lexical,
            structural,
            distractor_index,
            distractor_pairs,
            args.pool_size,
            args.temperatures,
        )
        for spec in CANDIDATE_SPACES
    ]

    payload = {
        "embeddings": str(args.embeddings),
        "eval_file": str(args.eval_file),
        "raw_dir": str(args.raw_dir),
        "pool_size": int(args.pool_size),
        "temperatures": args.temperatures,
        "train_relation_order_matches_vocab": sorted(relations)
        == sorted(load_train_relation_order(args.raw_dir)),
        "top_train_relations": frequency.most_common(5),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spaces": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[audit] wrote {args.output}")

    tau = min(args.temperatures)
    header = (
        f"{'space':<24} {'cos':>7} {'std':>6} {'hub':>4} {'rel@9':>6} {'lift':>5} "
        f"{'干扰项百分位':>10} {'入pool':>7} {'命中质量':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        geometry = row["geometry"]
        distractor = row.get("distractor") or {}
        print(
            f"{row['space']:<24} {geometry['mean_offdiag_cosine']:+.4f} "
            f"{row['offdiag_std']:6.4f} {geometry['top1_hub_count']:>4} "
            f"{row['neighbor_precision']['related@9'] * 100:5.1f}% "
            f"{row['neighbor_precision']['related@9_lift']:5.2f} "
            f"{distractor.get('mean_percentile', float('nan')):>10.3f} "
            f"{distractor.get('in_pool_rate', float('nan')):>7.3f} "
            f"{distractor.get('sampler_mass_on_true_distractors', {}).get(str(tau), float('nan')):>9.4f}"
        )
    if distractor_pairs:
        uniform_vocab = rows[0]["distractor"]["uniform_over_vocab_mass"]
        uniform_pool = rows[0]["distractor"]["uniform_over_pool_mass"]
        print(
            f"\n参考: 全词表均匀采样命中质量 {uniform_vocab:.4f}; "
            f"pool 内均匀 {uniform_pool:.4f}; 干扰项随机百分位 0.500; "
            f"干扰项随机入 pool 率 {args.pool_size / (len(relations) - 1):.3f}"
        )

    if args.markdown:
        lines = [
            "| 空间 | 平均两两余弦 | 百分位 | 入 pool 率 | 命中质量 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            distractor = row.get("distractor") or {}
            lines.append(
                f"| `{row['space']}` | {row['geometry']['mean_offdiag_cosine']:+.4f} | "
                f"{distractor.get('mean_percentile', float('nan')):.3f} | "
                f"{distractor.get('in_pool_rate', float('nan')):.3f} | "
                f"{distractor.get('sampler_mass_on_true_distractors', {}).get(str(tau), float('nan')):.4f} |"
            )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[audit] wrote {args.markdown}")


if __name__ == "__main__":
    main()
