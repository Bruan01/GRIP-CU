"""Rewrite a relation-embedding table through a linear transform.

Stage-1 relation-name embeddings live in a narrow cone (mean pairwise cosine
~0.88 on the 198 train relations), so the cosine softmax in
``--listed_negative_source embed_sim`` is nearly uniform inside the top-40 pool
and its ranking is mostly noise. Center / all-but-the-top / partial PCA-whitening
are linear maps that remove that cone and restore a usable similarity scale.

Writes the transformed table plus ``<stem>_whiten.json`` with the geometry before
and after, the ranking preview and the effective-negative count, so a training run
can be audited from the npz alone. Nothing here trains or needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.embed_negatives import (  # noqa: E402
    DEFAULT_EMBED_POOL_SIZE,
    DEFAULT_EMBED_TEMPERATURE,
    DEFAULT_WHITEN_ALPHA,
    DEFAULT_WHITEN_MODE,
    TRANSFORM_MODES,
    apply_linear_transform,
    geometry_summary,
    l2_normalize,
    load_relation_embeddings,
    save_relation_embeddings,
    top_neighbors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HNG / "results/relation_embeddings/s1_qwen7b_20260913.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input stem>_whiten_<mode>_k<k>_a<alpha>.npz next to the input.",
    )
    parser.add_argument("--mode", choices=TRANSFORM_MODES, default=DEFAULT_WHITEN_MODE)
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="pca_whiten: keep this many principal directions. 0 = full rank.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_WHITEN_ALPHA,
        help="pca_whiten scaling exponent. 0 = plain PCA, 1 = full whitening.",
    )
    parser.add_argument(
        "--remove",
        type=int,
        default=1,
        help="allbuttop: number of leading principal directions to drop.",
    )
    parser.add_argument("--pool_size", type=int, default=DEFAULT_EMBED_POOL_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_EMBED_TEMPERATURE)
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--preview_relations", type=int, default=5)
    return parser.parse_args()


def default_output(args: argparse.Namespace) -> Path:
    """Name the artifact after the transform so a run_config points at the recipe."""
    # str surgery: Path.with_suffix would eat the ".5" of a float alpha.
    if args.mode == "pca_whiten":
        tag = f"{args.mode}_k{args.k or 'full'}_a{args.alpha:g}"
    elif args.mode == "allbuttop":
        tag = f"{args.mode}_m{args.remove}"
    else:
        tag = args.mode
    return args.input.with_name(f"{args.input.stem}_{tag}.npz")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"missing input embedding table: {args.input}")
    output = args.output or default_output(args)
    relations, embeddings = load_relation_embeddings(args.input)
    k = args.k or None
    transformed = apply_linear_transform(
        embeddings, mode=args.mode, k=k, alpha=args.alpha, remove=args.remove
    )
    normalized = l2_normalize(transformed)

    before = geometry_summary(
        embeddings, pool_size=args.pool_size, temperature=args.temperature
    )
    after = geometry_summary(
        normalized, pool_size=args.pool_size, temperature=args.temperature
    )
    preview = {
        rel: top_neighbors(rel, relations, normalized, k=args.neighbor_k)
        for rel in relations[: min(args.preview_relations, len(relations))]
    }
    metadata = {
        "source": str(args.input),
        "transform": {
            "mode": args.mode,
            "k": k,
            "alpha": args.alpha if args.mode == "pca_whiten" else None,
            "remove": args.remove if args.mode == "allbuttop" else None,
        },
        "pooling": "mean_nonspecial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "why": (
            "Stage-1 relation embeddings share a dominant direction (mean pairwise "
            "cosine ~0.88), which flattens the embed_sim softmax. This linear map "
            "removes it and restores a usable similarity scale."
        ),
    }
    save_relation_embeddings(output, relations, normalized, metadata)

    report = {
        "output": str(output),
        "source": str(args.input),
        "transform": metadata["transform"],
        "geometry_before": before,
        "geometry_after": after,
        "pool_size": int(args.pool_size),
        "temperature": float(args.temperature),
        "neighbors_preview": preview,
        "metadata": metadata,
    }
    report_path = output.with_name(f"{output.stem}_whiten.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[whiten] {args.mode} k={k} alpha={args.alpha} remove={args.remove}")
    print(
        f"[whiten] 平均两两余弦 {before['mean_offdiag_cosine']:+.4f} -> "
        f"{after['mean_offdiag_cosine']:+.4f}  "
        f"(标准差 {before['std_offdiag_cosine']:.4f} -> {after['std_offdiag_cosine']:.4f})"
    )
    print(
        f"[whiten] top-1 hub {before['top1_hub_count']} -> {after['top1_hub_count']} 个关系共用"
    )
    print(
        f"[whiten] tau={args.temperature} 有效负样本 {before['mean_effective_negatives']:.1f}"
        f" -> {after['mean_effective_negatives']:.1f} /{args.pool_size}"
    )
    print(f"[whiten] wrote {output}")
    print(f"[whiten] wrote {report_path}")
    if abs(after["mean_effective_negatives"] - args.pool_size) < 1e-6:
        print(
            "[whiten] WARNING: effective negatives == pool size, the softmax is "
            "uniform; this transform erased the similarity ranking.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
