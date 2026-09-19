"""Encode train-graph relation vocabulary A with the Stage-1 adapter.

Writes an L2-normalized embedding table used by ``--listed_negative_source embed_sim``.
Does not train. Loads the 7B + S1 adapter, mean-pools each relation string, then exits
so listed training can take the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))
sys.path.insert(0, str(HNG / "scripts"))

from hard_negative_grip.embed_negatives import (  # noqa: E402
    l2_normalize,
    mean_offdiag_cosine,
    save_relation_embeddings,
    top_neighbors,
)
from hard_negative_grip.official_lists import (  # noqa: E402
    DEFAULT_RAW_NELL23K,
    load_train_relation_order,
)
from train_listed_contrastive import load_adapter, release_cuda  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s1_adapter",
        type=Path,
        default=HNG
        / "results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/s1_adapter",
    )
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_NELL23K)
    parser.add_argument(
        "--output",
        type=Path,
        default=HNG / "results/relation_embeddings/s1_qwen7b_20260913.npz",
    )
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", default=str(GRIP_EXP / "model_cache"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["down_proj", "up_proj", "gate_proj"],
    )
    return parser.parse_args()


def mean_pool(hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    weights = token_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_relations(model, tokenizer, relations: list[str], batch_size: int) -> torch.Tensor:
    model.eval()
    special_ids = set(tokenizer.all_special_ids or [])
    pieces: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for start in range(0, len(relations), batch_size):
        batch = relations[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        token_mask = encoded["attention_mask"].bool()
        if special_ids:
            special_mask = torch.ones_like(token_mask)
            for token_id in special_ids:
                special_mask &= encoded["input_ids"] != token_id
            token_mask = torch.where(special_mask.any(dim=1, keepdim=True), special_mask, token_mask)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                states = getattr(out, "hidden_states", None)
                if not states:
                    raise RuntimeError("model output has neither last_hidden_state nor hidden_states")
                hidden = states[-1]
            pooled = mean_pool(hidden, token_mask)
        pieces.append(pooled.float().cpu())
        print(
            f"[embed] {min(start + len(batch), len(relations))}/{len(relations)}",
            flush=True,
        )
    return torch.cat(pieces, dim=0)


def main() -> None:
    args = parse_args()
    if not args.s1_adapter.is_dir():
        raise FileNotFoundError(f"missing Stage-1 adapter: {args.s1_adapter}")
    relations = load_train_relation_order(args.raw_dir)
    print(
        f"[embed] encoding {len(relations)} train-graph relations with {args.s1_adapter}",
        flush=True,
    )
    model, tokenizer = load_adapter(args, args.s1_adapter, trainable=False)
    vectors = encode_relations(model, tokenizer, relations, args.batch_size)
    del model
    del tokenizer
    release_cuda()
    embeddings = l2_normalize(vectors.numpy())
    metadata = {
        "s1_adapter": str(args.s1_adapter),
        "raw_dir": str(args.raw_dir),
        "model_name": args.model_name,
        "pooling": "mean_nonspecial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_relations": len(relations),
        "dim": int(embeddings.shape[1]),
    }
    save_relation_embeddings(args.output, relations, embeddings, metadata)
    neighbors = {
        rel: top_neighbors(rel, relations, embeddings, k=args.neighbor_k)
        for rel in relations[: min(12, len(relations))]
    }
    audit = {
        "output": str(args.output),
        "n_relations": len(relations),
        "mean_offdiag_cosine": mean_offdiag_cosine(embeddings),
        "neighbors_preview": neighbors,
        "metadata": metadata,
    }
    audit_path = args.output.with_name(args.output.stem + "_neighbors.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[embed] wrote {args.output}", flush=True)
    print(f"[embed] wrote {audit_path}", flush=True)
    print(f"[embed] mean off-diagonal cosine={audit['mean_offdiag_cosine']:.4f}", flush=True)


if __name__ == "__main__":
    main()
