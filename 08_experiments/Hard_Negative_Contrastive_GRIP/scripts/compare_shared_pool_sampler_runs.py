#!/usr/bin/env python3
"""Write a three-arm smoke comparison against frozen B1.

CPU-only. Reads listed/summary.json from each sampler run and the frozen B1
summary copied into the first variant directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ("random_k", "top_k_hard", "coverage_adaptive_k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def em_of(summary: dict) -> float:
    return float(summary["all"]["em"])


def main() -> None:
    args = parse_args()
    variants: dict[str, dict] = {}
    for variant in VARIANTS:
        variant_dir = args.run_dir / variant
        variants[variant] = {
            "run_dir": str(variant_dir),
            "summary": load_json(variant_dir / "listed" / "summary.json"),
            "comparison": (
                load_json(variant_dir / "comparison.json")
                if (variant_dir / "comparison.json").is_file()
                else None
            ),
        }
    b1_path = args.run_dir / "random_k" / "b1" / "summary.json"
    if not b1_path.is_file():
        b1_path = args.run_dir / VARIANTS[0] / "b1" / "summary.json"
    b1 = load_json(b1_path)
    random_em = em_of(variants["random_k"]["summary"])
    top_em = em_of(variants["top_k_hard"]["summary"])
    adaptive_em = em_of(variants["coverage_adaptive_k"]["summary"])
    b1_em = em_of(b1)
    payload = {
        "run_dir": str(args.run_dir),
        "eval_n": int(b1["all"]["count"]),
        "b1": b1,
        "variants": {
            name: {
                "em": em_of(item["summary"]),
                "wrong_in_list": item["summary"].get("wrong_in_list"),
                "wrong_out_of_list": item["summary"].get("wrong_out_of_list"),
                "summary": item["summary"],
                "listed_minus_b1_em": em_of(item["summary"]) - b1_em,
            }
            for name, item in variants.items()
        },
        "deltas": {
            "top_k_hard_minus_random_k": top_em - random_em,
            "coverage_adaptive_k_minus_random_k": adaptive_em - random_em,
            "coverage_adaptive_k_minus_top_k_hard": adaptive_em - top_em,
        },
        "note": (
            "Compare listed EM against frozen B1 on the same eval split. "
            "The retired 64-QA / 10-step smoke is not a training budget. "
            "A direction among Random-K / Top-K Hard / Coverage-Adaptive K "
            "needs Random-K on the full paper task file (~12014 QA, accum=512, "
            "~230 steps), not a 3253-QA matchable-only slice."
        ),
    }
    output = args.output or (args.run_dir / "comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
