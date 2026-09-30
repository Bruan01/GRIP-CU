#!/usr/bin/env python3
"""Write the Prompt-6 five-arm comparison against frozen B1.

CPU-only. Reads listed summaries, optional closed-set files, and adapter
metadata. Missing mixture runs stay omitted until those adapters exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
CONTROL_VARIANTS = ("random_k", "top_k_hard", "coverage_adaptive_k")
MIXTURE_VARIANTS = ("soft_mix", "calibrated")
VARIANTS = CONTROL_VARIANTS + MIXTURE_VARIANTS
DEFAULT_RUNS = {
    "random_k": HNG / "results" / "runs" / "20260926_shared_pool_random_k_full",
    "top_k_hard": HNG / "results" / "runs" / "20260926_shared_pool_top_k_hard_full",
    "coverage_adaptive_k": HNG
    / "results"
    / "runs"
    / "20260926_shared_pool_coverage_adaptive_k_full",
    "soft_mix": HNG / "results" / "runs" / "20260929_shared_pool_soft_mix_full",
    "calibrated": HNG / "results" / "runs" / "20260929_shared_pool_calibrated_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    for variant in VARIANTS:
        parser.add_argument(f"--{variant}_dir", type=Path, default=None)
    parser.add_argument(
        "--require_all",
        action="store_true",
        help="Fail if any of the five listed runs is missing summary.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def em_of(summary: dict) -> float:
    return float(summary["all"]["em"])


# Longest-first NELL domain prefixes. Concatenated relation names such as
# athleteplaysforteam / athletehomestadium share "athlete", not the whole string.
_NELL_TYPE_PREFIXES = (
    "agriculturalproduct",
    "awardtrophytournament",
    "automobilemaker",
    "touristattraction",
    "televisionstation",
    "organization",
    "politicianus",
    "sportsgame",
    "academicfield",
    "academicprogram",
    "buildingfeature",
    "musicartist",
    "musicgenre",
    "radiostation",
    "stateorprovince",
    "visualartist",
    "arthropod",
    "automaker",
    "beverage",
    "bodypart",
    "building",
    "chemical",
    "clothing",
    "company",
    "country",
    "director",
    "invertebrate",
    "journalist",
    "language",
    "location",
    "musician",
    "newspaper",
    "politician",
    "profession",
    "stadium",
    "athlete",
    "airport",
    "animal",
    "artery",
    "bacteria",
    "bakedgood",
    "bank",
    "book",
    "city",
    "coach",
    "drug",
    "emotion",
    "father",
    "fish",
    "food",
    "furniture",
    "hotel",
    "item",
    "lake",
    "league",
    "mammal",
    "mother",
    "mountain",
    "museum",
    "object",
    "parent",
    "park",
    "person",
    "plant",
    "proxy",
    "river",
    "sport",
    "state",
    "team",
    "thing",
    "weapon",
    "agent",
    "actor",
)


def relation_family(name: str) -> str:
    local = str(name).split(":")[-1].lower()
    for prefix in _NELL_TYPE_PREFIXES:
        if local.startswith(prefix):
            return prefix
    match = re.match(r"[a-z]+", local)
    return match.group(0) if match else local


def family_confusion(path: Path) -> dict | None:
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None
    wrong = [row for row in rows if not row.get("correct")]
    in_list = [row for row in wrong if row.get("in_list")]
    same_family = 0
    for row in in_list:
        gold = row.get("target")
        gold_name = gold[0] if isinstance(gold, list) and gold else gold
        pred = row.get("response")
        if gold_name and pred and relation_family(str(gold_name)) == relation_family(str(pred)):
            same_family += 1
    return {
        "n": len(rows),
        "wrong": len(wrong),
        "wrong_in_list": len(in_list),
        "wrong_out_of_list": len(wrong) - len(in_list),
        "in_list_same_family": same_family,
        "in_list_same_family_rate": (same_family / len(in_list)) if in_list else 0.0,
    }


def resolve_variant_dir(args: argparse.Namespace, variant: str) -> Path:
    override = getattr(args, f"{variant}_dir")
    if override is not None:
        return override
    if args.run_dir is not None:
        return args.run_dir / variant
    return DEFAULT_RUNS[variant]


def load_variant(path: Path) -> dict:
    listed = path / "listed"
    summary = load_json(listed / "summary.json")
    closed = load_optional_json(listed / "summary_closed_set.json")
    metadata = load_optional_json(listed / "adapter" / "run_metadata.json")
    wiring = load_optional_json(path / "wiring.json")
    comparison = load_optional_json(path / "comparison.json")
    closed_comparison = load_optional_json(path / "comparison_closed_set.json")
    variant_key = next(iter((wiring or {}).get("variants") or {}), None)
    wiring_stats = ((wiring or {}).get("variants") or {}).get(variant_key or "", {})
    return {
        "run_dir": str(path),
        "summary": summary,
        "closed_set": closed,
        "comparison": comparison,
        "comparison_closed_set": closed_comparison,
        "run_metadata": metadata,
        "wiring": wiring_stats or None,
        "family_confusion": family_confusion(listed / "predictions_correct.jsonl"),
        "em": em_of(summary),
        "wrong_in_list": summary.get("wrong_in_list"),
        "wrong_out_of_list": summary.get("wrong_out_of_list"),
        "candidate_valid_rate": summary.get("candidate_valid_rate"),
        "generation_loss": None if metadata is None else metadata.get("last_generation_loss"),
        "candidate_loss": None if metadata is None else metadata.get("last_candidate_loss"),
        "last_lambda_candidate": None
        if metadata is None
        else metadata.get("last_lambda_candidate"),
        "candidate_forwards": None if metadata is None else metadata.get("candidate_forwards"),
        "seconds": None if metadata is None else metadata.get("seconds"),
        "mean_k": None if not wiring_stats else wiring_stats.get("k_mean"),
        "lambda_min": None if not wiring_stats else wiring_stats.get("lambda_min"),
        "lambda_max": None if not wiring_stats else wiring_stats.get("lambda_max"),
        "closed_em": None if closed is None else em_of(closed),
        "closed_mrr": None if closed is None else closed.get("mrr"),
        "closed_hits_at_1": None if closed is None else closed.get("hits@1"),
    }


def numeric_delta(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def calibrated_gate(variants: dict[str, dict]) -> dict | None:
    if "calibrated" not in variants or "random_k" not in variants:
        return None
    calibrated = variants["calibrated"]
    random_k = variants["random_k"]
    em_up = calibrated["em"] > random_k["em"]
    gen_loss_ok = True
    if calibrated["generation_loss"] is not None and random_k["generation_loss"] is not None:
        gen_loss_ok = float(calibrated["generation_loss"]) <= 1.1 * float(random_k["generation_loss"])
    ranking_up = True
    if calibrated["closed_mrr"] is not None and random_k["closed_mrr"] is not None:
        ranking_up = float(calibrated["closed_mrr"]) >= float(random_k["closed_mrr"])
    if calibrated["closed_hits_at_1"] is not None and random_k["closed_hits_at_1"] is not None:
        ranking_up = ranking_up and float(calibrated["closed_hits_at_1"]) >= float(
            random_k["closed_hits_at_1"]
        )
    ool_ok = True
    if calibrated["wrong_out_of_list"] is not None and random_k["wrong_out_of_list"] is not None:
        ool_ok = int(calibrated["wrong_out_of_list"]) <= int(random_k["wrong_out_of_list"])
    passed = bool(em_up and gen_loss_ok and ranking_up and ool_ok)
    return {
        "em_above_random_k": em_up,
        "generation_loss_not_up_10pct": gen_loss_ok,
        "closed_ranking_not_down": ranking_up,
        "out_of_list_not_up": ool_ok,
        "continue": passed,
        "note": (
            "Continue only if Calibrated EM > Random-K, generation loss is not "
            "materially higher, closed-set MRR/Hits@1 is not down, and "
            "out-of-list errors do not increase. Smoke 96 is not a paper claim."
        ),
    }


def main() -> None:
    args = parse_args()
    variants: dict[str, dict] = {}
    missing: list[str] = []
    for variant in VARIANTS:
        path = resolve_variant_dir(args, variant)
        summary_path = path / "listed" / "summary.json"
        if not summary_path.is_file():
            missing.append(variant)
            if args.require_all:
                raise FileNotFoundError(summary_path)
            continue
        variants[variant] = load_variant(path)

    b1_path = None
    for variant in VARIANTS:
        candidate = resolve_variant_dir(args, variant) / "b1" / "summary.json"
        if candidate.is_file():
            b1_path = candidate
            break
    if b1_path is None:
        raise FileNotFoundError("frozen B1 summary.json not found in any variant run")
    b1 = load_json(b1_path)
    b1_closed = load_optional_json(b1_path.with_name("summary_closed_set.json"))
    b1_em = em_of(b1)

    payload_variants = {}
    for name, item in variants.items():
        payload_variants[name] = {
            **item,
            "listed_minus_b1_em": item["em"] - b1_em,
            "listed_minus_b1_closed_em": numeric_delta(item["closed_em"], None if b1_closed is None else em_of(b1_closed)),
        }

    deltas = {}
    if "random_k" in variants:
        random_em = variants["random_k"]["em"]
        for name in VARIANTS:
            if name == "random_k" or name not in variants:
                continue
            deltas[f"{name}_minus_random_k"] = variants[name]["em"] - random_em
            deltas[f"{name}_minus_random_k_closed_em"] = numeric_delta(
                variants[name]["closed_em"],
                variants["random_k"]["closed_em"],
            )
            deltas[f"{name}_minus_random_k_generation_loss"] = numeric_delta(
                variants[name]["generation_loss"],
                variants["random_k"]["generation_loss"],
            )
    if "top_k_hard" in variants and "coverage_adaptive_k" in variants:
        deltas["coverage_adaptive_k_minus_top_k_hard"] = (
            variants["coverage_adaptive_k"]["em"] - variants["top_k_hard"]["em"]
        )
    if "soft_mix" in variants and "calibrated" in variants:
        deltas["calibrated_minus_soft_mix"] = (
            variants["calibrated"]["em"] - variants["soft_mix"]["em"]
        )

    output = args.output
    if output is None:
        if args.run_dir is not None:
            output = args.run_dir / "comparison.json"
        else:
            output = HNG / "results" / "runs" / "20260929_shared_pool_mixture_samplers" / "ablation_comparison.json"

    payload = {
        "run_dir": None if args.run_dir is None else str(args.run_dir),
        "eval_n": int(b1["all"]["count"]),
        "b1": b1,
        "b1_closed_set": b1_closed,
        "present_variants": list(variants),
        "missing_variants": missing,
        "variants": payload_variants,
        "deltas": deltas,
        "calibrated_gate": calibrated_gate(variants),
        "note": (
            "Five-arm listed comparison on the same eval split: B1, Random-K, "
            "Top-K Hard, Coverage-Adaptive K, Soft-Mix, Calibrated. "
            "Closed-set ranking is inference-only and does not count as a "
            "training method. The retired 64-QA / 10-step smoke is not a "
            "training budget. A paper claim still needs the official NELL23K "
            "test split, not this 96-question smoke."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
