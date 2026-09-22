"""Split a GRIP task file's QA samples into train and held-out mining sets.

The context samples remain unchanged. QA samples are assigned to A/B with a
fixed seed; B is used only for frozen-teacher error mining and is excluded
from Stage-2 training in the held-out experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-holdout-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("qa_samples"), list):
        raise ValueError(f"{path} is not a GRIP task file")
    if "context_samples" not in payload:
        raise ValueError(f"{path} has no context_samples")
    return payload


def write(path: Path, payload: dict, split: str, indices: list[int], source: Path, args):
    output = dict(payload)
    output["qa_samples"] = [payload["qa_samples"][index] for index in indices]
    output["heldout_split"] = split
    output["heldout_source"] = str(source)
    output["heldout_seed"] = args.seed
    output["heldout_fraction"] = args.holdout_fraction
    output["heldout_original_indices"] = indices
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    payload = load(args.input)
    count = len(payload["qa_samples"])
    rng = random.Random(args.seed)
    holdout_count = max(1, round(count * args.holdout_fraction))
    holdout_set = set(rng.sample(range(count), holdout_count))
    train_indices = [index for index in range(count) if index not in holdout_set]
    holdout_indices = [index for index in range(count) if index in holdout_set]
    if args.max_train_samples < 0 or args.max_holdout_samples < 0:
        raise ValueError("max sample limits must be non-negative")
    if args.max_train_samples:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_holdout_samples:
        holdout_indices = holdout_indices[: args.max_holdout_samples]
    write(args.train_output, payload, "train", train_indices, args.input, args)
    write(args.holdout_output, payload, "holdout", holdout_indices, args.input, args)
    print(
        json.dumps(
            {
                "qa_samples": count,
                "train_samples": len(train_indices),
                "holdout_samples": len(holdout_indices),
                "seed": args.seed,
                "holdout_fraction": args.holdout_fraction,
                "train_output": str(args.train_output),
                "holdout_output": str(args.holdout_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
