#!/usr/bin/env python3
"""Slice paper QA while keeping frozen ``task_qa:N`` identities.

Naive prefix slices would renumber questions to ``task_qa:0..K-1`` and attach
the wrong frozen-manifest negatives. This script samples matchable relation QA
from a frozen shared-pool manifest and writes ``original_question_ids``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.task_file import (  # noqa: E402
    is_grip_task_file,
    load_json_payload,
    load_score_hard_manifest,
    sample_question_ids,
    subset_task_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json_payload(args.input)
    if not is_grip_task_file(payload):
        raise ValueError(f"{args.input} is not a GRIP task file")
    manifest = load_score_hard_manifest(args.manifest)
    selected = sample_question_ids(
        list(manifest),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    output = subset_task_payload(payload, question_ids=selected)
    output["subset_source"] = str(args.input)
    output["subset_manifest"] = str(args.manifest)
    output["subset_seed"] = args.seed
    output["subset_max_samples"] = args.max_samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "manifest": str(args.manifest),
                "output": str(args.output),
                "source_qa": len(payload["qa_samples"]),
                "manifest_qa": len(manifest),
                "subset_qa": len(selected),
                "original_question_ids": selected,
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
