#!/usr/bin/env python3
"""Rebuild confusion labels from an existing full-vocabulary score dump.

This CPU-only operation reuses candidate_score and candidate_token_length from
candidate_scores.jsonl. It never loads a model and never recomputes scores.
Only false-negative labels, negative_mass, negative_rank, and QA summaries are
recomputed for the selected KG filter splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.confusion_analysis import iter_jsonl, iter_qa_groups_from_scores  # noqa: E402
from hard_negative_grip.offline_mining import atomic_append_qa_block, load_jsonl  # noqa: E402
from hard_negative_grip.offline_scoring import (  # noqa: E402
    relabel_qa_score_group,
    validate_candidate_rows,
)
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.task_file import known_pair_relations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_metadata", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--filter_splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=["train"],
        help="KG splits used to identify true relations; defaults to train only.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    relation_order = load_train_relation_order(args.raw_dir)
    filter_splits = tuple(args.filter_splits)
    known = known_pair_relations(args.raw_dir, splits=filter_splits)
    output_dir = args.output_dir
    scores_path = output_dir / "candidate_scores.jsonl"
    summary_path = output_dir / "qa_summary.jsonl"
    metadata_path = output_dir / "metadata.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (scores_path, summary_path):
        if path.exists():
            path.unlink()

    count_qa = 0
    count_rows = 0
    if not any(iter_jsonl(args.scores)):
        raise ValueError(f"{args.scores} contains no score rows")

    for group in iter_qa_groups_from_scores(args.scores):
        rebuilt, summary = relabel_qa_score_group(
            group,
            relation_order=relation_order,
            known_relations=known.get(
                (str(group[0].get("head_entity")), str(group[0].get("tail_entity"))),
                set(),
            )
            if group[0].get("head_entity") and group[0].get("tail_entity")
            else set(),
            temperature=args.temperature,
        )
        atomic_append_qa_block(scores_path, summary_path, rebuilt, summary)
        count_qa += 1
        count_rows += len(rebuilt)

    output_rows = load_jsonl(scores_path)
    failures = validate_candidate_rows(
        output_rows,
        relation_order=relation_order,
        known_relations=known,
        temperature=args.temperature,
    )
    if failures:
        raise ValueError("relabel validation failed:\n" + "\n".join(failures[:20]))

    source_metadata = {}
    if args.source_metadata is not None and args.source_metadata.is_file():
        source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    elif (args.scores.parent / "metadata.json").is_file():
        source_metadata = json.loads((args.scores.parent / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "source_scores": str(args.scores),
        "filter_splits": list(filter_splits),
        "rescored": False,
        "relabel_only": True,
        "relation_vocab_size": len(relation_order),
        "number_of_qa": count_qa,
        "number_of_rows": count_rows,
        "temperature": float(args.temperature),
    }
    write_json(metadata_path, metadata)
    print(json.dumps({
        "output_dir": str(output_dir),
        "candidate_scores": str(scores_path),
        "qa_summary": str(summary_path),
        "metadata": str(metadata_path),
        "qa_count": count_qa,
        "row_count": count_rows,
        "filter_splits": list(filter_splits),
        "rescored": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
