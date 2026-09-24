#!/usr/bin/env python3
"""Describe GRIP relation-vocabulary confusion from saved scores.

CPU-only. Does not load a language model and does not train. Prefers a complete
Prompt-4 ``candidate_scores.jsonl``; otherwise expands the frozen confusion DB
into the same valid-negative schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.confusion_analysis import (  # noqa: E402
    analyze_score_groups,
    iter_jsonl,
    iter_qa_groups,
    write_analysis_outputs,
)
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Prompt-4 candidate_scores.jsonl. Used only if every QA has a full vocab.",
    )
    parser.add_argument(
        "--confusion_db",
        type=Path,
        default=None,
        help="Frozen confusion_db.jsonl used when --scores is incomplete.",
    )
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--expected_qa",
        type=int,
        default=3253,
        help="Complete dump size. Scores files with fewer QA fall back to confusion_db.",
    )
    return parser.parse_args()


def scores_are_complete(path: Path, *, expected_qa: int, vocab_size: int) -> bool:
    if not path.is_file():
        return False
    counts: dict[str, int] = {}
    for row in iter_jsonl(path):
        qa_id = str(row.get("qa_id") or "")
        if not qa_id:
            return False
        counts[qa_id] = counts.get(qa_id, 0) + 1
    if len(counts) != expected_qa:
        return False
    return all(count == vocab_size for count in counts.values())


def main() -> None:
    args = parse_args()
    relation_order = load_train_relation_order(args.raw_dir)
    source = None
    groups = None
    if args.scores is not None and scores_are_complete(
        args.scores,
        expected_qa=args.expected_qa,
        vocab_size=len(relation_order),
    ):
        source = str(args.scores)
        groups = iter_qa_groups(scores_path=args.scores)
        print(f"[confusion_analysis] using candidate_scores {source}", flush=True)
    elif args.confusion_db is not None:
        source = str(args.confusion_db)
        groups = iter_qa_groups(
            confusion_db_path=args.confusion_db,
            relation_order=relation_order,
            temperature=args.temperature,
        )
        print(
            f"[confusion_analysis] candidate_scores incomplete or missing; "
            f"expanding frozen confusion DB {source}",
            flush=True,
        )
    else:
        raise ValueError("provide a complete --scores file or --confusion_db")

    result = analyze_score_groups(groups)
    payload = write_analysis_outputs(result, args.output_dir, source=source)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
