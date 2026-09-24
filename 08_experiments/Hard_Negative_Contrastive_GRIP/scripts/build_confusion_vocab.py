#!/usr/bin/env python3
"""Build a reusable relation-global confusion vocabulary.

CPU-only. Does not load a language model, does not train, and does not
overwrite QA-level ``analysis/`` outputs. ``--min_support`` filters figures
and optional display views; the raw pair table stays complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.confusion_analysis import (  # noqa: E402
    expand_confusion_db_row,
    iter_jsonl,
)
from hard_negative_grip.confusion_vocab import (  # noqa: E402
    aggregate_relation_pairs,
    iter_valid_negative_rows,
    write_confusion_vocab_outputs,
)
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--confusion_db", type=Path, default=None)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dump_metadata", type=Path, default=None)
    parser.add_argument("--min_support", type=int, default=10)
    parser.add_argument("--neighbor_k", type=int, default=20)
    parser.add_argument("--heatmap_top_n", type=int, default=30)
    parser.add_argument("--neighborhood_plot_n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--expected_qa", type=int, default=3253)
    return parser.parse_args()


def scores_are_complete(path: Path, *, expected_qa: int, vocab_size: int) -> bool:
    if path is None or not path.is_file():
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


def load_dump_metadata(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: metadata must be an object")
    return payload


def valid_rows_from_confusion_db(
    path: Path, relation_order: list[str], temperature: float
):
    for row in iter_jsonl(path):
        rows, _summary = expand_confusion_db_row(
            row, relation_order=relation_order, temperature=temperature
        )
        for item in rows:
            if item.get("is_valid_negative"):
                yield item


def main() -> None:
    args = parse_args()
    if args.min_support < 1:
        raise ValueError("--min_support must be >= 1")
    relation_order = load_train_relation_order(args.raw_dir)
    dump_meta = load_dump_metadata(args.dump_metadata)
    source = None
    if args.scores is not None and scores_are_complete(
        args.scores,
        expected_qa=args.expected_qa,
        vocab_size=len(relation_order),
    ):
        source = str(args.scores)
        print(f"[confusion_vocab] using candidate_scores {source}", flush=True)
        rows = iter_valid_negative_rows(args.scores)
    elif args.confusion_db is not None:
        source = str(args.confusion_db)
        print(
            f"[confusion_vocab] candidate_scores incomplete or missing; "
            f"expanding frozen confusion DB {source}",
            flush=True,
        )
        rows = valid_rows_from_confusion_db(
            args.confusion_db, relation_order, args.temperature
        )
    else:
        raise ValueError("provide a complete --scores file or --confusion_db")

    pair_payload = aggregate_relation_pairs(rows)
    metadata = {
        "checkpoint": dump_meta.get("checkpoint"),
        "dataset": dump_meta.get("dataset"),
        "split": dump_meta.get("split", "train"),
        "T": dump_meta.get("temperature", args.temperature),
        "scoring_version": dump_meta.get("scoring_version"),
        "source": source,
        "dump_metadata": str(args.dump_metadata) if args.dump_metadata else None,
    }
    payload = write_confusion_vocab_outputs(
        pair_payload=pair_payload,
        relation_order=relation_order,
        output_dir=args.output_dir,
        metadata=metadata,
        min_support=args.min_support,
        neighbor_k=args.neighbor_k,
        heatmap_top_n=args.heatmap_top_n,
        neighborhood_plot_n=args.neighborhood_plot_n,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
