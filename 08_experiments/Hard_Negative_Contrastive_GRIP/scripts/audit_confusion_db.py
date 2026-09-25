#!/usr/bin/env python3
"""CPU-only quality audit of the production relation-confusion dump.

Does not load a 7B teacher and does not train. Optional ``--live_rescore``
ingests a previously written 20-QA GPU comparison JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.confusion_audit import (  # noqa: E402
    audit_dump,
    write_audit_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--qa_summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frozen_db", type=Path, default=None)
    parser.add_argument("--compare_scores", type=Path, default=None)
    parser.add_argument("--live_rescore", type=Path, default=None)
    parser.add_argument("--listed_adapter", type=str, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copies = []
    if args.report is not None:
        copies.append(args.report)
    result = audit_dump(
        scores_path=args.scores,
        summary_path=args.qa_summary,
        metadata_path=args.metadata,
        task_file=args.task_file,
        raw_dir=args.raw_dir,
        frozen_db_path=args.frozen_db,
        compare_scores_path=args.compare_scores,
        live_rescore_path=args.live_rescore,
        listed_adapter=args.listed_adapter,
        seed=args.seed,
        temperature=args.temperature,
    )
    written = write_audit_outputs(result, args.output_dir, markdown_copies=copies)
    ids_path = args.output_dir / "sampled_scorer_qa_ids.json"
    ids_path.write_text(
        json.dumps(result.get("sampled_scorer_qa_ids") or [], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "final_status": result["final_status"],
        "blocking_issues": result["blocking_issues"],
        "issues": result["issues"],
        "n_qa": result["dataset"]["n_scored_qa"],
        "n_rows": result["dataset"]["n_candidate_rows"],
        "sampled_scorer_qa_ids": str(ids_path),
        **written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if result["final_status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
