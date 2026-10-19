"""Extract complete rankings for selected offline-scoring QA records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--qa-id", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(args.qa_id)
    grouped = defaultdict(list)
    with Path(args.candidate_scores).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["qa_id"] in wanted:
                grouped[row["qa_id"]].append(row)

    missing = sorted(wanted - set(grouped))
    if missing:
        raise ValueError(f"QA ids not found: {missing}")

    output = []
    for qa_id in args.qa_id:
        rows = sorted(grouped[qa_id], key=lambda row: row["candidate_rank_all"])
        output.append({
            "qa_id": qa_id,
            "question": rows[0]["question"],
            "gold_relation": rows[0]["gold_relation"],
            "gold_score": rows[0]["gold_score"],
            "gold_rank": next(row["candidate_rank_all"] for row in rows if row["is_gold"]),
            "all_true_relations": rows[0]["all_true_relations"],
            "ranking": [
                {
                    "rank": row["candidate_rank_all"],
                    "candidate_relation": row["candidate_relation"],
                    "candidate_score": row["candidate_score"],
                    "score_gap": row["score_gap"],
                    "is_gold": row["is_gold"],
                    "is_valid_negative": row["is_valid_negative"],
                    "invalid_reason": row["invalid_reason"],
                    "pairwise_confusion": row["pairwise_confusion"],
                    "negative_mass": row["negative_mass"],
                    "negative_rank": row["negative_rank"],
                    "candidate_token_length": row["candidate_token_length"],
                }
                for row in rows
            ],
        })

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_file": str(output_path.resolve()),
        "qa_count": len(output),
        "candidate_count_per_qa": [len(item["ranking"]) for item in output],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
