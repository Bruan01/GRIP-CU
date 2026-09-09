"""Rewrite RecurrentGRIP val/test 10-way lists to the official GRIP protocol.

Train questions keep the RecurrentGRIP lists because the paper processor does
not emit train QA. Validation and test questions are rewritten in place using
``numpy.random`` after ``seed=2026``, matching
``process_raw_data.py --datasets nell23k --seed 2026``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hard_negative_grip.official_lists import (  # noqa: E402
    build_official_nell23k_lists,
    official_candidates_for_question,
    parse_listed_relations,
    rewrite_question_with_official_list,
)

DEFAULT_RAW = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29/grip-exp/data/raw_datasets/nell23k"
)
DEFAULT_PREPARED = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29/grip-exp/outputs/data/nell23k/"
    "recurrent_relation_prediction.json"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def align_record(record: dict, official_lists: dict[str, list[list[str]]]) -> tuple[dict, dict]:
    stats = {
        "train_kept": 0,
        "eval_rewritten": 0,
        "eval_missing": 0,
        "identical_10way": 0,
        "jaccard_sum": 0.0,
        "by_split": {},
    }
    questions = []
    for sample in record.get("recurrent_questions", []):
        split = str(sample.get("split", ""))
        if split == "train":
            updated = dict(sample)
            updated.setdefault("candidate_source", "recurrentgrip")
            questions.append(updated)
            stats["train_kept"] += 1
            continue
        official = official_candidates_for_question(sample, official_lists)
        split_stats = stats["by_split"].setdefault(
            split, {"rewritten": 0, "identical": 0, "missing": 0, "jaccard_sum": 0.0}
        )
        if official is None:
            questions.append(sample)
            stats["eval_missing"] += 1
            split_stats["missing"] += 1
            continue
        previous = sample.get("candidate_relations") or parse_listed_relations(
            str(sample.get("question", ""))
        )
        updated = rewrite_question_with_official_list(sample, official)
        questions.append(updated)
        stats["eval_rewritten"] += 1
        split_stats["rewritten"] += 1
        same = list(previous) == list(official)
        if same:
            stats["identical_10way"] += 1
            split_stats["identical"] += 1
        overlap = jaccard(previous, official)
        stats["jaccard_sum"] += overlap
        split_stats["jaccard_sum"] += overlap
    updated_record = dict(record)
    updated_record["recurrent_questions"] = questions
    metadata = dict(record.get("dataset_metadata") or {})
    metadata["candidate_source"] = "official_grip_nell23k_seed2026"
    metadata["train_candidate_source"] = "recurrentgrip"
    updated_record["dataset_metadata"] = metadata
    rewritten = stats["eval_rewritten"]
    stats["mean_jaccard"] = (stats["jaccard_sum"] / rewritten) if rewritten else None
    for split_stats in stats["by_split"].values():
        count = split_stats["rewritten"]
        split_stats["mean_jaccard"] = (
            split_stats["jaccard_sum"] / count if count else None
        )
        del split_stats["jaccard_sum"]
    del stats["jaccard_sum"]
    return updated_record, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--report_file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    official_lists = build_official_nell23k_lists(args.raw_dir, seed=args.seed)
    records = load_jsonl(args.input_file)
    aligned = []
    reports = []
    for record in records:
        updated, stats = align_record(record, official_lists)
        aligned.append(updated)
        reports.append({"graph_id": record.get("id", record.get("title")), **stats})
    save_jsonl(args.output_file, aligned)
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "protocol": "process_raw_data.py --datasets nell23k --seed 2026",
                    "raw_dir": str(args.raw_dir),
                    "input_file": str(args.input_file),
                    "output_file": str(args.output_file),
                    "records": reports,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {"output_file": str(args.output_file), "records": reports},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
