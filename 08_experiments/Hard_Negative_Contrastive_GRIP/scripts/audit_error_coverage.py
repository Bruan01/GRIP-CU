"""Measure whether negative families land in the model's actual decision set.

quick01 errors split into:

- in-list: the generated string is one of the prompt's 10-way distractors;
- offlist in-vocab: a real NELL relation that was not listed;
- true OOV: a string outside the relation vocabulary.

``listed`` only covers in-list errors when it is built from the *same* 10-way
the model saw. Official-list alignment is eval hygiene; it does not retrofit
coverage onto an already-run generation dump that used RecurrentGRIP lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hard_negative_grip.candidates import parse_listed_relations  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_error(pred: str, listed: list[str], vocab: set[str]) -> str:
    if pred in listed:
        return "in_list"
    if pred in vocab:
        return "offlist_invocab"
    return "true_oov"


def family_relations(row: dict) -> dict[str, set[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for candidate in row.get("hard_candidates", []) + row.get("random_candidates", []):
        families[candidate["kind"]].add(candidate["relation"])
    return families


def coverage_table(hits: dict[str, dict[str, int]], totals: dict[str, int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for family, by_type in hits.items():
        row = {}
        for error_type, total in totals.items():
            hit = by_type.get(error_type, 0)
            row[error_type] = {
                "hit": hit,
                "total": total,
                "rate": (hit / total) if total else None,
            }
        out[family] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    vocab: set[str] = set()
    if args.prepared is not None:
        prepared = load_jsonl(args.prepared)[0]
        vocab = {edge[1] for edge in prepared["graph"]["edge_list"]}
        prepared_questions = {
            sample["question_id"]: sample for sample in prepared["recurrent_questions"]
        }
    else:
        prepared_questions = {}

    audit_rows = {}
    if args.audit is not None:
        payload = json.loads(args.audit.read_text(encoding="utf-8"))
        rows = payload[0]["rows"] if isinstance(payload, list) else payload
        audit_rows = {row["question_id"]: row for row in rows}

    totals = {"in_list": 0, "offlist_invocab": 0, "true_oov": 0, "correct": 0}
    overlap_totals = {"in_list": 0, "offlist_invocab": 0, "true_oov": 0, "correct": 0}
    eval_list_hits = defaultdict(int)
    official_list_hits = defaultdict(int)
    family_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overlap_n = 0
    examples = {"in_list": [], "offlist_invocab": [], "true_oov": []}

    for row in predictions:
        gold = row["target"][0] if isinstance(row["target"], list) else row["target"]
        pred = row["response"]
        eval_listed = parse_listed_relations(row["question"])
        error_type = "correct" if row.get("correct") else classify_error(pred, eval_listed, vocab)
        totals[error_type] += 1
        if error_type == "correct":
            continue
        if pred in eval_listed and pred != gold:
            eval_list_hits[error_type] += 1
        prepared_sample = prepared_questions.get(row["question_id"])
        if prepared_sample is not None:
            official_listed = prepared_sample.get("candidate_relations") or parse_listed_relations(
                str(prepared_sample.get("question", ""))
            )
            if pred in official_listed and pred != gold:
                official_list_hits[error_type] += 1
        audit_row = audit_rows.get(row["question_id"])
        if audit_row is not None:
            overlap_n += 1
            overlap_totals[error_type] += 1
            if error_type != "correct":
                for family, relations in family_relations(audit_row).items():
                    if pred in relations:
                        family_hits[family][error_type] += 1
        if len(examples[error_type]) < 8:
            examples[error_type].append(
                {
                    "question_id": row["question_id"],
                    "gold": gold,
                    "prediction": pred,
                    "eval_listed": eval_listed,
                }
            )

    wrong = totals["in_list"] + totals["offlist_invocab"] + totals["true_oov"]
    result = {
        "predictions": str(args.predictions),
        "audit": str(args.audit) if args.audit else None,
        "prepared": str(args.prepared) if args.prepared else None,
        "n_predictions": len(predictions),
        "n_overlap_with_audit": overlap_n,
        "totals": totals,
        "wrong": wrong,
        "eval_list_covers_wrong_prediction": {
            error_type: {
                "hit": eval_list_hits[error_type],
                "total": totals[error_type],
                "rate": (eval_list_hits[error_type] / totals[error_type])
                if totals[error_type]
                else None,
            }
            for error_type in ("in_list", "offlist_invocab", "true_oov")
        },
        "aligned_official_list_covers_wrong_prediction": {
            error_type: {
                "hit": official_list_hits[error_type],
                "total": totals[error_type],
                "rate": (official_list_hits[error_type] / totals[error_type])
                if totals[error_type]
                else None,
            }
            for error_type in ("in_list", "offlist_invocab", "true_oov")
        },
        "overlap_totals": overlap_totals,
        "constructed_family_coverage_on_audit_overlap": coverage_table(
            family_hits,
            {k: v for k, v in overlap_totals.items() if k != "correct"},
        )
        if audit_rows
        else None,
        "note": (
            "eval_list_covers in_list is 100% by construction: those errors *are* "
            "the prompt distractors. aligned_official_list_covers is low unless the "
            "generation dump used the official 10-way. true_oov cannot be retrieved "
            "by an a-priori string generator."
        ),
        "examples": examples,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_predictions", "wrong", "totals")}, ensure_ascii=False))
    print("eval-list cover", result["eval_list_covers_wrong_prediction"])
    print("official-list cover", result["aligned_official_list_covers_wrong_prediction"])
    if result["constructed_family_coverage_on_audit_overlap"]:
        print("family cover", result["constructed_family_coverage_on_audit_overlap"])


if __name__ == "__main__":
    main()
