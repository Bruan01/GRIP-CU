#!/usr/bin/env python3
"""Import an aligned historical closed-set candidate-rerank run.

The imported source is kept as historical evidence, not presented as a new
model run.  The converter verifies every question and target against the
current official processed test record before writing the current evaluator's
JSONL schema and a provenance-rich summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(answer: str) -> str:
    import re
    import string

    text = answer.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    words = {"man": "men", "woman": "women"}
    return " ".join(words.get(word, word) for word in text.split())


def exact_match(response: object, targets: list[object]) -> bool:
    value = normalize_answer(str(response))
    return any(value == normalize_answer(str(target)) for target in targets)


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--source_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_label", default="historical_i0_candidate_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("the official processed test file must contain one graph record")
    questions = payload.get("questions") or []
    answers = payload.get("answers") or []
    if len(questions) != len(answers):
        raise ValueError("questions and answers have different lengths")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    temporary_predictions = prediction_path.with_name(
        f".{prediction_path.name}.{os.getpid()}.tmp"
    )
    rows: list[dict[str, Any]] = []
    with args.source_predictions.open("r", encoding="utf-8") as stream, temporary_predictions.open(
        "w", encoding="utf-8"
    ) as output:
        for index, line in enumerate(stream):
            source = json.loads(line)
            if index >= len(questions):
                raise ValueError("historical source has more rows than the current test set")
            question = questions[index][0] if isinstance(questions[index], (list, tuple)) else str(questions[index])
            targets = answers[index] if isinstance(answers[index], list) else [answers[index]]
            if str(source.get("question")) != str(question):
                raise ValueError(f"question mismatch at index {index}")
            source_targets = source.get("target")
            source_targets = source_targets if isinstance(source_targets, list) else [source_targets]
            if [normalize_answer(str(item)) for item in source_targets] != [
                normalize_answer(str(item)) for item in targets
            ]:
                raise ValueError(f"target mismatch at index {index}")

            scored_candidates: list[dict[str, Any]] = []
            for item in source.get("candidate_scores") or []:
                candidate = str(item["candidate"])
                score = item.get("score", item.get("lm_score"))
                if score is None:
                    raise ValueError(f"missing candidate score at index {index}")
                scored_candidates.append({"candidate": candidate, "score": float(score)})
            if not scored_candidates:
                raise ValueError(f"no candidates at index {index}")
            ranked = sorted(
                scored_candidates, key=lambda item: item["score"], reverse=True
            )
            response = str(source.get("response", ""))
            gold_rank = next(
                (
                    rank
                    for rank, item in enumerate(ranked, start=1)
                    if exact_match(item["candidate"], targets)
                ),
                None,
            )
            row = {
                "dataset_index": index,
                "question_id": f"{payload.get('id', 'graph')}:test:{index}",
                "split": "test",
                "question": str(question),
                "target": [str(item) for item in targets],
                "response": response,
                "correct": exact_match(response, targets),
                "candidate_valid": any(
                    normalize_answer(response) == normalize_answer(item["candidate"])
                    for item in scored_candidates
                ),
                "candidate_scores": scored_candidates,
                "top_candidates": ranked[:5],
                "gold_rank": gold_rank,
                "score_margin": (
                    float(ranked[0]["score"] - ranked[1]["score"])
                    if len(ranked) > 1
                    else None
                ),
                "historical_source_id": source.get("id"),
                "historical_decoding_mode": source.get("decoding_mode"),
                "historical_relation_card_mode": source.get("relation_card_mode"),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    if len(rows) != len(questions):
        raise ValueError(
            f"historical source has {len(rows)} rows but current test set has {len(questions)}"
        )
    temporary_predictions.replace(prediction_path)

    input_sha256 = sha256_file(args.input_file)
    source_sha256 = sha256_file(args.source_predictions)
    correct = [bool(row["correct"]) for row in rows]
    valid = [bool(row["candidate_valid"]) for row in rows]
    summary = {
        "status": "completed",
        "count": len(rows),
        "requested_count": len(rows),
        "em": sum(correct) / len(correct) if correct else 0.0,
        "candidate_valid_rate": sum(valid) / len(valid) if valid else 0.0,
        "reference_count": 0,
        "reference_em": None,
        "rerank_minus_reference_em": None,
        "rerank_gains": 0,
        "rerank_regressions": 0,
        "input_file": str(args.input_file),
        "source_predictions": str(args.source_predictions),
        "source_label": args.source_label,
        "provenance": "historical_import_aligned_and_verified; not rerun in current session",
        "updated_at_utc": now_utc(),
    }
    metadata = {
        "format_version": 1,
        "method": "official_candidate_teacher_forced_rerank",
        "model_mode": args.source_label,
        "created_at_utc": now_utc(),
        "input_file": str(args.input_file),
        "input_sha256": input_sha256,
        "source_predictions": str(args.source_predictions),
        "source_predictions_sha256": source_sha256,
        "candidate_protocol": "historical official candidate-list score; current test questions and targets verified row by row",
        "reproducibility": "source weights are not present in the current workspace; this artifact is historical evidence, not a fresh rerun",
        "total_questions": len(questions),
    }
    atomic_write(output_dir / "summary.json", summary)
    atomic_write(output_dir / "metadata.json", metadata)
    (output_dir / "import.log").write_text(
        f"imported_at_utc={now_utc()}\n"
        f"source={args.source_predictions}\n"
        f"source_sha256={source_sha256}\n"
        f"input_sha256={input_sha256}\n"
        f"rows={len(rows)}\n"
        f"em={summary['em']:.8f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
