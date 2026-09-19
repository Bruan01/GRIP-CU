"""JSONL helpers so listed-only decode can resume after a kill."""

from __future__ import annotations

import json
from pathlib import Path


def question_key(row: dict, fallback_index: int | None = None) -> str:
    qid = row.get("question_id")
    if qid:
        return str(qid)
    split = row.get("split", "")
    question = row.get("question", "")
    if split or question:
        return f"{split}:{question}"
    if fallback_index is None:
        raise ValueError("row needs question_id, question, or a fallback index")
    return f"index:{fallback_index}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_by_question_id(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for index, row in enumerate(rows):
        indexed[question_key(row, fallback_index=index)] = row
    return indexed


def eval_samples(record: dict) -> list[dict]:
    return [
        item
        for item in record["recurrent_questions"]
        if item.get("split") in {"validation", "test"}
    ]


def merge_resumed_rows(
    samples: list[dict], existing_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Keep finished rows in eval order and return samples still missing."""
    done = index_by_question_id(existing_rows)
    kept: list[dict] = []
    missing: list[dict] = []
    for index, sample in enumerate(samples):
        key = question_key(sample, fallback_index=index)
        if key in done:
            kept.append(done[key])
        else:
            missing.append(sample)
    return kept, missing


def predictions_complete(path: Path, n_eval: int) -> bool:
    if n_eval <= 0:
        return False
    return len(index_by_question_id(load_jsonl(path))) >= n_eval
