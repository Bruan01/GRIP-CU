from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.decode_io import (  # noqa: E402
    append_jsonl,
    index_by_question_id,
    load_jsonl,
    merge_resumed_rows,
    predictions_complete,
    question_key,
)


def test_load_jsonl_skips_truncated_last_line(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"question_id": "nell23k:validation:1", "correct": True})
        + "\n{truncated",
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["question_id"] == "nell23k:validation:1"


def test_merge_resumed_rows_keeps_eval_order(tmp_path: Path) -> None:
    samples = [
        {"question_id": "nell23k:validation:1", "split": "validation"},
        {"question_id": "nell23k:test:2", "split": "test"},
        {"question_id": "nell23k:test:3", "split": "test"},
    ]
    existing = [
        {"question_id": "nell23k:test:3", "response": "later"},
        {"question_id": "nell23k:validation:1", "response": "first"},
    ]
    kept, missing = merge_resumed_rows(samples, existing)
    assert [row["question_id"] for row in kept] == [
        "nell23k:validation:1",
        "nell23k:test:3",
    ]
    assert [row["question_id"] for row in missing] == ["nell23k:test:2"]
    append_jsonl(tmp_path / "out.jsonl", {"question_id": "nell23k:test:2"})
    assert predictions_complete(tmp_path / "out.jsonl", 1) is True
    assert question_key(samples[1]) == "nell23k:test:2"
    assert set(index_by_question_id(existing)) == {
        "nell23k:test:3",
        "nell23k:validation:1",
    }
