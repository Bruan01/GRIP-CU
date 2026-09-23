from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from types import SimpleNamespace

import torch

from hard_negative_grip.offline_mining import (  # noqa: E402
    atomic_append_qa_block,
    compare_score_rows,
    complete_qa_ids_from_scores,
    drop_incomplete_last_qa,
    merge_confusion_shards,
    shard_items,
)
from hard_negative_grip.offline_scoring import build_qa_score_rows  # noqa: E402
from hard_negative_grip.scoring import score_candidates  # noqa: E402


class _Tok:
    pad_token_id = 0
    unk_token_id = 1

    def __call__(self, text, add_special_tokens=False, truncation=True, padding=False):
        ids = [(ord(char) % 30) + 2 for char in text] or [self.unk_token_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _LM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(2026)
        self.embed = torch.nn.Embedding(32, 8)
        self.lm_head = torch.nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def _rows(qa_id: str, n: int = 2) -> list[dict]:
    return [
        {"qa_id": qa_id, "candidate_relation": f"r{index}", "candidate_score": float(index)}
        for index in range(n)
    ]


def test_shard_items_are_deterministic_and_cover_all() -> None:
    items = [f"task_qa:{index}" for index in range(10)]
    zero = shard_items(items, num_shards=3, shard_id=0)
    one = shard_items(items, num_shards=3, shard_id=1)
    two = shard_items(items, num_shards=3, shard_id=2)
    assert zero == ["task_qa:0", "task_qa:3", "task_qa:6", "task_qa:9"]
    assert one == ["task_qa:1", "task_qa:4", "task_qa:7"]
    assert two == ["task_qa:2", "task_qa:5", "task_qa:8"]
    assert sorted(zero + one + two) == items
    assert shard_items(items, num_shards=3, shard_id=0) == zero


def test_resume_ignores_truncated_trailing_qa(tmp_path: Path) -> None:
    scores = tmp_path / "candidate_scores.jsonl"
    scores.write_text(
        json.dumps(_rows("task_qa:0", 2)[0])
        + "\n"
        + json.dumps(_rows("task_qa:0", 2)[1])
        + "\n"
        + json.dumps(_rows("task_qa:1", 2)[0])
        + "\n",
        encoding="utf-8",
    )
    completed = complete_qa_ids_from_scores(scores, vocab_size=2)
    assert completed == {"task_qa:0"}
    dropped = drop_incomplete_last_qa(scores, vocab_size=2)
    assert dropped == "task_qa:1"
    remaining = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    assert {row["qa_id"] for row in remaining} == {"task_qa:0"}
    assert complete_qa_ids_from_scores(scores, vocab_size=2) == {"task_qa:0"}


def test_atomic_append_then_merge_rejects_duplicates(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    shard0.mkdir()
    shard1.mkdir()
    meta = {
        "scoring_version": "offline_confusion_v1",
        "checkpoint": "b1",
        "tokenizer": "qwen",
        "dataset": "tasks.json",
        "split": "train",
        "relation_vocab_size": 2,
        "temperature": 1.0,
        "dtype": "bfloat16",
        "num_shards": 2,
        "number_of_qa": 1,
        "candidate_batch_size": 8,
    }
    (shard0 / "metadata.json").write_text(json.dumps({**meta, "shard_id": 0}), encoding="utf-8")
    (shard1 / "metadata.json").write_text(json.dumps({**meta, "shard_id": 1}), encoding="utf-8")
    atomic_append_qa_block(
        shard0 / "candidate_scores.jsonl",
        shard0 / "qa_summary.jsonl",
        _rows("task_qa:0", 2),
        {"qa_id": "task_qa:0"},
    )
    atomic_append_qa_block(
        shard1 / "candidate_scores.jsonl",
        shard1 / "qa_summary.jsonl",
        _rows("task_qa:1", 2),
        {"qa_id": "task_qa:1"},
    )
    merged = merge_confusion_shards([shard0, shard1], output_dir=tmp_path / "merged")
    assert merged["number_of_qa"] == 2
    summaries = [json.loads(line) for line in (tmp_path / "merged" / "qa_summary.jsonl").read_text().splitlines()]
    assert [row["qa_id"] for row in summaries] == ["task_qa:0", "task_qa:1"]
    try:
        atomic_append_qa_block(
            shard1 / "candidate_scores.jsonl",
            shard1 / "qa_summary.jsonl",
            _rows("task_qa:0", 2),
            {"qa_id": "task_qa:0"},
        )
        merge_confusion_shards([shard0, shard1], output_dir=tmp_path / "dup")
    except ValueError as error:
        assert "duplicate qa_id" in str(error)
    else:
        raise AssertionError("duplicate qa_id should fail")


def test_merge_requires_every_shard(tmp_path: Path) -> None:
    shard0 = tmp_path / "only0"
    shard0.mkdir()
    meta = {
        "scoring_version": "offline_confusion_v1",
        "checkpoint": "b1",
        "tokenizer": "qwen",
        "dataset": "tasks.json",
        "split": "train",
        "relation_vocab_size": 2,
        "temperature": 1.0,
        "dtype": "bfloat16",
        "num_shards": 2,
        "shard_id": 0,
        "number_of_qa": 1,
        "candidate_batch_size": 8,
    }
    (shard0 / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    atomic_append_qa_block(
        shard0 / "candidate_scores.jsonl",
        shard0 / "qa_summary.jsonl",
        _rows("task_qa:0", 2),
        {"qa_id": "task_qa:0"},
    )
    try:
        merge_confusion_shards([shard0], output_dir=tmp_path / "merged")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing shard should fail")


def test_compare_score_rows_detects_drift() -> None:
    left, _ = build_qa_score_rows(
        qa_id="task_qa:0",
        question="q",
        head_entity="a",
        tail_entity="b",
        gold_relation="concept:gold",
        matched_train_relation="concept:gold",
        relation_order=["concept:gold", "concept:hard"],
        scores={"concept:gold": -0.8, "concept:hard": -0.9},
        token_lengths={"concept:gold": 1, "concept:hard": 2},
        known_relations=[],
        temperature=1.0,
        model_checkpoint="b1",
    )
    right = [dict(row) for row in left]
    assert compare_score_rows(left, right) == []
    right[1]["candidate_score"] = -0.5
    failures = compare_score_rows(left, right)
    assert any("candidate_score" in item for item in failures)


def test_batched_vocab_scoring_matches_single_candidate() -> None:
    tokenizer = _Tok()
    model = _LM()
    device = torch.device("cpu")
    prefix = "Q<answer>"
    relations = ["r", "owns", "concept:worksfor", "concept:atdate"]
    batched = score_candidates(model, tokenizer, prefix, relations, device=device)
    singles = [
        score_candidates(model, tokenizer, prefix, [rel], device=device)["candidate_score"][0][0]
        for rel in relations
    ]
    for left, right in zip(batched["candidate_score"][0], singles):
        assert abs(left - right) < 1e-6
    assert batched["candidate_token_length"][0] == [
        score_candidates(model, tokenizer, prefix, [rel], device=device)["candidate_token_length"][0][0]
        for rel in relations
    ]


def test_resume_rewrites_truncated_jsonl_last_line(tmp_path: Path) -> None:
    scores = tmp_path / "candidate_scores.jsonl"
    complete = _rows("task_qa:0", 2)
    scores.write_text(
        json.dumps(complete[0])
        + "\n"
        + json.dumps(complete[1])
        + "\n"
        + '{"qa_id": "task_qa:1", "candidate_relation": "r0"'
        + "\n",
        encoding="utf-8",
    )
    assert complete_qa_ids_from_scores(scores, vocab_size=2) == {"task_qa:0"}
    dropped = drop_incomplete_last_qa(scores, vocab_size=2)
    assert dropped is None
    remaining = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    assert remaining == complete
    for line in scores.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_merge_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    shard0.mkdir()
    shard1.mkdir()
    meta = {
        "scoring_version": "offline_confusion_v1",
        "checkpoint": "b1",
        "tokenizer": "qwen",
        "dataset": "tasks.json",
        "split": "train",
        "relation_vocab_size": 2,
        "temperature": 1.0,
        "dtype": "bfloat16",
        "num_shards": 2,
        "number_of_qa": 1,
        "candidate_batch_size": 8,
    }
    (shard0 / "metadata.json").write_text(json.dumps({**meta, "shard_id": 0}), encoding="utf-8")
    (shard1 / "metadata.json").write_text(
        json.dumps({**meta, "shard_id": 1, "temperature": 0.5}),
        encoding="utf-8",
    )
    atomic_append_qa_block(
        shard0 / "candidate_scores.jsonl",
        shard0 / "qa_summary.jsonl",
        _rows("task_qa:0", 2),
        {"qa_id": "task_qa:0"},
    )
    atomic_append_qa_block(
        shard1 / "candidate_scores.jsonl",
        shard1 / "qa_summary.jsonl",
        _rows("task_qa:1", 2),
        {"qa_id": "task_qa:1"},
    )
    try:
        merge_confusion_shards([shard0, shard1], output_dir=tmp_path / "merged")
    except ValueError as error:
        assert "metadata mismatch" in str(error)
    else:
        raise AssertionError("inconsistent metadata should fail")
