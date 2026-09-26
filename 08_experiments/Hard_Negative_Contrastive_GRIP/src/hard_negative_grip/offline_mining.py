"""Production helpers for offline full-vocabulary confusion mining.

Scoring stays in ``score_candidates``. This module only handles sharding,
atomic JSONL resume, metadata, and merge. It does not change the mean-logprob
definition used by listed InfoNCE.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCORING_VERSION = "offline_confusion_v1"
COMPARE_FIELDS = (
    "candidate_relation",
    "candidate_score",
    "gold_score",
    "score_gap",
    "pairwise_confusion",
    "negative_mass",
    "candidate_rank_all",
    "negative_rank",
    "is_gold",
    "is_valid_negative",
    "invalid_reason",
    "candidate_token_length",
)
METADATA_COMPARE_KEYS = (
    "scoring_version",
    "checkpoint",
    "tokenizer",
    "dataset",
    "split",
    "relation_vocab_size",
    "temperature",
    "dtype",
)


def shard_items(items: list, *, num_shards: int, shard_id: int) -> list:
    """Keep items whose original index satisfies ``index % num_shards == shard_id``."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return [item for index, item in enumerate(items) if index % num_shards == shard_id]


def qa_index_from_id(qa_id: str) -> int:
    prefix, _, rest = str(qa_id).partition(":")
    if prefix != "task_qa" or not rest.isdigit():
        raise ValueError(f"invalid qa_id {qa_id!r}")
    return int(rest)


def complete_qa_ids_from_scores(
    path: Path,
    *,
    vocab_size: int,
) -> set[str]:
    """Return QA ids that already have a full vocabulary of candidate rows.

    A truncated last QA is dropped from the completed set so resume will redo it.
    """
    if not path.is_file() or vocab_size < 1:
        return set()
    counts: dict[str, int] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            qa_id = str(row.get("qa_id") or "")
            if not qa_id:
                continue
            if qa_id not in counts:
                counts[qa_id] = 0
                order.append(qa_id)
            counts[qa_id] += 1
    completed = {qa_id for qa_id, count in counts.items() if count == vocab_size}
    if order:
        last = order[-1]
        if counts.get(last, 0) != vocab_size:
            completed.discard(last)
    return completed


def drop_incomplete_last_qa(path: Path, *, vocab_size: int) -> str | None:
    """Rewrite ``path`` without a truncated trailing QA. Return that qa_id if any.

    A truncated JSONL last line is discarded. Incomplete candidate groups are
    rewritten atomically so resume can append a full QA block.
    """
    if not path.is_file() or vocab_size < 1:
        return None
    rows: list[dict] = []
    truncated = False
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                truncated = True
                break
    if not rows:
        if truncated:
            atomic_write_jsonl(path, [])
        return None
    last_id = str(rows[-1].get("qa_id") or "")
    last_count = sum(1 for row in rows if str(row.get("qa_id")) == last_id)
    if last_count == vocab_size:
        if truncated:
            atomic_write_jsonl(path, rows)
        return None
    kept = [row for row in rows if str(row.get("qa_id")) != last_id]
    atomic_write_jsonl(path, kept)
    return last_id or None


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_append_qa_block(
    scores_path: Path,
    summary_path: Path,
    rows: list[dict],
    summary: dict,
) -> None:
    """Append one finished QA as a single block, then fsync both files."""
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("a", encoding="utf-8") as score_stream:
        for row in rows:
            score_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        score_stream.flush()
        os.fsync(score_stream.fileno())
    with summary_path.open("a", encoding="utf-8") as summary_stream:
        summary_stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        summary_stream.flush()
        os.fsync(summary_stream.fileno())


def git_commit_hash(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref = repo / ".git" / value.split(":", 1)[1].strip()
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        return None
    return value or None


def build_metadata(
    *,
    checkpoint: str,
    tokenizer: str,
    dataset: str,
    split: str,
    relation_vocab_size: int,
    number_of_qa: int,
    shard_id: int,
    num_shards: int,
    candidate_batch_size: int,
    dtype: str,
    temperature: float,
    git_commit: str | None,
    filter_splits: list[str] | None = None,
    scoring_version: str = SCORING_VERSION,
) -> dict:
    return {
        "scoring_version": scoring_version,
        "git_commit": git_commit,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "split": split,
        "relation_vocab_size": int(relation_vocab_size),
        "number_of_qa": int(number_of_qa),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "candidate_batch_size": int(candidate_batch_size),
        "dtype": dtype,
        "temperature": float(temperature),
        "filter_splits": list(filter_splits or ("train", "valid", "test")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _metadata_signature(meta: dict) -> dict:
    return {key: meta.get(key) for key in METADATA_COMPARE_KEYS}


def merge_confusion_shards(
    shard_dirs: list[Path],
    *,
    output_dir: Path,
) -> dict:
    if not shard_dirs:
        raise ValueError("shard_dirs must be non-empty")
    metas = []
    score_rows: list[dict] = []
    summaries: list[dict] = []
    seen_ids: set[str] = set()
    shard_ids: list[int] = []
    for directory in shard_dirs:
        meta_path = directory / "metadata.json"
        scores_path = directory / "candidate_scores.jsonl"
        summary_path = directory / "qa_summary.jsonl"
        if not meta_path.is_file():
            raise ValueError(f"missing metadata.json in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metas.append(meta)
        shard_ids.append(int(meta["shard_id"]))
        rows = load_jsonl(scores_path)
        qa_rows = load_jsonl(summary_path)
        for row in qa_rows:
            qa_id = str(row.get("qa_id") or "")
            if not qa_id:
                raise ValueError(f"{summary_path}: missing qa_id")
            if qa_id in seen_ids:
                raise ValueError(f"duplicate qa_id across shards: {qa_id}")
            seen_ids.add(qa_id)
        score_rows.extend(rows)
        summaries.extend(qa_rows)

    num_shards = int(metas[0]["num_shards"])
    expected = set(range(num_shards))
    found = set(shard_ids)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra or len(found) != len(shard_ids):
        raise ValueError(f"shard coverage mismatch: missing={missing} extra={extra}")
    for meta in metas:
        if int(meta.get("num_shards", -1)) != num_shards:
            raise ValueError(
                f"shard {meta.get('shard_id')} num_shards={meta.get('num_shards')} "
                f"!= {num_shards}"
            )
    signatures = [_metadata_signature(meta) for meta in metas]
    first = signatures[0]
    for index, signature in enumerate(signatures[1:], start=1):
        if signature != first:
            raise ValueError(f"metadata mismatch between shard 0 and shard {shard_ids[index]}")

    summaries.sort(key=lambda row: qa_index_from_id(str(row["qa_id"])))
    by_qa = {str(row["qa_id"]): [] for row in summaries}
    leftover: dict[str, list[dict]] = {}
    for row in score_rows:
        qa_id = str(row.get("qa_id") or "")
        if qa_id in by_qa:
            by_qa[qa_id].append(row)
        else:
            leftover.setdefault(qa_id, []).append(row)
    if leftover:
        raise ValueError(f"candidate rows have qa_ids missing from summaries: {sorted(leftover)[:5]}")
    ordered_scores: list[dict] = []
    vocab_size = int(metas[0]["relation_vocab_size"])
    for summary in summaries:
        qa_id = str(summary["qa_id"])
        group = by_qa[qa_id]
        if len(group) != vocab_size:
            raise ValueError(f"{qa_id}: expected {vocab_size} candidate rows, found {len(group)}")
        group.sort(key=lambda row: str(row["candidate_relation"]))
        ordered_scores.extend(group)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_dir / "candidate_scores.jsonl", ordered_scores)
    atomic_write_jsonl(output_dir / "qa_summary.jsonl", summaries)
    merged_meta = dict(first)
    merged_meta.update(
        {
            "shard_id": None,
            "num_shards": num_shards,
            "number_of_qa": len(summaries),
            "source_shards": [str(path) for path in shard_dirs],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(merged_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged_meta


def compare_score_rows(
    left: list[dict],
    right: list[dict],
    *,
    atol: float = 1e-6,
) -> list[str]:
    """Compare Prompt-3 vs production rows for the same QA ids."""
    failures: list[str] = []

    def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
        indexed: dict[tuple[str, str], dict] = {}
        for row in rows:
            key = (str(row["qa_id"]), str(row["candidate_relation"]))
            if key in indexed:
                failures.append(f"duplicate row {key}")
                continue
            indexed[key] = row
        return indexed

    left_index = _index(left)
    right_index = _index(right)
    missing = sorted(set(left_index) - set(right_index))
    extra = sorted(set(right_index) - set(left_index))
    if missing:
        failures.append(f"missing {len(missing)} production rows, e.g. {missing[:3]}")
    if extra:
        failures.append(f"extra {len(extra)} production rows, e.g. {extra[:3]}")
    for key in sorted(set(left_index) & set(right_index)):
        old = left_index[key]
        new = right_index[key]
        for field in COMPARE_FIELDS:
            old_value = old.get(field)
            new_value = new.get(field)
            if isinstance(old_value, float) or isinstance(new_value, float):
                if old_value is None or new_value is None:
                    if old_value != new_value:
                        failures.append(f"{key}: {field} {old_value!r} != {new_value!r}")
                    continue
                if abs(float(old_value) - float(new_value)) > atol:
                    failures.append(
                        f"{key}: {field} {old_value!r} != {new_value!r}"
                    )
            elif old_value != new_value:
                failures.append(f"{key}: {field} {old_value!r} != {new_value!r}")
    return failures
