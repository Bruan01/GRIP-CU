"""Join existing GRIP predictions with newly computed support-depth labels."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .io_utils import iter_jsonl

QUESTION_RE = re.compile(
    r"between word node (?P<head>\S+) and word node (?P<tail>\S+?)\?",
    flags=re.IGNORECASE,
)


def prediction_condition(row: dict) -> str:
    parts = [
        f"adapter={row.get('adapter_control', 'unknown')}",
        f"train_k={row.get('recurrent_train_k', 'na')}",
        f"eval_k={row.get('recurrence_k', row.get('eval_k', 'na'))}",
    ]
    decoder = row.get("decoder_type")
    if decoder is not None:
        parts.append(f"decoder={decoder}")
    return ";".join(parts)


def _question_entities(row: dict) -> tuple[str, str] | None:
    metadata = row.get("metadata") or {}
    head = metadata.get("head")
    tail = metadata.get("tail")
    if head and tail:
        return str(head), str(tail)
    question = row.get("question") or metadata.get("question") or ""
    match = QUESTION_RE.search(question)
    if not match:
        return None
    return match.group("head"), match.group("tail")


def analyze_prediction_files(paths: Iterable[Path], labels: Iterable[dict], *, mode: str = "undirected") -> tuple[list[dict], dict]:
    lookup: dict[tuple[str, str, str, str], dict] = {}
    pair_lookup: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for label in labels:
        split = "validation" if label["split"] == "valid" else label["split"]
        key = (split, label["head"], label["tail"], label["relation"])
        lookup[key] = label
        pair_lookup[(split, label["head"], label["tail"])].append(label)

    joined: list[dict] = []
    unmatched = 0
    ambiguous = 0
    total = 0
    for path in paths:
        for row in iter_jsonl(path):
            total += 1
            metadata = row.get("metadata") or {}
            split = str(metadata.get("split") or row.get("split") or "unknown")
            entities = _question_entities(row)
            if entities is None:
                unmatched += 1
                continue
            head, tail = entities
            targets = row.get("target") or []
            relation = targets[0] if len(targets) == 1 else None
            label = lookup.get((split, head, tail, relation)) if relation else None
            if label is None:
                candidates = pair_lookup.get((split, head, tail), [])
                if len(candidates) == 1:
                    label = candidates[0]
                elif len(candidates) > 1:
                    ambiguous += 1
                    continue
            if label is None:
                unmatched += 1
                continue
            joined.append(
                {
                    "source_file": str(path),
                    "question_id": row.get("question_id"),
                    "split": split,
                    "condition": prediction_condition(row),
                    "correct": bool(row.get("correct")),
                    "depth_mode": mode,
                    "depth_bucket": label[f"{mode}_bucket"],
                    "distance": label[f"{mode}_distance"],
                    "relation": label["relation"],
                    "relation_train_frequency": label["relation_train_frequency"],
                }
            )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in joined:
        grouped[(row["split"], row["condition"], row["depth_bucket"])].append(row)
    metrics = []
    for (split, condition, bucket), members in sorted(grouped.items()):
        correct = sum(int(row["correct"]) for row in members)
        metrics.append(
            {
                "split": split,
                "condition": condition,
                "depth_mode": mode,
                "depth_bucket": bucket,
                "n": len(members),
                "correct": correct,
                "accuracy": correct / len(members),
            }
        )
    audit = {
        "prediction_rows_total": total,
        "prediction_rows_joined": len(joined),
        "prediction_rows_unmatched": unmatched,
        "prediction_rows_ambiguous": ambiguous,
        "join_rate": len(joined) / total if total else None,
    }
    return metrics, audit
