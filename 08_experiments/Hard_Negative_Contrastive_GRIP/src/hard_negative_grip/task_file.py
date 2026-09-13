"""Load GRIP paper task files for listed-vs-B1 training.

``grip_nell23k_tasks.json`` (format_version 2) stores Qwen-chat strings:

- ``context_samples``: graph recitation + summarization (Stage 1)
- ``qa_samples``: generated context/reasoning QA (Stage 2)

Those QA prompts have no official 10-way list. Relation-like items get 9
in-vocab negatives sampled from the task file itself so the listed fork can
run InfoNCE. Evaluation still uses the aligned relation-prediction split.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RELATION_QUESTION_RE = re.compile(r"relation between", re.I)
YES_NO = {"yes", "no"}
LISTED_NEGATIVE_K = 9


def is_grip_task_file(payload: object) -> bool:
    return isinstance(payload, dict) and "context_samples" in payload and "qa_samples" in payload


def load_json_payload(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) == 1 and isinstance(rows[0], list):
        return rows[0]
    return rows


def load_graph_record(path: Path) -> dict:
    payload = load_json_payload(path)
    if isinstance(payload, dict) and "recurrent_questions" in payload:
        return payload
    if isinstance(payload, list) and payload:
        record = payload[0]
        if not isinstance(record, dict) or "recurrent_questions" not in record:
            raise ValueError(f"{path} is not a graph record with recurrent_questions")
        return record
    raise ValueError(f"{path} is not an aligned NELL23K graph record")


def assistant_gold(text: str) -> str:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        raise ValueError("QA sample has no <answer> span")
    return matches[-1].group(1).strip()


def assistant_answer_prefix(text: str) -> str:
    idx = text.rfind("<answer>")
    if idx < 0:
        raise ValueError("QA sample has no <answer> tag")
    return text[: idx + len("<answer>")]


def normalize_relation(label: str) -> str:
    return label.strip().rstrip(".")


def is_relation_gold(gold: str, text: str) -> bool:
    label = normalize_relation(gold)
    if not label or label.lower() in YES_NO:
        return False
    if label.startswith("concept:"):
        return True
    if RELATION_QUESTION_RE.search(text) and " " not in label and not label.startswith("concept_"):
        return True
    return False


def relation_vocab(qa_texts: list[str]) -> list[str]:
    vocab: set[str] = set()
    for text in qa_texts:
        gold = assistant_gold(text)
        if is_relation_gold(gold, text):
            vocab.add(normalize_relation(gold))
    return sorted(vocab)


def sample_listed_negatives(
    gold: str,
    vocab: list[str],
    *,
    k: int = LISTED_NEGATIVE_K,
    rng: random.Random,
) -> list[str]:
    positive = normalize_relation(gold)
    pool = [item for item in vocab if item != positive]
    if not pool:
        return []
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def build_qa_assets_from_task_texts(
    qa_texts: list[str],
    *,
    seed: int,
    listed_negative_k: int = LISTED_NEGATIVE_K,
) -> tuple[list[str], list[dict]]:
    if not qa_texts:
        raise ValueError("task file contains no QA samples")
    vocab = relation_vocab(qa_texts)
    texts: list[str] = []
    metas: list[dict] = []
    for index, text in enumerate(qa_texts):
        gold = assistant_gold(text)
        listed: list[str] = []
        if is_relation_gold(gold, text) and vocab:
            listed = sample_listed_negatives(
                gold,
                vocab,
                k=listed_negative_k,
                rng=random.Random(seed + index),
            )
        texts.append(text)
        metas.append(
            {
                "question_id": f"task_qa:{index}",
                "positive_relation": gold,
                "listed_relations": listed,
                "prefix_text": assistant_answer_prefix(text),
            }
        )
    return texts, metas
