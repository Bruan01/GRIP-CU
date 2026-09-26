"""Load GRIP paper task files for listed-vs-B1 training.

``grip_nell23k_tasks.json`` (format_version 2) stores Qwen-chat strings:

- ``context_samples``: graph recitation + summarization (Stage 1)
- ``qa_samples``: generated context/reasoning QA (Stage 2)

Those QA prompts have no official 10-way list. Relation-like items get 9
InfoNCE negatives. The default pool is the official train-graph relation
vocabulary (198 on NELL23K), sampled with the same ``process.py`` rule as
eval 10-way. ``embed_sim`` keeps that same vocabulary but prefers
cosine-similar relations from a Stage-1 embedding table. ``qa_vocab`` keeps
the older 370-relation QA-gold pool. Evaluation still uses the aligned
relation-prediction split.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

from .embed_negatives import (
    DEFAULT_EMBED_POOL_SIZE,
    DEFAULT_EMBED_TEMPERATURE,
    RelationNeighborIndex,
)
from .official_lists import official_negatives
from .score_hard import merge_negative_sources

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RELATION_QUESTION_RE = re.compile(r"relation between", re.I)
YES_NO = {"yes", "no"}
LISTED_NEGATIVE_K = 9  # official 10-way = 1 gold + 9 distractors; independent of |A|
LISTED_NEGATIVE_SOURCES = (
    "train_graph",
    "embed_sim",
    "score_hard",
    "rollout_hard",
    "qa_vocab",
)
CONCEPT_PREFIX = "concept:"


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


def canonical_rollout_candidate(value: str) -> str:
    """Keep relation-shaped rollout output, including OOV concept labels."""
    value = str(value or "").strip()
    value = value.replace("<answer>", "").replace("</answer>", "").strip()
    if not value or value.lower() in {"yes", "no", "i don't know"}:
        return ""
    if not value.lower().startswith("concept:"):
        return ""
    if "\n" in value or "<" in value or ">" in value:
        return ""
    return value


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
    """Legacy QA-vocab sampler. Prefer ``train_graph`` + ``official_negatives``."""
    positive = normalize_relation(gold)
    pool = [item for item in vocab if item != positive]
    if not pool:
        return []
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def train_relation_alias_index(relation_order: list[str]) -> dict[str, str]:
    """Map official names and stripped ``concept:`` aliases onto train relations."""
    index: dict[str, str] = {}
    for rel in relation_order:
        index.setdefault(rel, rel)
        if rel.startswith(CONCEPT_PREFIX):
            index.setdefault(rel[len(CONCEPT_PREFIX) :], rel)
        else:
            index.setdefault(f"{CONCEPT_PREFIX}{rel}", rel)
    return index


def match_train_relation(gold: str, alias_index: dict[str, str]) -> str | None:
    return alias_index.get(normalize_relation(gold))


def load_score_hard_manifest(path: Path) -> dict[str, dict]:
    """Load and index the immutable score-hard JSONL manifest by question ID."""
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: manifest row must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id!r}")
            hard = row.get("hard_negative_relations")
            uniform = row.get("uniform_negative_relations")
            negatives = row.get("negative_relations")
            if not isinstance(hard, list) or not isinstance(uniform, list):
                raise ValueError(f"{path}:{line_number}: missing hard/uniform relation lists")
            if not isinstance(negatives, list):
                raise ValueError(f"{path}:{line_number}: missing negative_relations")
            row["question_id"] = question_id
            row["hard_negative_relations"] = [str(rel) for rel in hard]
            row["uniform_negative_relations"] = [str(rel) for rel in uniform]
            row["negative_relations"] = [str(rel) for rel in negatives]
            rows[question_id] = row
    if not rows:
        raise ValueError(f"{path} contains no manifest rows")
    return rows


VALID_FILTER_SPLITS = ("train", "valid", "test")


def known_pair_relations(
    raw_dir: Path,
    *,
    splits: tuple[str, ...] = VALID_FILTER_SPLITS,
) -> dict[tuple[str, str], set[str]]:
    """Index triples from selected dataset splits for false-negative filtering."""
    unknown = set(splits).difference(VALID_FILTER_SPLITS)
    if unknown:
        raise ValueError(
            f"unknown filter split(s): {sorted(unknown)}; "
            f"expected a subset of {VALID_FILTER_SPLITS}"
        )
    known: dict[tuple[str, str], set[str]] = {}
    for split in splits:
        filename = f"{split}.txt"
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing NELL23K split: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if len(fields) != 3:
                continue
            source, relation, target = fields
            known.setdefault((source, target), set()).add(relation)
    return known


def question_entity_pair(text: str) -> tuple[str, str] | None:
    """Extract the ordered entity pair from a generated relation question."""
    match = re.search(
        r"relation between (?:word node )?(\S+) and (?:word node )?([^?]+)\?",
        text,
        flags=re.I,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def normalize_manifest_row(
    row: dict,
    *,
    question_id: str,
    gold: str,
    relation_order: list[str],
    allow_out_of_vocab: bool = False,
) -> list[str]:
    """Validate a mined row and return its fixed decision-set negatives."""
    if str(row.get("question_id")) != question_id:
        raise ValueError(f"manifest question_id mismatch: expected {question_id!r}")
    if str(row.get("positive_relation")) != gold:
        raise ValueError(f"manifest gold mismatch for {question_id!r}")
    allowed = set(relation_order)
    hard = [str(rel) for rel in row.get("hard_negative_relations", [])]
    uniform = [str(rel) for rel in row.get("uniform_negative_relations", [])]
    negatives = [str(rel) for rel in row.get("negative_relations", [])]
    if negatives != merge_negative_sources(hard, uniform):
        raise ValueError(f"manifest negative provenance mismatch for {question_id!r}")
    if not allow_out_of_vocab and any(rel not in allowed for rel in negatives):
        raise ValueError(f"manifest contains relation outside train vocabulary for {question_id!r}")
    if any(not rel.strip() for rel in negatives):
        raise ValueError(f"manifest contains an empty negative for {question_id!r}")
    if gold in negatives or len(negatives) != len(set(negatives)):
        raise ValueError(f"manifest contains gold/duplicate negative for {question_id!r}")
    return negatives

def build_qa_assets_from_task_texts(
    qa_texts: list[str],
    *,
    seed: int,
    listed_negative_k: int = LISTED_NEGATIVE_K,
    listed_negative_source: str = "train_graph",
    relation_order: list[str] | None = None,
    relation_embeddings=None,
    embed_pool_size: int = DEFAULT_EMBED_POOL_SIZE,
    embed_sample_temperature: float = DEFAULT_EMBED_TEMPERATURE,
    score_hard_manifest: dict[str, dict] | None = None,
) -> tuple[list[str], list[dict]]:
    if not qa_texts:
        raise ValueError("task file contains no QA samples")
    if listed_negative_k < 0:
        raise ValueError(f"listed_negative_k must be non-negative, got {listed_negative_k}")
    if listed_negative_source not in LISTED_NEGATIVE_SOURCES:
        raise ValueError(
            f"listed_negative_source must be one of {LISTED_NEGATIVE_SOURCES}, "
            f"got {listed_negative_source!r}"
        )
    use_train_graph = listed_negative_source in {
        "train_graph",
        "embed_sim",
        "score_hard",
        "rollout_hard",
    }
    neighbor_index = None
    score_hard_manifest = score_hard_manifest or {}
    if listed_negative_source == "score_hard" and not score_hard_manifest:
        raise ValueError("score_hard negatives require a non-empty manifest")
    if listed_negative_source == "rollout_hard" and not score_hard_manifest:
        raise ValueError("rollout_hard negatives require a non-empty manifest")
    if use_train_graph:
        if not relation_order:
            raise ValueError("train_graph negatives require a non-empty relation_order")
        alias_index = train_relation_alias_index(relation_order)
        vocab: list[str] = list(relation_order)
        stream = np.random.RandomState(seed)
        if listed_negative_source == "embed_sim":
            if relation_embeddings is None:
                raise ValueError("embed_sim negatives require relation_embeddings")
            neighbor_index = RelationNeighborIndex(
                relation_order,
                relation_embeddings,
                pool_size=embed_pool_size,
            )
    else:
        alias_index = {}
        vocab = relation_vocab(qa_texts)
        stream = None
    texts: list[str] = []
    metas: list[dict] = []
    cosine_values: list[float] = []
    for index, text in enumerate(qa_texts):
        question_id = f"task_qa:{index}"
        gold = assistant_gold(text)
        listed: list[str] = []
        matched = match_train_relation(gold, alias_index) if use_train_graph else None
        if is_relation_gold(gold, text):
            if listed_negative_source in {"score_hard", "rollout_hard"}:
                if matched is not None:
                    row = score_hard_manifest.get(question_id)
                    if row is None:
                        raise ValueError(
                            f"{listed_negative_source} manifest has no row for {question_id!r}"
                        )
                    listed = normalize_manifest_row(
                        row,
                        question_id=question_id,
                        gold=gold,
                        relation_order=relation_order or [],
                        allow_out_of_vocab=listed_negative_source == "rollout_hard",
                    )
            elif listed_negative_source == "embed_sim":
                if matched is not None and neighbor_index is not None:
                    listed = neighbor_index.sample(
                        matched,
                        k=listed_negative_k,
                        rng=stream,
                        temperature=embed_sample_temperature,
                    )
                    cosine_values.extend(neighbor_index.sampled_cosines(matched, listed))
            elif use_train_graph:
                if matched is not None:
                    listed = official_negatives(
                        matched,
                        relation_order or [],
                        way=listed_negative_k + 1,
                        rng=stream,
                    )
            elif vocab:
                listed = sample_listed_negatives(
                    gold,
                    vocab,
                    k=listed_negative_k,
                    rng=random.Random(seed + index),
                )
        texts.append(text)
        metas.append(
            {
                "question_id": question_id,
                "positive_relation": gold,
                "listed_relations": listed,
                "prefix_text": assistant_answer_prefix(text),
                "listed_negative_source": listed_negative_source,
                "matched_train_relation": matched,
            }
        )
    if listed_negative_source in {"score_hard", "rollout_hard"}:
        expected_ids = {
            f"task_qa:{index}"
            for index, text in enumerate(qa_texts)
            if is_relation_gold(assistant_gold(text), text)
            and match_train_relation(assistant_gold(text), alias_index) is not None
        }
        missing_ids = expected_ids.difference(score_hard_manifest)
        if missing_ids:
            raise ValueError(
                f"{listed_negative_source} manifest is missing {len(missing_ids)} question rows"
            )
    if listed_negative_source == "embed_sim" and cosine_values:
        print(
            f"[data] embed_sim pool={embed_pool_size} tau={embed_sample_temperature} "
            f"sampled_negatives={len(cosine_values)} mean_cosine={float(np.mean(cosine_values)):.4f}",
            flush=True,
        )
    return texts, metas
