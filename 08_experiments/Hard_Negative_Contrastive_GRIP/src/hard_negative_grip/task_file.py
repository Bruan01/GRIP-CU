"""Load standard GRIP task files for candidate-level InfoNCE training.

The training task file stores the original GRIP context samples and QA chat
strings. Standard processed graph records store ``questions`` and
``answers``; they are used for generation evaluation and never require a
recursive executor or a recursive question schema.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

from .official_lists import official_negatives

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RELATION_QUESTION_RE = re.compile(r"relation between", re.I)
YES_NO = {"yes", "no"}
LISTED_NEGATIVE_K = 9  # official 10-way = 1 gold + 9 distractors; independent of |A|
LISTED_NEGATIVE_SOURCES = (
    "train_graph",
    "error_memory",
)
CONCEPT_PREFIX = "concept:"


def merge_negative_sources(hard: list[str], uniform: list[str]) -> list[str]:
    """Keep a deterministic, duplicate-free negative list."""
    merged: list[str] = []
    seen: set[str] = set()
    for relation in [*hard, *uniform]:
        value = str(relation)
        if value and value not in seen:
            seen.add(value)
            merged.append(value)
    return merged


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
    if isinstance(payload, dict) and {"questions", "answers", "graph"}.issubset(payload):
        if len(payload["questions"]) != len(payload["answers"]):
            raise ValueError(f"{path} has mismatched questions and answers")
        return payload
    if isinstance(payload, list) and payload:
        record = payload[0]
        if (
            not isinstance(record, dict)
            or not {"questions", "answers", "graph"}.issubset(record)
        ):
            raise ValueError(f"{path} is not a standard GRIP graph record")
        if len(record["questions"]) != len(record["answers"]):
            raise ValueError(f"{path} has mismatched questions and answers")
        return record
    raise ValueError(f"{path} is not a standard GRIP graph record")


def assistant_gold(text: str) -> str:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        raise ValueError("QA sample has no <answer> span")
    return matches[-1].group(1).strip()


def parse_model_answer(text: str) -> str:
    """Parse a standard GRIP ``<answer>`` response, with a plain-text fallback."""
    matches = list(ANSWER_RE.finditer(str(text)))
    if matches:
        return matches[-1].group(1).strip()
    return str(text).strip()


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


def is_relation_gold(
    gold: str, text: str, *, relation_alias_index: dict[str, str] | None = None
) -> bool:
    label = normalize_relation(gold)
    if not label or label.lower() in YES_NO:
        return False
    if label.startswith("concept:"):
        return True
    # FB15K-237 relation descriptions contain spaces. Require both a relation
    # question and membership in the train vocabulary to distinguish entities.
    if RELATION_QUESTION_RE.search(text) and relation_alias_index and label in relation_alias_index:
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


def task_qa_id(index: int) -> str:
    if int(index) != index or index < 0:
        raise ValueError(f"QA index must be a non-negative integer, got {index!r}")
    return f"task_qa:{int(index)}"


def parse_task_qa_id(qa_id: str) -> int:
    prefix, _, rest = str(qa_id).partition(":")
    if prefix != "task_qa" or not rest.isdigit():
        raise ValueError(f"invalid qa_id {qa_id!r}")
    return int(rest)


def original_question_ids_from_payload(payload: dict, n_texts: int) -> list[str] | None:
    """Keep frozen-manifest IDs when a task file is a subset of paper QA."""
    raw_ids = payload.get("original_question_ids")
    raw_indices = payload.get("original_qa_indices")
    if raw_ids is None and raw_indices is None:
        return None
    if raw_ids is not None:
        ids = [str(item) for item in raw_ids]
        if len(ids) != n_texts:
            raise ValueError(
                f"original_question_ids has {len(ids)} entries, expected {n_texts}"
            )
        for question_id in ids:
            parse_task_qa_id(question_id)
        if raw_indices is not None:
            derived = [task_qa_id(int(index)) for index in raw_indices]
            if derived != ids:
                raise ValueError("original_question_ids do not match original_qa_indices")
        return ids
    ids = [task_qa_id(int(index)) for index in raw_indices]
    if len(ids) != n_texts:
        raise ValueError(
            f"original_qa_indices has {len(ids)} entries, expected {n_texts}"
        )
    return ids


def sample_question_ids(
    question_ids: list[str],
    *,
    max_samples: int,
    seed: int,
) -> list[str]:
    """Deterministically sample frozen-manifest IDs; keep original index order."""
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")
    unique: list[str] = []
    seen: set[str] = set()
    for question_id in question_ids:
        parse_task_qa_id(question_id)
        if question_id in seen:
            raise ValueError(f"duplicate question_id {question_id!r}")
        seen.add(question_id)
        unique.append(str(question_id))
    if max_samples == 0 or max_samples >= len(unique):
        return sorted(unique, key=parse_task_qa_id)
    chosen = random.Random(seed).sample(unique, max_samples)
    return sorted(chosen, key=parse_task_qa_id)


def subset_task_payload(payload: dict, *, question_ids: list[str]) -> dict:
    """Keep selected paper QA rows without renumbering ``task_qa:N`` identities."""
    if not is_grip_task_file(payload):
        raise ValueError("payload is not a GRIP task file")
    qa_samples = list(payload["qa_samples"])
    indices = [parse_task_qa_id(question_id) for question_id in question_ids]
    if len(indices) != len(set(indices)):
        raise ValueError("question_ids contain duplicates")
    for index, question_id in zip(indices, question_ids):
        if index >= len(qa_samples):
            raise ValueError(
                f"{question_id} is outside the task file ({len(qa_samples)} QA samples)"
            )
    output = dict(payload)
    output["qa_samples"] = [qa_samples[index] for index in indices]
    output["original_qa_indices"] = indices
    output["original_question_ids"] = [task_qa_id(index) for index in indices]
    output["subset_n_qa"] = len(indices)
    return output


def load_error_memory_manifest(path: Path) -> dict[str, dict]:
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
            if "lambda_candidate" in row and row["lambda_candidate"] is not None:
                row["lambda_candidate"] = float(row["lambda_candidate"])
                if row["lambda_candidate"] < 0:
                    raise ValueError(
                        f"{path}:{line_number}: lambda_candidate must be non-negative"
                    )
            rows[question_id] = row
    if not rows:
        raise ValueError(f"{path} contains no manifest rows")
    return rows


def validate_error_memory_manifest(rows: dict[str, dict], *, k: int = LISTED_NEGATIVE_K) -> None:
    """Reject a non-frozen or non-train-only error-memory bank."""
    if k != LISTED_NEGATIVE_K or not rows:
        raise ValueError("error-memory sampling requires K=9 and a non-empty bank")
    for question_id, row in rows.items():
        negatives = row["negative_relations"]
        selected = row.get("selected_negatives")
        if (
            row.get("negative_source") != "train_only_error_memory_score_violation_uniform_floor"
            or row.get("filter_splits") != ["train"]
            or row.get("k") != len(negatives)
            or row.get("actual_k") != len(negatives)
            or row.get("requested_k") != k
            or len(negatives) > k
            or not isinstance(selected, list)
            or len(selected) != len(negatives)
            or [item.get("relation") for item in selected] != negatives
            or row.get("hard_negative_relations") != negatives
            or row.get("uniform_negative_relations") != []
            or row.get("selection_rule") is None
            or not 0.0 <= float(row.get("uniform_mix", -1.0)) <= 1.0
            or float(row.get("error_weight", -1.0)) < 0
            or float(row.get("violation_weight", -1.0)) < 0
            or float(row.get("sampling_temperature", 0.0)) <= 0
        ):
            raise ValueError(
                f"{question_id}: error-memory manifest provenance/K/parameter mismatch"
            )


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
    if "lambda_candidate" in row and row["lambda_candidate"] is not None:
        weight = float(row["lambda_candidate"])
        if weight < 0:
            raise ValueError(f"manifest lambda_candidate is negative for {question_id!r}")
        row["lambda_candidate"] = weight
    return negatives

def build_qa_assets_from_task_texts(
    qa_texts: list[str],
    *,
    seed: int,
    listed_negative_k: int = LISTED_NEGATIVE_K,
    listed_negative_source: str = "train_graph",
    relation_order: list[str] | None = None,
    error_memory_manifest: dict[str, dict] | None = None,
    question_ids: list[str] | None = None,
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
    error_memory_manifest = error_memory_manifest or {}
    if listed_negative_source == "error_memory" and not error_memory_manifest:
        raise ValueError("error_memory negatives require a non-empty manifest")
    if not relation_order:
        raise ValueError("candidate sampling requires a non-empty relation_order")
    alias_index = train_relation_alias_index(relation_order)
    stream = np.random.RandomState(seed)
    if question_ids is not None and len(question_ids) != len(qa_texts):
        raise ValueError(
            f"question_ids has {len(question_ids)} entries, expected {len(qa_texts)}"
        )
    resolved_ids: list[str]
    if question_ids is None:
        resolved_ids = [task_qa_id(index) for index in range(len(qa_texts))]
    else:
        resolved_ids = []
        for question_id in question_ids:
            parse_task_qa_id(question_id)
            resolved_ids.append(str(question_id))
    texts: list[str] = []
    metas: list[dict] = []
    for index, text in enumerate(qa_texts):
        question_id = resolved_ids[index]
        gold = assistant_gold(text)
        listed: list[str] = []
        lambda_candidate = None
        matched = match_train_relation(gold, alias_index)
        if is_relation_gold(gold, text, relation_alias_index=alias_index):
            if listed_negative_source == "error_memory":
                if matched is not None:
                    row = error_memory_manifest.get(question_id)
                    if row is None:
                        raise ValueError(
                            f"{listed_negative_source} manifest has no row for {question_id!r}"
                        )
                    listed = normalize_manifest_row(
                        row,
                        question_id=question_id,
                        gold=gold,
                        relation_order=relation_order or [],
                    )
                    if row.get("lambda_candidate") is not None:
                        lambda_candidate = float(row["lambda_candidate"])
            else:
                if matched is not None:
                    listed = official_negatives(
                        matched,
                        relation_order or [],
                        way=listed_negative_k + 1,
                        rng=stream,
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
                "lambda_candidate": lambda_candidate,
            }
        )
    if listed_negative_source == "error_memory":
        expected_ids = {
            resolved_ids[index]
            for index, text in enumerate(qa_texts)
            if is_relation_gold(assistant_gold(text), text, relation_alias_index=alias_index)
            and match_train_relation(assistant_gold(text), alias_index) is not None
        }
        missing_ids = expected_ids.difference(error_memory_manifest)
        if missing_ids:
            raise ValueError(
                f"error_memory manifest is missing {len(missing_ids)} question rows"
            )
    return texts, metas
