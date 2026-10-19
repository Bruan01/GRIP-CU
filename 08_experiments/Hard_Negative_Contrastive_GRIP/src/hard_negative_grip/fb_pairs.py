"""FB15K-237 entity-pair handling for offline scoring and false-negative filtering.

NELL questions name entities with single tokens (``concept_x``), so
``task_file.question_entity_pair`` splits on whitespace. FB questions use
``label, description`` text that contains spaces, commas, and sometimes
`` and ``. Here the question span is split at every `` and `` whose two halves
are both known entity texts. Known relations are keyed by entity *text* and
relation *text*, unioned over every Freebase MID sharing a text (including the
``Unknown`` fallback), so collisions filter more candidates, never fewer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .task_file import VALID_FILTER_SPLITS

FB_PAIR_SPAN_RE = re.compile(
    r"relation between (.+?)\?\s*Response in the following format",
    re.IGNORECASE | re.DOTALL,
)
NODE_PREFIX = "node "
UNKNOWN_ENTITY_TEXT = "Unknown"


def official_entity_text(value: dict) -> str:
    """Same rule as ``scripts/prepare_fb15k237.py`` (label + ', ' + description)."""
    entity = ""
    if value.get("label") is not None:
        entity += value["label"] + ", "
    if value.get("description") is not None:
        entity += value["description"]
    if not entity:
        entity = "missing"
    return entity.strip().strip(".")


def load_fb_entity_text(raw_dir: Path) -> dict[str, str]:
    """MID -> question text; train entities absent from wikidata become ``Unknown``."""
    data = json.loads((raw_dir / "entity2wikidata.json").read_text(encoding="utf-8"))
    entity_text = {mid: official_entity_text(value) for mid, value in data.items()}
    for line in (raw_dir / "train.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 3:
            for mid in (fields[0], fields[2]):
                entity_text.setdefault(mid, UNKNOWN_ENTITY_TEXT)
    return entity_text


def fb_known_pair_relations(
    raw_dir: Path,
    *,
    entity_text: dict[str, str],
    relation_text: dict[str, str],
    splits: tuple[str, ...] = VALID_FILTER_SPLITS,
) -> dict[tuple[str, str], set[str]]:
    """(head text, tail text) -> relation texts over the selected splits."""
    unknown = set(splits).difference(VALID_FILTER_SPLITS)
    if unknown:
        raise ValueError(f"unknown filter split(s): {sorted(unknown)}")
    known: dict[tuple[str, str], set[str]] = {}
    for split in splits:
        path = raw_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing FB15K-237 split: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 3:
                continue
            source, relation, target = fields
            if source not in entity_text or target not in entity_text:
                continue
            pair = (entity_text[source], entity_text[target])
            known.setdefault(pair, set()).add(relation_text[relation])
    return known


def _strip_node(value: str) -> str:
    value = value.strip()
    if value.lower().startswith(NODE_PREFIX):
        value = value[len(NODE_PREFIX):]
    return value.strip()


def fb_question_entity_pairs(text: str, entity_texts: set[str]) -> list[tuple[str, str]]:
    """Every (head, tail) split of the question span with both halves known."""
    matches = list(FB_PAIR_SPAN_RE.finditer(text))
    if not matches:
        return []
    span = matches[-1].group(1)
    pairs = []
    for match in re.finditer(r" and ", span):
        head = _strip_node(span[: match.start()])
        tail = _strip_node(span[match.end():])
        if head in entity_texts and tail in entity_texts:
            pairs.append((head, tail))
    return pairs
