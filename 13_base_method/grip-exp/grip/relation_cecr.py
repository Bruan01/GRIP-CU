"""Candidate--Evidence--Counterfactual Reasoning (CECR) utilities.

This module contains the small, dependency-light pieces used by the CECR
inference MVP.  It deliberately keeps the relation memory separate from the
model so that the memory can be built from the training graph only and audited
without loading a language model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_CANDIDATE_RE = re.compile(
    r"Selected\s+from\s+the\s+following\s+candidate\s+answers\s*:\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"word\s+node\s+([^\s?]+)", re.IGNORECASE)

# These are the high-value confusion groups found in the original GRIP 7B
# errors.  The groups are intentionally symmetric: a card for r lists the
# relations which should be explicitly contrasted with r.
CONFUSION_SETS: tuple[tuple[str, ...], ...] = (
    (
        "concept:subpartof",
        "concept:locationlocatedwithinlocation",
        "concept:proxyfor",
        "concept:atlocation",
        "concept:agentactsinlocation",
        "concept:citylocatedinstate",
        "concept:citylocatedincountry",
    ),
    (
        "concept:proxyfor",
        "concept:mutualproxyfor",
        "concept:atlocation",
        "concept:agentactsinlocation",
        "concept:citylocatedinstate",
        "concept:citylocatedincountry",
    ),
    (
        "concept:animalsuchasinsect",
        "concept:animalsuchasinvertebrate",
        "concept:arthropodandotherarthropod",
        "concept:animalthatfeedoninsect",
        "concept:animalpreyson",
        "concept:animalistypeofanimal",
    ),
    (
        "concept:worksfor",
        "concept:personleadsorganization",
        "concept:organizationhiredperson",
        "concept:agentcontrols",
        "concept:agentcollaborateswithagent",
        "concept:hasofficeincity",
        "concept:organizationheadquarteredinstateorprovince",
    ),
)


def _clean_relation(value: str) -> str:
    value = value.strip().strip("`'\" ")
    value = re.sub(r"[\s\.,;]+$", "", value)
    return value


def parse_relation_candidates(question: str) -> list[str]:
    """Extract candidate relation labels while preserving their order.

    Evaluation questions use a semicolon-separated list and may end in a
    full-stop.  Returning an empty list for direct-format questions makes the
    function safe to use with both the old and the candidate-aware datasets.
    """
    match = _CANDIDATE_RE.search(question)
    if not match:
        return []
    return [candidate for candidate in (_clean_relation(x) for x in match.group(1).split(";")) if candidate]


def parse_question_entities(question: str) -> tuple[str, str] | tuple[None, None]:
    """Return the two entity names used in a NELL relation question."""
    entities = _ENTITY_RE.findall(question)
    if len(entities) >= 2:
        return entities[0], entities[1]
    return None, None


def infer_entity_type(entity_name: str) -> str:
    """Infer a coarse NELL entity type from its canonical name.

    NELL names are mostly ``concept_<type>_<instance>``.  This is a fallback
    only; observed role statistics in :class:`RelationMemory` take precedence.
    """
    name = entity_name.lower()
    if name.startswith("concept_"):
        name = name[len("concept_"):]
    head = name.split("_", 1)[0]
    aliases = {
        "stateorprovince": "geopolitical_location",
        "country": "geopolitical_location",
        "city": "geopolitical_location",
        "location": "geopolitical_location",
        "person": "person",
        "organization": "organization",
        "company": "organization",
        "animal": "animal",
        "bird": "animal",
        "fish": "animal",
        "insect": "animal",
        "food": "food",
        "event": "event",
    }
    return aliases.get(head, head or "unknown")


def _read_triples(triples: str | Path | Iterable[Sequence[str]]) -> Iterable[tuple[str, str, str]]:
    if isinstance(triples, (str, Path)):
        with open(triples, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                yield fields[0], fields[1], fields[2]
        return
    for row in triples:
        if len(row) >= 3:
            yield str(row[0]), str(row[1]), str(row[2])


@dataclass(frozen=True)
class RelationCard:
    relation_name: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    frequency: int
    support_examples: tuple[tuple[str, str], ...]
    confusing_relations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class RelationMemory:
    """Auditable relation prototypes built from training triples only."""

    def __init__(
        self,
        cards: Mapping[str, RelationCard],
        entity_roles: Mapping[str, Mapping[str, int]],
        pair_relations: Mapping[str, Sequence[str]] | None = None,
        composition_counts: Mapping[str, Mapping[str, int]] | None = None,
        path_signatures: Mapping[str, Mapping[str, int]] | None = None,
        composition_patterns: Mapping[str, Mapping[str, int]] | None = None,
    ):
        self.cards = dict(cards)
        self.entity_roles = {key: dict(value) for key, value in entity_roles.items()}
        self.pair_relations = {
            key: tuple(sorted(set(values))) for key, values in (pair_relations or {}).items()
        }
        self.composition_counts = {
            key: dict(values) for key, values in (composition_counts or {}).items()
        }
        self.path_signatures = {
            key: dict(values) for key, values in (path_signatures or {}).items()
        }
        self.composition_patterns = {
            key: dict(values) for key, values in (composition_patterns or {}).items()
        }

    def card(self, relation: str) -> RelationCard | None:
        return self.cards.get(relation)

    def entity_type(self, entity: str, role: str | None = None) -> str:
        roles = self.entity_roles.get(entity, {})
        if role:
            typed = [key.split("|", 1)[1] for key, count in roles.items() if key.startswith(role + "|") and count]
            if typed:
                return Counter(typed).most_common(1)[0][0]
        typed = [key.split("|", 1)[1] for key, count in roles.items() if count]
        return Counter(typed).most_common(1)[0][0] if typed else infer_entity_type(entity)

    def relation_prior(self, relation: str, source: str | None = None, target: str | None = None) -> float:
        """Return a conservative type-compatibility prior in [0, 1]."""
        card = self.card(relation)
        if card is None:
            return 0.0
        if not source or not target:
            return 0.0
        source_type = self.entity_type(source, "source")
        target_type = self.entity_type(target, "target")
        source_ok = source_type in card.source_types
        target_ok = target_type in card.target_types
        if source_ok and target_ok:
            return 1.0
        if source_ok or target_ok:
            return 0.5
        return 0.0

    @staticmethod
    def _pair_key(source: str | None, target: str | None) -> str:
        return f"{source}\t{target}" if source and target else ""

    def evidence_score(self, relation: str, source: str | None = None, target: str | None = None) -> float:
        """Return train-only exact or two-hop composition evidence in [0, 1]."""
        key = self._pair_key(source, target)
        if not key:
            return 0.0
        if relation in self.pair_relations.get(key, ()):
            return 1.0
        signatures = self.path_signatures.get(key, {})
        if not signatures:
            return 0.0
        total_paths = sum(max(0, int(value)) for value in signatures.values())
        if total_paths <= 0:
            return 0.0
        support = 0.0
        for signature, path_count in signatures.items():
            pattern_counts = self.composition_patterns.get(signature, {})
            pattern_total = sum(max(0, int(value)) for value in pattern_counts.values())
            if pattern_total:
                support += int(path_count) * int(pattern_counts.get(relation, 0)) / pattern_total
        return min(0.95, support / total_paths)

    def evidence_detail(self, relation: str, source: str | None = None, target: str | None = None) -> dict:
        key = self._pair_key(source, target)
        return {
            "exact_train_edge": bool(key and relation in self.pair_relations.get(key, ())),
            "pair_relations": list(self.pair_relations.get(key, ())) if key else [],
            "composition_counts": dict(self.composition_counts.get(key, {})) if key else {},
            "path_signatures": dict(self.path_signatures.get(key, {})) if key else {},
            "score": self.evidence_score(relation, source, target),
        }

    def render_card(
        self,
        relation: str,
        candidates: Sequence[str] | None = None,
        max_types: int = 8,
        max_examples: int = 3,
    ) -> str:
        """Render a compact relation card for optional prompt injection.

        The card is deliberately short and candidate-local.  It exposes the
        prototype and its closest competitors, rather than dumping the full
        training graph into the prompt.
        """
        card = self.card(relation)
        if card is None:
            return (
                f"Candidate relation: {relation}\n"
                "No training prototype is available; rely on the question and candidate set."
            )
        candidate_set = set(candidates or ())
        confusing = [
            item for item in card.confusing_relations
            if not candidate_set or item in candidate_set
        ]
        examples = [f"({source}, {target})" for source, target in card.support_examples[:max_examples]]
        lines = [
            "CECR relation card for the candidate currently being evaluated:",
            f"- relation: {card.relation_name}",
            f"- observed source types: {', '.join(card.source_types[:max_types]) or 'unknown'}",
            f"- observed target types: {', '.join(card.target_types[:max_types]) or 'unknown'}",
            f"- training frequency: {card.frequency}",
            f"- support examples: {'; '.join(examples) or 'none'}",
            f"- confusing candidate relations: {', '.join(confusing) or 'none'}",
            "Use this card only to contrast candidates; do not output the card.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "cards": {name: card.to_dict() for name, card in self.cards.items()},
            "entity_count": len(self.entity_roles),
            "entity_roles": self.entity_roles,
            "pair_relations": {key: list(values) for key, values in self.pair_relations.items()},
            "composition_counts": self.composition_counts,
            "path_signatures": self.path_signatures,
            "composition_patterns": self.composition_patterns,
        }


def build_relation_memory_from_triples(
    triples: str | Path | Iterable[Sequence[str]],
    max_support_examples: int = 8,
) -> RelationMemory:
    """Build relation cards without reading validation/test edges.

    ``triples`` should normally be ``outputs/data/nell23k/train.txt``.  The
    function accepts an iterable as well to keep unit tests and small probes
    easy to write.
    """
    rows = list(_read_triples(triples))
    relation_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"source_types": Counter(), "target_types": Counter(), "examples": [], "frequency": 0}
    )
    entity_roles: dict[str, Counter] = defaultdict(Counter)
    pair_relations: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, relation, target in rows:
        source_type, target_type = infer_entity_type(source), infer_entity_type(target)
        stats = relation_stats[relation]
        stats["frequency"] = int(stats["frequency"]) + 1
        stats["source_types"][source_type] += 1
        stats["target_types"][target_type] += 1
        if len(stats["examples"]) < max_support_examples:
            stats["examples"].append((source, target))
        entity_roles[source]["source|" + source_type] += 1
        entity_roles[target]["target|" + target_type] += 1
        pair_relations[f"{source}\t{target}"].add(relation)
        outgoing[source].append((relation, target))

    # Learn relation-level two-hop evidence from train graph paths whose
    # endpoint also has a train edge.  No validation/test edges are read.
    composition_counts: dict[str, Counter] = defaultdict(Counter)
    path_signatures: dict[str, Counter] = defaultdict(Counter)
    composition_patterns: dict[str, Counter] = defaultdict(Counter)
    for source, first_relation, middle in rows:
        for second_relation, target in outgoing.get(middle, ()):
            if source == target:
                continue
            pair_key = f"{source}\t{target}"
            signature = f"{first_relation}\t{second_relation}"
            path_signatures[pair_key][signature] += 1
            for target_relation in pair_relations.get(f"{source}\t{target}", ()):
                composition_counts[pair_key][target_relation] += 1
                composition_patterns[signature][target_relation] += 1

    confusion_by_relation: dict[str, set[str]] = defaultdict(set)
    for group in CONFUSION_SETS:
        for relation in group:
            confusion_by_relation[relation].update(other for other in group if other != relation)

    cards = {}
    for relation, stats in relation_stats.items():
        cards[relation] = RelationCard(
            relation_name=relation,
            source_types=tuple(x for x, _ in stats["source_types"].most_common()),
            target_types=tuple(x for x, _ in stats["target_types"].most_common()),
            frequency=int(stats["frequency"]),
            support_examples=tuple(stats["examples"]),
            confusing_relations=tuple(sorted(confusion_by_relation.get(relation, set()))),
        )
    return RelationMemory(
        cards,
        entity_roles,
        pair_relations,
        composition_counts,
        path_signatures,
        composition_patterns,
    )


def save_relation_memory(memory: RelationMemory, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(memory.to_dict(), handle, ensure_ascii=False, indent=2)


def load_relation_memory(path: str | Path) -> RelationMemory:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cards = {
        name: RelationCard(
            relation_name=data["relation_name"],
            source_types=tuple(data.get("source_types", [])),
            target_types=tuple(data.get("target_types", [])),
            frequency=int(data.get("frequency", 0)),
            support_examples=tuple(tuple(x) for x in data.get("support_examples", [])),
            confusing_relations=tuple(data.get("confusing_relations", [])),
        )
        for name, data in payload.get("cards", {}).items()
    }
    return RelationMemory(
        cards,
        payload.get("entity_roles", {}),
        payload.get("pair_relations", {}),
        payload.get("composition_counts", {}),
        payload.get("path_signatures", {}),
        payload.get("composition_patterns", {}),
    )
