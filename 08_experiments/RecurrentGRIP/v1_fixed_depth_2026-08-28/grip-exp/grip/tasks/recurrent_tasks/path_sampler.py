from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .frontier_builder import bfs_frontiers, build_adjacency, shortest_path

_STATION_SHORTEST_RE = re.compile(
    r"How many stations are between\s+(.+?)\s+and\s+(.+?)\?",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class PathQuestion:
    question_id: str
    question: str
    answer: str
    question_type: str
    source_node: int
    target_node: int
    true_hop: int
    shortest_path: list[int]
    frontiers: list[list[int]]
    split: str = "unassigned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_node_name(node_text: str) -> str:
    text = str(node_text).strip()
    for marker in (" does not have ", " has "):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text


def infer_question_type(question: str) -> Optional[str]:
    if _STATION_SHORTEST_RE.search(question):
        return "StationShortestCount"
    return None


def extract_station_shortest_endpoints(question: str, node_list: list[str]) -> tuple[int, int]:
    match = _STATION_SHORTEST_RE.search(question)
    if match is None:
        raise ValueError("question does not match StationShortestCount")
    source_name, target_name = (part.strip() for part in match.groups())
    names = [canonical_node_name(text) for text in node_list]
    source_matches = [index for index, name in enumerate(names) if name == source_name]
    target_matches = [index for index, name in enumerate(names) if name == target_name]
    if len(source_matches) != 1 or len(target_matches) != 1:
        raise ValueError(
            f"endpoint extraction ambiguous: {source_name!r}->{source_matches}, "
            f"{target_name!r}->{target_matches}"
        )
    if source_matches[0] == target_matches[0]:
        raise ValueError("source and target must be distinct")
    return source_matches[0], target_matches[0]


def _parse_integer_label(answer: Any) -> int:
    if isinstance(answer, bool):
        raise ValueError("boolean labels are not valid path-count answers")
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    match = re.fullmatch(r"[-+]?\d+", text)
    if match is None:
        raise ValueError(f"answer is not an integer: {answer!r}")
    return int(text)


def make_path_question(
    graph_id: str,
    local_index: int,
    graph: dict[str, Any],
    question: str,
    answer: Any,
    question_type: Optional[str] = None,
) -> PathQuestion:
    resolved_type = question_type or infer_question_type(question)
    if resolved_type != "StationShortestCount":
        raise ValueError(f"unsupported question type: {resolved_type!r}")
    node_list = graph["node_list"]
    source, target = extract_station_shortest_endpoints(question, node_list)
    adjacency = build_adjacency(len(node_list), graph["edge_index"], directed=False)
    path = shortest_path(adjacency, source, target)
    if not path:
        raise ValueError("endpoints are disconnected")
    true_hop = len(path) - 1
    expected_label = max(true_hop - 1, 0)
    actual_label = _parse_integer_label(answer)
    if actual_label != expected_label:
        raise ValueError(f"label/path mismatch: expected {expected_label}, got {actual_label}")
    return PathQuestion(
        question_id=f"{graph_id}:{local_index}",
        question=question,
        answer=str(answer),
        question_type=resolved_type,
        source_node=source,
        target_node=target,
        true_hop=true_hop,
        shortest_path=path,
        frontiers=bfs_frontiers(adjacency, source, true_hop),
    )
