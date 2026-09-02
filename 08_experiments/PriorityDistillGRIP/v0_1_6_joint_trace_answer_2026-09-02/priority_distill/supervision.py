"""Registered training-only evidence variants for the oracle gate."""

from __future__ import annotations

import hashlib
from typing import Iterable

from .records import build_training_answer_prompt

METHODS = (
    "answer_only",
    "more_qa_equal_token",
    "random_path_equal_token",
    "all_paths_equal_token",
    "oracle_priority_equal_token",
    "all_paths_terminal_masked_equal_token",
    "explicit_path_selection",
    "explicit_path_selection_terminal_masked",
    "graph_free_trace_terminal_masked",
    "graph_free_trace_answer_joint",
)
STAGE1_METHODS = METHODS[1:]


def _stable_int(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def serialize_path(candidate: dict) -> str:
    nodes = candidate["path_nodes"]
    relations = candidate["path_relations"]
    pieces = [str(nodes[0])]
    for relation, node in zip(relations, nodes[1:]):
        pieces.append(f"--{relation}--> {node}")
    return " ".join(pieces)


def _evidence_prompt(question: str, serialized_paths: Iterable[str]) -> str:
    paths = list(serialized_paths)
    lines = [
        "You are internalizing graph evidence for later graph-free reasoning.",
        "Use the selected path evidence to learn the answer, then return only the final entity identifier.",
        "",
        "Question:",
        question,
        "",
        "Candidate evidence:",
    ]
    for index, path in enumerate(paths, start=1):
        lines.append(f"Path {index}: " + path)
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def _ordered_candidates(pool: list[dict], seed: int, task_id: str) -> list[dict]:
    return sorted(pool, key=lambda candidate: _stable_int(seed, task_id, "order", candidate["source_task_id"]))


def _masked_path(candidate: dict) -> str:
    """Serialize a path without exposing its terminal entity."""
    nodes = candidate["path_nodes"]
    relations = candidate["path_relations"]
    pieces = [str(nodes[0])]
    for index, relation in enumerate(relations):
        tail = "<MASKED_TERMINAL>" if index == len(relations) - 1 else str(nodes[index + 1])
        pieces.append(f"--{relation}--> {tail}")
    return " ".join(pieces)


def _anti_copy_evidence_prompt(question: str, candidates: list[dict], seed: int, task_id: str) -> str:
    """Present paths with hidden terminals and an unassociated endpoint pool.

    The endpoint list makes the diagnostic task well-defined, while hiding the
    terminal from each path prevents a one-token terminal-to-answer copy.
    The model must associate one endpoint with the relation chain.
    """
    ordered = _ordered_candidates(candidates, seed, task_id)
    endpoints = sorted(
        (str(candidate["answer"]) for candidate in ordered),
        key=lambda answer: _stable_int(seed, task_id, "endpoint", answer),
    )
    lines = [
        "You are internalizing graph evidence for later graph-free reasoning.",
        "Match the question to the correct candidate path, infer its hidden terminal from the graph evidence, then return only the final entity identifier.",
        "",
        "Question:",
        question,
        "",
        "Candidate paths (terminal entity hidden):",
    ]
    for index, candidate in enumerate(ordered, start=1):
        lines.append(f"Path {index}: " + _masked_path(candidate))
    lines.extend(["", "Unassociated candidate terminal entities:"])
    lines.extend(f"- {answer}" for answer in endpoints)
    lines.extend(["", "Answer:"])
    return "\n".join(lines)



def _explicit_selection_prompt(
    question: str,
    candidates: list[dict],
    seed: int,
    task_id: str,
    terminal_masked: bool = False,
) -> tuple[str, int]:
    """Build a numbered candidate-selection prompt and return gold position.

    The ordering is deterministic but independent of the gold flag.  In the
    masked variant, terminals are hidden from paths and presented separately,
    so emitting the answer cannot be solved by copying the last path token.
    """
    ordered = _ordered_candidates(candidates, seed, task_id)
    gold_position = next(index for index, candidate in enumerate(ordered) if candidate["is_gold"])
    lines = [
        "You are learning to select the correct path for later graph-free reasoning.",
        "Read the question and all candidate paths, select the path that answers the question, and return exactly two lines.",
        "Line 1 must be: Selected path: <path number>",
        "Line 2 must be: Answer: <entity identifier>",
        "",
        "Question:",
        question,
        "",
        "Candidate paths" + (" (terminal entity hidden):" if terminal_masked else ":"),
    ]
    for index, candidate in enumerate(ordered, start=1):
        path = _masked_path(candidate) if terminal_masked else serialize_path(candidate)
        lines.append(f"Path {index}: {path}")
    if terminal_masked:
        endpoints = sorted(
            (str(candidate["answer"]) for candidate in ordered),
            key=lambda answer: _stable_int(seed, task_id, "endpoint", answer),
        )
        lines.extend(["", "Unassociated candidate terminal entities:"])
        lines.extend(f"- {answer}" for answer in endpoints)
    lines.extend(["", "Output:"])
    return "\n".join(lines), gold_position

def _graph_free_trace_prompt(question: str) -> str:
    """Build a graph-free prompt for structured trace supervision.

    The prompt intentionally contains only the question.  Gold intermediate
    entities are targets, never inputs, so evaluation can continue to use the
    ordinary graph-free prompt without leaking graph evidence.
    """
    return (
        "Learn to solve this graph-path question by composing the ordered "
        "relations. Return exactly two lines: a Trace line and an Answer line."
        "\n\nQuestion:\n"
        f"{question}\n\nOutput:\n"
    )


def _graph_free_trace_target(row: dict, *, joint: bool = False) -> tuple[str, int, list[dict] | None]:
    """Return a trace with only intermediate nodes and a masked terminal.

    For depth 1 the trace is still valid: ``head -> <MASKED_TERMINAL>``.
    The final entity is supervised separately after ``Answer:``.
    """
    nodes = [str(node) for node in row["path_nodes"]]
    trace_nodes = nodes[:-1] + ["<MASKED_TERMINAL>"]
    trace_line = f"Trace: {' -> '.join(trace_nodes)}"
    answer_line = f"Answer: {row['answer']}"
    target = trace_line + "\n" + answer_line
    segments = ([{"text": trace_line, "weight": 0.25}, {"text": "\n" + answer_line, "weight": 1.0}] if joint else None)
    return target, max(0, len(nodes) - 2), segments


def build_method_supervision(rows: list[dict], pools: dict[str, list[dict]], method: str, seed: int) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    supervision: list[dict] = []
    for row in sorted(rows, key=lambda item: item["task_id"]):
        pool = pools[row["task_id"]]
        gold = next(candidate for candidate in pool if candidate["is_gold"])
        distractors = [candidate for candidate in pool if not candidate["is_gold"]]
        selected: list[dict]
        evidence_kind: str
        gold_position = -1
        target_text = row["answer"]
        target_segments = None
        intermediate_node_count = 0
        if method in {"answer_only", "more_qa_equal_token"}:
            selected = []
            evidence_kind = "answer_rehearsal" if method == "more_qa_equal_token" else "answer_only"
            prompt = build_training_answer_prompt(row["text"])
        elif method in {"graph_free_trace_terminal_masked", "graph_free_trace_answer_joint"}:
            selected = []
            joint = method == "graph_free_trace_answer_joint"
            evidence_kind = "graph_free_structured_trace_answer_joint" if joint else "graph_free_structured_trace_target"
            prompt = _graph_free_trace_prompt(row["text"])
            target_text, intermediate_node_count, target_segments = _graph_free_trace_target(row, joint=joint)
        elif method in {"explicit_path_selection", "explicit_path_selection_terminal_masked"}:
            selected = _ordered_candidates(pool, seed, row["task_id"])
            terminal_masked = method == "explicit_path_selection_terminal_masked"
            evidence_kind = "explicit_candidate_selection_terminal_masked" if terminal_masked else "explicit_candidate_selection"
            prompt, gold_position = _explicit_selection_prompt(
                row["text"], selected, seed, row["task_id"], terminal_masked=terminal_masked
            )
        elif method == "oracle_priority_equal_token":
            selected = [gold]
            evidence_kind = "oracle_gold_path"
            gold_position = 0
            prompt = _evidence_prompt(row["text"], [serialize_path(gold)])
        elif method == "random_path_equal_token":
            index = _stable_int(seed, row["task_id"], "random_distractor") % len(distractors)
            selected = [distractors[index]]
            evidence_kind = "random_same_depth_path"
            prompt = _evidence_prompt(row["text"], [serialize_path(selected[0])])
        elif method == "all_paths_terminal_masked_equal_token":
            selected = _ordered_candidates(pool, seed, row["task_id"])
            evidence_kind = "gold_plus_distractors_terminal_masked"
            gold_position = next(index for index, candidate in enumerate(selected) if candidate["is_gold"])
            prompt = _anti_copy_evidence_prompt(row["text"], selected, seed, row["task_id"])
        else:
            selected = _ordered_candidates(pool, seed, row["task_id"])
            evidence_kind = "gold_plus_distractors_unprioritized"
            gold_position = next(index for index, candidate in enumerate(selected) if candidate["is_gold"])
            prompt = _evidence_prompt(row["text"], [serialize_path(candidate) for candidate in selected])
        supervision.append(
            {
                "task_id": row["task_id"],
                "source_split": row["split"],
                "method": method,
                "prompt": prompt,
                "answer": row["answer"],
                "target_text": (
                    f"Selected path: {gold_position + 1}\nAnswer: {row['answer']}"
                    if method in {"explicit_path_selection", "explicit_path_selection_terminal_masked"}
                    else target_text
                ),
                "trace_target": target_text if method in {"graph_free_trace_terminal_masked", "graph_free_trace_answer_joint"} else None,
                "target_segments": target_segments,
                "trace_loss_weight": 0.25 if method == "graph_free_trace_answer_joint" else 1.0,
                "answer_loss_weight": 1.0,
                "intermediate_node_count": intermediate_node_count,
                "selection_target": gold_position + 1 if gold_position >= 0 else None,
                "depth_label": int(row["depth_label"]),
                "evidence_kind": evidence_kind,
                "candidate_count": len(selected),
                "gold_position": gold_position,
                "selected_source_task_ids": [candidate["source_task_id"] for candidate in selected],
                "selected_source_splits": [candidate["source_split"] for candidate in selected],
            }
        )
    return supervision


def supervision_audit(method_rows: dict[str, list[dict]]) -> dict:
    return {
        method: {
            "count": len(rows),
            "depth_counts": {str(depth): sum(row["depth_label"] == depth for row in rows) for depth in range(1, 5)},
            "candidate_count_mean": sum(row["candidate_count"] for row in rows) / len(rows),
            "gold_inclusion_rate": sum(row["gold_position"] >= 0 for row in rows) / len(rows),
            "train_only_selected_sources": all(
                split == "train" for row in rows for split in row["selected_source_splits"]
            ),
        }
        for method, rows in method_rows.items()
    }


def build_graph_free_trace_evaluation_prompt(question: str) -> str:
    """Use the exact structured output contract learned by trace protocols."""
    return _graph_free_trace_prompt(question)


def _gold_trace_text(row: dict) -> str:
    nodes = [str(node) for node in row["path_nodes"]]
    return " -> ".join(nodes[:-1] + ["<MASKED_TERMINAL>"])


def build_gold_trace_prompt(row: dict) -> str:
    """Diagnostic prompt: reveal correct intermediate nodes, hide the terminal."""
    return (
        "Use the supplied gold intermediate trace to answer the graph-path question. "
        "The terminal entity is hidden, so infer it rather than copying it. "
        "Return exactly two lines: a Trace line and an Answer line.\n\n"
        f"Question:\n{row['text']}\n\n"
        f"Gold intermediate trace (terminal hidden): {_gold_trace_text(row)}\n\n"
        "Output:\n"
    )


def build_gold_trace_unmasked_prompt(row: dict) -> str:
    """Diagnostic prompt: reveal the complete gold path, including its terminal.

    This is deliberately an oracle-only sanity check.  It is not a graph-free
    metric: the prompt contains the answer token itself.  Comparing it with
    ``build_gold_trace_prompt`` isolates whether terminal masking, rather than
    relation-chain use, is the immediate bottleneck.
    """
    full_trace = " -> ".join(str(node) for node in row["path_nodes"])
    return (
        "Use the supplied complete gold trace to answer the graph-path question. "
        "The terminal is visible; return exactly two lines: a Trace line and an Answer line.\n\n"
        f"Question:\n{row['text']}\n\n"
        f"Gold trace (terminal visible): {full_trace}\n\n"
        "Output:\n"
    )


def build_oracle_evidence_prompt(row: dict) -> str:
    """Build the diagnostic-only prompt that exposes the row's gold path."""
    candidate = {
        "path_nodes": row["path_nodes"],
        "path_relations": row["path_relations"],
    }
    return _evidence_prompt(row["text"], [serialize_path(candidate)])
