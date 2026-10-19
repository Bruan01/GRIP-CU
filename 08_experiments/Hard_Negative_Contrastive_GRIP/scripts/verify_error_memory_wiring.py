"""CPU-only wiring gate for the frozen error-memory bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "src"))

from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_gold,
    build_qa_assets_from_task_texts,
    is_grip_task_file,
    is_relation_gold,
    load_json_payload,
    load_counterfactual_manifest,
    load_error_memory_manifest,
    original_question_ids_from_payload,
    train_relation_alias_index,
    validate_structure_counterfactual_manifest,
    validate_error_memory_manifest,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument(
        "--relation_text_file",
        type=Path,
        default=None,
        help="Optional raw-relation to task-label mapping, required for FB15K237-2.",
    )
    parser.add_argument("--expected_listed", type=int, default=3253)
    parser.add_argument(
        "--negative_source",
        choices=("error_memory", "structure_counterfactual"),
        default="error_memory",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json_payload(args.task_file)
    if not is_grip_task_file(payload):
        raise ValueError(f"{args.task_file} is not a GRIP task file")
    qa_texts = list(payload["qa_samples"])
    question_ids = original_question_ids_from_payload(payload, len(qa_texts))
    relation_order = load_train_relation_order(
        args.raw_dir, relation_text_file=args.relation_text_file
    )
    if args.negative_source == "error_memory":
        manifest = load_error_memory_manifest(args.manifest)
        validate_error_memory_manifest(manifest)
    else:
        manifest = load_counterfactual_manifest(args.manifest)
        validate_structure_counterfactual_manifest(manifest)
    resolved_ids = question_ids or [f"task_qa:{index}" for index in range(len(qa_texts))]
    _, metas = build_qa_assets_from_task_texts(
        qa_texts,
        seed=2026,
        listed_negative_k=9,
        listed_negative_source=args.negative_source,
        relation_order=relation_order,
        error_memory_manifest=manifest if args.negative_source == "error_memory" else None,
        counterfactual_manifest=(
            manifest if args.negative_source == "structure_counterfactual" else None
        ),
        question_ids=resolved_ids,
    )
    alias_index = train_relation_alias_index(relation_order)
    relation_n = sum(
        1
        for text in qa_texts
        if is_relation_gold(assistant_gold(text), text, relation_alias_index=alias_index)
    )
    listed_counts = [len(meta["listed_relations"]) for meta in metas]
    listed_n = sum(1 for count in listed_counts if count > 0)
    if args.expected_listed >= 0 and listed_n != args.expected_listed:
        raise ValueError(
            f"{args.negative_source} bank attaches to {listed_n} QA rows, "
            f"expected {args.expected_listed}"
        )
    if listed_n != relation_n:
        raise ValueError(
            f"error-memory bank covers {listed_n}/{relation_n} relation QA rows; "
            "unmatched train relations are not allowed in the full-task run"
        )
    if any(count != 9 for count in listed_counts if count > 0):
        raise ValueError("error-memory wiring found a listed row whose K is not 9")
    output = {
        "task_file": str(args.task_file),
        "manifest": str(args.manifest),
        "negative_source": args.negative_source,
        "manifest_sha256": file_sha256(args.manifest),
        "raw_dir": str(args.raw_dir),
        "n_qa": len(qa_texts),
        "relation_qa": relation_n,
        "listed_qa": listed_n,
        "question_id_source": "original" if question_ids is not None else "positional",
        "relation_vocab_size": len(relation_order),
        "k": 9,
        "cpu_only": True,
        "train_only": True,
        "same_questions_as_task": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
