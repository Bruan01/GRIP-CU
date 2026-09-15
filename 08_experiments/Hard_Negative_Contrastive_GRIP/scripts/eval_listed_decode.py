"""Evaluate a saved Stage-2 adapter with greedy generation and/or closed-set ranking.

Closed-set ranking scores the prompt's official 10-way continuations with the
same length-normalized likelihood used by listed InfoNCE. It does not train.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))
sys.path.insert(0, str(HNG / "scripts"))

from constants import SYSTEM_PROMPT  # noqa: E402
from evaluation.recurrent_metrics import exact_match  # noqa: E402
from grip.tasks.recurrent_tasks.task_dataset import QUESTION_TEMPLATE  # noqa: E402
from hard_negative_grip.listed_training import (  # noqa: E402
    format_answer_prefix,
    pack_decision_set_rows,
    pick_closed_set_answer,
    score_candidate_rows,
)
from hard_negative_grip.metrics import summarize_candidate_scores  # noqa: E402
from hard_negative_grip.official_lists import listed_relations_from_sample  # noqa: E402
from hard_negative_grip.task_file import load_graph_record  # noqa: E402
from train_listed_contrastive import (  # noqa: E402
    em_summary,
    evaluate_adapter,
    load_adapter,
    release_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument(
        "--model_cache_dir",
        default=str(GRIP_EXP / "model_cache"),
    )
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument(
        "--decode",
        choices=["generate", "closed_set", "both"],
        default="closed_set",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Merge b1/ and listed/ summaries under --output_dir; skip model load.",
    )
    return parser.parse_args()


def closed_set_summary(rows: list[dict]) -> dict:
    summary = em_summary(rows)
    ranking_rows = []
    for row in rows:
        gold = row["target"][0]
        scores = row["scores"]
        ranking_rows.append(
            {
                "positive_score": scores[gold],
                "negative_scores": [score for rel, score in scores.items() if rel != gold],
            }
        )
    ranking = summarize_candidate_scores(ranking_rows, ks=(1, 3, 10))
    summary["mrr"] = ranking["mrr"]
    summary["hits@1"] = ranking["hits@1"]
    summary["hits@3"] = ranking["hits@3"]
    summary["hits@10"] = ranking["hits@10"]
    return summary


def evaluate_closed_set(model, tokenizer, record: dict) -> list[dict]:
    samples = [
        item for item in record["recurrent_questions"] if item["split"] in {"validation", "test"}
    ]
    title = record.get("title", "nell23k")
    device = next(model.parameters()).device
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            gold = str(sample["answer"])
            relations = list(listed_relations_from_sample(sample))
            if gold not in relations:
                relations.append(gold)
            prefix = format_answer_prefix(
                tokenizer,
                title=title,
                question=sample["question"],
                system_prompt=SYSTEM_PROMPT,
                question_template=QUESTION_TEMPLATE,
            )
            token_rows, prefix_lens = pack_decision_set_rows(tokenizer, prefix, relations)
            scores = score_candidate_rows(model, tokenizer, token_rows, prefix_lens, device)
            score_list = [float(value) for value in scores.tolist()]
            predicted = pick_closed_set_answer(relations, score_list)
            score_map = {relation: score for relation, score in zip(relations, score_list)}
            target = [gold]
            rows.append(
                {
                    "question_id": sample.get("question_id"),
                    "split": sample.get("split"),
                    "question": sample["question"],
                    "target": target,
                    "response": predicted,
                    "correct": exact_match(predicted, target),
                    "in_list": predicted in relations,
                    "scores": score_map,
                    "gold_score": score_map[gold],
                }
            )
            if index == 1 or index % 16 == 0 or index == len(samples):
                hits = sum(bool(row["correct"]) for row in rows)
                print(
                    f"[closed_set] {index}/{len(samples)} hits@1={hits / len(rows):.4f}",
                    flush=True,
                )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_decode_comparison(output_dir: Path) -> dict:
    b1_gen = load_optional_json(output_dir / "b1" / "summary.json")
    listed_gen = load_optional_json(output_dir / "listed" / "summary.json")
    b1_closed = load_optional_json(output_dir / "b1" / "summary_closed_set.json")
    listed_closed = load_optional_json(output_dir / "listed" / "summary_closed_set.json")
    if b1_closed is None or listed_closed is None:
        raise FileNotFoundError(f"need closed-set summaries under {output_dir}/b1 and listed")
    comparison = {
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "b1_generate": b1_gen,
        "listed_generate": listed_gen,
        "b1_closed_set": b1_closed,
        "listed_closed_set": listed_closed,
        "listed_minus_b1_generate_em": (
            None
            if b1_gen is None or listed_gen is None
            else listed_gen["all"]["em"] - b1_gen["all"]["em"]
        ),
        "listed_closed_set_minus_b1_generate_em": (
            None if b1_gen is None else listed_closed["all"]["em"] - b1_gen["all"]["em"]
        ),
        "listed_closed_set_minus_b1_closed_set_em": (
            listed_closed["all"]["em"] - b1_closed["all"]["em"]
        ),
        "listed_closed_set_minus_listed_generate_em": (
            None if listed_gen is None else listed_closed["all"]["em"] - listed_gen["all"]["em"]
        ),
        "note": (
            "Closed-set EM is argmax over the official 10-way continuations. "
            "A paper signal needs listed closed-set > B1 generate on the test split, "
            "not only on this smoke/pilot slice."
        ),
    }
    write_json(output_dir / "comparison_closed_set.json", comparison)
    print("\n=== closed-set comparison ===", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return comparison


def evaluate_one(args: argparse.Namespace) -> None:
    if args.adapter_dir is None:
        raise ValueError("--adapter_dir is required unless --compare is set")
    record = load_graph_record(args.eval_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_args = SimpleNamespace(
        model_name=args.model_name,
        model_cache_dir=args.model_cache_dir,
    )
    model, tokenizer = load_adapter(load_args, args.adapter_dir, trainable=False)
    try:
        if args.decode in {"generate", "both"}:
            rows = evaluate_adapter(model, tokenizer, record, args.gen_max_length)
            summary = em_summary(rows)
            write_jsonl(args.output_dir / "predictions_correct.jsonl", rows)
            write_json(args.output_dir / "summary.json", summary)
            print(
                f"[eval generate] {args.adapter_dir} EM={summary['all']['em']:.4f} "
                f"n={summary['all']['count']}",
                flush=True,
            )
        if args.decode in {"closed_set", "both"}:
            rows = evaluate_closed_set(model, tokenizer, record)
            summary = closed_set_summary(rows)
            write_jsonl(args.output_dir / "predictions_closed_set.jsonl", rows)
            write_json(args.output_dir / "summary_closed_set.json", summary)
            print(
                f"[eval closed_set] {args.adapter_dir} EM={summary['all']['em']:.4f} "
                f"hits@1={summary['hits@1']:.4f} n={summary['all']['count']}",
                flush=True,
            )
    finally:
        del model
        del tokenizer
        release_cuda()


def main() -> None:
    args = parse_args()
    if args.compare:
        write_decode_comparison(args.output_dir)
        return
    evaluate_one(args)


if __name__ == "__main__":
    main()
