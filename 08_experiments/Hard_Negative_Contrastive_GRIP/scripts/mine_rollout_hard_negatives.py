"""Mine hard negatives from frozen B1 free-generation errors on training QA.

The script only runs on ``qa_samples`` from the training task file.  For each
relation QA item it greedily generates one answer from a frozen B1 adapter.  If
the parsed answer is wrong and looks like a relation label, it becomes one
hard negative.  The remaining negatives are sampled uniformly from the
training-graph relation vocabulary, excluding known true relations.  This
keeps the hard negative aligned with an observed model error while avoiding
validation/test predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE  # noqa: E402
from evaluation.recurrent_metrics import parse_recurrent_answer  # noqa: E402
from hard_negative_grip.official_lists import load_train_relation_order  # noqa: E402
from hard_negative_grip.task_file import (  # noqa: E402
    assistant_answer_prefix,
    assistant_gold,
    is_relation_gold,
    known_pair_relations,
    load_json_payload,
    match_train_relation,
    question_entity_pair,
    train_relation_alias_index,
)
from models.utils import get_hf_llm_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--b1_adapter", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument("--total_k", type=int, default=9)
    parser.add_argument("--candidate_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and skip question IDs already present.",
    )
    return parser.parse_args()


def load_b1(args: argparse.Namespace):
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=str(args.model_cache_dir),
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, str(args.b1_adapter), is_trainable=False)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def encode(tokenizer, text: str) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if ids:
        return ids
    unk = getattr(tokenizer, "unk_token_id", None)
    return [0 if unk is None else unk]


def canonical_candidate(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace("<answer>", "").replace("</answer>", "").strip()
    if not value or value.lower() in {"yes", "no", "i don't know"}:
        return ""
    # Keep only relation-shaped outputs. This includes malformed concept:
    # labels, which are precisely the OOV errors this miner is meant to expose.
    if not value.lower().startswith("concept:"):
        return ""
    if "\n" in value or "<" in value or ">" in value:
        return ""
    return value


def generate_one(model, tokenizer, prefix: str, max_new_tokens: int) -> tuple[str, str]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([encode(tokenizer, prefix)], device=device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(
        generated[0][input_ids.shape[-1] :], skip_special_tokens=True
    ).strip()
    return raw, parse_recurrent_answer(raw)


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("question_id"):
                raise ValueError(f"{path}:{line_number}: invalid manifest row")
            question_id = str(row["question_id"])
            if question_id in completed:
                raise ValueError(f"{path}:{line_number}: duplicate {question_id!r}")
            completed.add(question_id)
    return completed


def main() -> None:
    args = parse_args()
    if args.total_k < 1:
        raise ValueError("total_k must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")

    payload = load_json_payload(args.input_file)
    if not isinstance(payload, dict) or "qa_samples" not in payload:
        raise ValueError(f"{args.input_file} is not a GRIP task file")
    relation_order = load_train_relation_order(args.raw_dir)
    alias_index = train_relation_alias_index(relation_order)
    known_relations = known_pair_relations(args.raw_dir)
    model, tokenizer = load_b1(args)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = (
        load_completed_ids(args.output) if args.resume and args.output.is_file() else set()
    )
    mode = "a" if args.resume else "w"
    stats = {"processed": 0, "rollout_errors": 0, "rollout_oov": 0, "fallback": 0}
    started = time.time()

    with args.output.open(mode, encoding="utf-8") as stream:
        for index, text in enumerate(payload["qa_samples"]):
            question_id = f"task_qa:{index}"
            if question_id in completed_ids:
                continue
            gold = assistant_gold(text)
            if not is_relation_gold(gold, text):
                continue
            matched = match_train_relation(gold, alias_index)
            if matched is None:
                continue
            if args.limit and stats["processed"] >= args.limit:
                break

            raw, parsed = generate_one(
                model, tokenizer, assistant_answer_prefix(text), args.max_new_tokens
            )
            pair = question_entity_pair(text)
            excluded = known_relations.get(pair, set()) if pair else set()
            predicted = canonical_candidate(parsed)
            hard: list[str] = []
            excluded_canonical = {
                rel if rel.startswith("concept:") else f"concept:{rel}"
                for rel in excluded
            }
            predicted_canonical = (
                predicted if predicted.startswith("concept:") else f"concept:{predicted}"
            ) if predicted else ""
            if (
                predicted
                and predicted_canonical not in excluded_canonical
                and predicted != gold
                and predicted != matched
            ):
                hard = [predicted]
                stats["rollout_errors"] += 1
                if predicted not in relation_order:
                    stats["rollout_oov"] += 1

            pair = question_entity_pair(text)
            excluded = known_relations.get(pair, set()) if pair else set()
            pool = [
                rel
                for rel in relation_order
                if rel != matched and rel not in excluded and rel not in hard
            ]
            uniform_k = max(0, args.total_k - len(hard))
            uniform = rng.sample(pool, min(uniform_k, len(pool)))
            negatives = [*hard, *uniform]
            if len(negatives) < args.total_k:
                stats["fallback"] += 1

            row = {
                "question_id": question_id,
                "split": "train",
                "positive_relation": gold,
                "matched_train_relation": matched,
                "negative_relations": negatives,
                "hard_negative_relations": hard,
                "uniform_negative_relations": uniform,
                "rollout_raw": raw,
                "rollout_parsed": parsed,
                "rollout_hard_negative": predicted or None,
                "rollout_hard_is_train_vocab": predicted in relation_order if predicted else False,
                "known_pair_relations": sorted(excluded),
                "entity_pair": list(pair) if pair else None,
                "teacher": "frozen_b1_free_generation",
                "negative_source": "b1_rollout_error_plus_uniform",
                "seed": args.seed,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            completed_ids.add(question_id)
            stats["processed"] += 1
            if args.progress_every > 0 and (
                stats["processed"] == 1
                or stats["processed"] % args.progress_every == 0
            ):
                elapsed = time.time() - started
                print(
                    f"[mine] {stats['processed']} items errors={stats['rollout_errors']} "
                    f"oov={stats['rollout_oov']} elapsed={elapsed / 3600:.2f}h",
                    flush=True,
                )

    stats["seconds"] = round(time.time() - started, 1)
    stats["output"] = str(args.output)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
