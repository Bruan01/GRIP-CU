"""Zero-training H2 gate: score candidate continuations under an existing adapter.

This script does NOT train anything. It loads the already-trained NELL23K
graph adapter produced by the RecurrentGRIP pilot run and scores, for each
question in the candidate audit, the normalized log-likelihood of the positive
relation and every generated negative relation continuation.

``--control correct`` scores under the trained graph adapter; ``--control none``
scores under the base language model with the adapter disabled. Comparing the
two separates "the negative family design is wrong" from "the adapter did not
learn the graph".

Output answers one question: are the structure-aware negative families
(tail_range, path_local) *harder* than the controls (uniform, random)?
Harder means the model assigns the wrong relation a higher continuation score,
i.e. it is more confusable with the positive answer.

The prompt prefix (system + user question + assistant "<answer>") is identical
across all candidates of a question, so any score difference is attributable to
the relation continuation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

# --- Paths -------------------------------------------------------------------
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
HNG = Path("/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP")
ADAPTER_DIR = (
    VERSION_DIR
    / "results/runs/pilot_20260902_031653_nell23k_qwen05b_pilot/adapters/nell23k"
)
PREPARED = GRIP_EXP / "outputs/data/nell23k/recurrent_relation_prediction.json"
AUDIT = HNG / "results_nell23k_audit.json"
OUTPUT = HNG / "results" / "h2_gate_results.json"

sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import SYSTEM_PROMPT  # noqa: E402
from grip.recurrent import build_recurrent_peft_model  # noqa: E402
from hard_negative_grip.scoring import normalized_continuation_log_likelihood  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prefix(tokenizer, title: str, question: str) -> torch.Tensor:
    """Tokenize the question prefix ending right after the literal '<answer>'."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_str = base + "<answer>"
    return torch.tensor(
        tokenizer(prefix_str, add_special_tokens=False).input_ids, dtype=torch.long
    )


def score_question(
    model,
    tokenizer,
    prefix_ids: torch.Tensor,
    relations: list[str],
    device: torch.device,
) -> list[float]:
    """Return normalized continuation log-likelihood for each relation string."""
    rows: list[list[int]] = []
    sequence_lengths: list[int] = []
    for relation in relations:
        cont = tokenizer(relation, add_special_tokens=False).input_ids
        if not cont:
            cont = [tokenizer.unk_token_id or 0]
        rows.append((prefix_ids.tolist() + cont))
        sequence_lengths.append(len(prefix_ids) + len(cont))

    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[i, : len(row)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prefix_lengths = torch.full((len(rows),), len(prefix_ids), dtype=torch.long, device=device)
    sequence_lengths_t = torch.tensor(sequence_lengths, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    scores = normalized_continuation_log_likelihood(
        logits, input_ids, prefix_lengths, sequence_lengths_t
    )
    return scores.detach().float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=["correct", "none"], default="correct")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate] loading model on {device} (control={args.control}) ...", flush=True)
    model, tokenizer, _ = build_recurrent_peft_model(
        model_name="qwen-0.5b",
        model_source="auto",
        model_cache_dir=str(GRIP_EXP / "model_cache"),
        local_files_only=False,
        dtype="bfloat16",
        lora_r=4,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        executor_layer_index=12,
        recurrent_depth_train=2,
        adapter_name="scratch",
    )
    model.load_adapter(str(ADAPTER_DIR), adapter_name="eval_correct", is_trainable=False)
    adapter_context = nullcontext()
    if args.control == "correct":
        model.set_adapter("eval_correct")
    else:
        adapter_context = model.disable_adapter()
    model.to(device)
    model.eval()
    print("[gate] model + adapter loaded", flush=True)

    prepared = load_jsonl(PREPARED)[0]["recurrent_questions"]
    question_map = {q["question_id"]: q for q in prepared}
    audit_records = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_rows = audit_records[0]["rows"] if isinstance(audit_records, list) else audit_records

    title = "nell23k"
    rows_out = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    family_margins: dict[str, list[float]] = defaultdict(list)
    family_hits1: dict[str, list[bool]] = defaultdict(list)

    with adapter_context:
        for i, row in enumerate(audit_rows):
            qid = row["question_id"]
            q = question_map.get(qid)
            if q is None:
                continue
            positive = row["positive_relation"]
            hard = row["hard_candidates"]
            random_cands = row["random_candidates"]

            prefix_ids = build_prefix(tokenizer, title, q["question"])

            def unique_rels(cands):
                seen, out = set(), []
                for c in cands:
                    if c["relation"] not in seen and c["relation"] != positive:
                        seen.add(c["relation"])
                        out.append(c["relation"])
                return out

            family_rels: dict[str, list[str]] = {
                "uniform_relation": unique_rels([c for c in hard if c["kind"] == "uniform_relation"]),
                "tail_range_relation": unique_rels([c for c in hard if c["kind"] == "tail_range_relation"]),
                "path_relation": unique_rels([c for c in hard if c["kind"] == "path_relation"]),
                "random": unique_rels(random_cands),
            }

            all_rels = [positive] + [r for rels in family_rels.values() for r in rels]
            scores = score_question(model, tokenizer, prefix_ids, all_rels, device)
            positive_score = scores[0]

            score_index: dict[str, float] = {}
            cursor = 1
            for family, rels in family_rels.items():
                for rel in rels:
                    score_index[(family, rel)] = scores[cursor]
                    cursor += 1

            record = {
                "question_id": qid,
                "split": q.get("split"),
                "positive_relation": positive,
                "positive_score": positive_score,
                "families": {},
            }
            for family, rels in family_rels.items():
                neg_scores = [score_index[(family, rel)] for rel in rels]
                if not neg_scores:
                    record["families"][family] = {"count": 0}
                    continue
                mean_neg = sum(neg_scores) / len(neg_scores)
                margin = positive_score - mean_neg
                hits1 = all(positive_score > s for s in neg_scores)
                family_scores[family].append(mean_neg)
                family_margins[family].append(margin)
                family_hits1[family].append(hits1)
                record["families"][family] = {
                    "count": len(neg_scores),
                    "mean_negative_score": mean_neg,
                    "margin_positive_minus_negative": margin,
                    "hits1": hits1,
                    "negative_scores": neg_scores,
                }
            rows_out.append(record)

            if (i + 1) % 40 == 0:
                print(f"[gate] scored {i + 1}/{len(audit_rows)} questions", flush=True)

    summary: dict[str, dict] = {}
    order = ["uniform_relation", "tail_range_relation", "path_relation", "random"]
    for family in order:
        scores = family_scores.get(family, [])
        margins = family_margins.get(family, [])
        hits = family_hits1.get(family, [])
        if not scores:
            summary[family] = {"count": 0}
            continue
        summary[family] = {
            "count": len(scores),
            "mean_negative_score": sum(scores) / len(scores),
            "mean_margin": sum(margins) / len(margins),
            "hits1": sum(hits) / len(hits),
        }

    result = {
        "adapter_dir": str(ADAPTER_DIR),
        "control": args.control,
        "device": str(device),
        "questions_scored": len(rows_out),
        "note": "Harder negatives have HIGHER mean_negative_score and LOWER margin/hits1.",
        "family_summary": summary,
        "per_question": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[gate] wrote", args.output, flush=True)
    print("\n=== H2 family summary ===", flush=True)
    for family in order:
        s = summary.get(family, {})
        if not s or "mean_negative_score" not in s:
            print(f"{family:22s} (no candidates)", flush=True)
            continue
        print(
            f"{family:22s} neg_score={s['mean_negative_score']:.4f} "
            f"margin={s['mean_margin']:.4f} hits1={s['hits1']:.4f} n={s['count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
