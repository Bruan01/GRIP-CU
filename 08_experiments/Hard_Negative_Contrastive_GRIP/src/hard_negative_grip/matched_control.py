"""Comparison guards and paper exports for the single-seed random control."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

MATCHED_FIELDS = (
    "dataset", "model_name", "dtype", "lora_r", "lora_alpha", "target_modules",
    "num_train_epochs", "involve_qa_epochs", "s1_stop_loss_threshold",
    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "adam_epsilon", "max_grad_norm", "lambda_candidate",
    "memory_size", "temperature", "gen_max_length", "seed", "listed_negative_k",
    "stop_after_step", "s1_gradient_checkpointing",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"refusing empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parameter_diff(reference: dict, actual: dict) -> list[dict]:
    rows = []
    for field in MATCHED_FIELDS:
        if field not in reference or field not in actual:
            raise ValueError(f"missing matched training field {field}")
        left, right = reference[field], actual[field]
        equal = set(left) == set(right) if field == "target_modules" else left == right
        rows.append({"parameter": field, "reference": left, "random_control": right, "equal": equal})
    if not all(row["equal"] for row in rows):
        raise ValueError("non-sampling training configuration changed: "+
                         ", ".join(row["parameter"] for row in rows if not row["equal"]))
    if reference.get("training_arguments") != actual.get("training_arguments"):
        raise ValueError("training_arguments differ from the parent")
    return rows


def training_command(python: str, script: Path, reference: dict, *, inputs: Path,
                     bank: Path, output: Path, model_cache: Path) -> list[str]:
    cmd = [python, str(script), "--stage", "listed", "--input_file", str(inputs/"task.json"),
           "--eval_file", str(inputs/"validation.json"), "--output_dir", str(output),
           "--s1_adapter", str(inputs/"stage1_adapter"), "--raw_dir", str(inputs/"raw"),
           "--relation_text_file", str(inputs/"raw/rel2text.json"),
           "--relation_eligibility", str(inputs/"relation_eligibility.json"),
           "--listed_negative_source", "frozen_random", "--negative_sampling_mode", "random",
           "--random_manifest", str(bank), "--listed_negative_seed", str(reference["seed"]),
           "--model_cache_dir", str(model_cache), "--save_steps", "8", "--save_total_limit", "4",
           "--defer_eval", "--swanlab_mode", "offline"]
    for field in MATCHED_FIELDS:
        value = reference[field]
        if field == "target_modules":
            cmd.extend(["--target_modules", *value])
        elif field == "s1_gradient_checkpointing":
            if value:
                cmd.append("--s1_gradient_checkpointing")
        else:
            cmd.extend(["--"+field, str(value)])
    return cmd


def audit_predictions(rows: list[dict], samples: list[dict], normalize) -> dict[str, dict]:
    """Reject stale, duplicate, incomplete, relabelled, or mismatched predictions."""
    if len(rows) != len(samples):
        raise ValueError("incomplete prediction cohort")
    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {r["question_id"] for r in samples}:
        raise ValueError("duplicate or mismatched prediction IDs")
    for sample in samples:
        row = indexed[sample["question_id"]]
        question = sample["question"]
        question = str(question[0]) if isinstance(question, (list, tuple)) else str(question)
        answer = sample["answer"]
        targets = [str(t) for t in answer] if isinstance(answer, list) else [str(answer)]
        correct = any(normalize(str(row["response"])) == normalize(t) for t in targets)
        if (row["question"] != question or row["target"] != targets or row["split"] != sample["split"]
                or type(row["correct"]) is not bool or row["correct"] != correct):
            raise ValueError(f"prediction prompt/target/flag mismatch: {sample['question_id']}")
    return indexed


def paired_result(reference: dict[str, dict], random_control: dict[str, dict]) -> tuple[dict, list[dict]]:
    if not reference or reference.keys() != random_control.keys():
        raise ValueError("paired result requires identical nonempty cohorts")
    pairs, counts = [], {"both_correct": 0, "rescue": 0, "regression": 0, "both_wrong": 0}
    for qid, left in reference.items():
        right = random_control[qid]
        for key in ("question", "target", "split"):
            if left[key] != right[key]:
                raise ValueError(f"paired {key} mismatch")
        # Direction is always SC-CL minus Random-9.
        sc, rand = left["correct"], right["correct"]
        outcome = "both_correct" if sc and rand else "rescue" if sc else "regression" if rand else "both_wrong"
        counts[outcome] += 1
        pairs.append({"question_id": qid, "sccl_correct": int(sc), "random9_correct": int(rand),
                      "outcome_sccl_vs_random9": outcome,
                      "question_sha256": hashlib.sha256(left["question"].encode()).hexdigest()})
    n = len(pairs)
    sc_correct = counts["both_correct"]+counts["rescue"]
    rand_correct = counts["both_correct"]+counts["regression"]
    net = sc_correct-rand_correct
    return {"n": n, "sccl_correct": sc_correct, "random9_correct": rand_correct,
            "sccl_em": sc_correct/n, "random9_em": rand_correct/n,
            "delta_sccl_minus_random9_pp": 100*net/n, "net_correct": net,
            "paired_counts": counts}, pairs
