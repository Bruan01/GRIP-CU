#!/usr/bin/env python3
"""Evaluate one frozen GRIP/LoRA adapter with resumable inference artifacts.

This wrapper deliberately reuses the project's existing evaluation code so
baseline and innovation runs share prompt construction, answer parsing, exact
match normalization, and candidate-valid accounting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HNG = Path(__file__).resolve().parents[1]
SCRIPTS = HNG / "scripts"
REPO_ROOT = HNG.parents[1]
GRIP_EXP = Path(
    os.environ.get(
        "GRIP_BASELINE_DIR", str(REPO_ROOT / "13_base_method" / "grip-exp")
    )
)
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))
sys.path.insert(0, str(SCRIPTS))

from constants import TORCH_DTYPE_NAME, HF_DECODER_ONLY_LLMS  # noqa: E402
import hpo_reuse as reuse  # noqa: E402
from hard_negative_grip.decode_io import (  # noqa: E402
    eval_samples,
    load_jsonl,
    merge_resumed_rows,
    write_jsonl,
)
from hard_negative_grip.task_file import load_graph_record  # noqa: E402
from train_listed_contrastive import (  # noqa: E402
    em_summary,
    evaluate_adapter,
    load_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--model_cache_dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=os.environ.get("GRIP_TORCH_DTYPE", TORCH_DTYPE_NAME),
    )
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--label", default="adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = load_graph_record(args.eval_file)
    samples = eval_samples(record)
    pred_path = args.output_dir / "predictions.jsonl"
    existing = load_jsonl(pred_path) if args.resume else []
    identity = reuse.inference_identity(args.adapter, args, record, HF_DECODER_ONLY_LLMS[args.model_name])
    identity_file = args.output_dir / 'reuse_identity.json'
    if identity_file.exists() and reuse.read(identity_file) != identity:
        raise ValueError('Evaluation weights, decoder, or source cohort changed')
    if existing and not identity_file.exists():
        raise ValueError('Existing predictions lack a full decoder/source identity; '
                         'an explicit audited migration is required')
    reuse.write(identity_file, identity)
    expected = {sample['question_id']: sample for sample in samples}
    for row in existing:
        sample = expected.get(row.get('question_id'))
        if sample is None or sample['question'] != row.get('question') or reuse.target_list(sample['answer']) != row.get('target'):
            raise ValueError('Resumed question ID refers to a different prompt or target')
    shared = reuse.reusable_predictions(identity, samples)
    # Local complete work takes priority over a shared subset.
    existing = list({row['question_id']: row for row in [*shared, *existing]}.values())
    kept, missing = merge_resumed_rows(samples, existing)
    write_jsonl(pred_path, kept)
    reuse.write(args.output_dir / 'reuse_stats.json', {
        'shared_rows_available': len(shared), 'completed_rows_before_inference': len(kept),
        'new_rows_required': len(missing), 'total_rows': len(samples)})

    model = tokenizer = None
    try:
        if missing:
            model, tokenizer = load_adapter(args, args.adapter, trainable=False)
            evaluate_adapter(
                model,
                tokenizer,
                record,
                args.gen_max_length,
                progress_every=args.progress_every,
                skip_question_ids={str(row["question_id"]) for row in kept},
                pred_path=pred_path,
            )
        rows, missing = merge_resumed_rows(samples, load_jsonl(pred_path))
        if missing:
            raise RuntimeError(f"evaluation incomplete: {len(missing)} questions missing")
        write_jsonl(pred_path, rows)
        reuse.remember_predictions(identity, rows)
        summary = em_summary(rows)
        summary["label"] = args.label
        summary["adapter"] = str(args.adapter)
        summary["eval_file"] = str(args.eval_file)
        summary["dtype"] = args.dtype
        summary["gen_max_length"] = args.gen_max_length
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "label": args.label,
                    "adapter": str(args.adapter),
                    "eval_file": str(args.eval_file),
                    "dtype": args.dtype,
                    "gen_max_length": args.gen_max_length,
                    "count": len(rows),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer


if __name__ == "__main__":
    main()
