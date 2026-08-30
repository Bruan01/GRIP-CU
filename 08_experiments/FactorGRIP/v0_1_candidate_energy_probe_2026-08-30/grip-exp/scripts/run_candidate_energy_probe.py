"""Run the inference-only FactorGRIP v0.1 candidate-energy probe."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Permit execution as ``python scripts/run_candidate_energy_probe.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidate_energy.artifacts import (  # noqa: E402
    append_jsonl,
    completed_question_ids,
    make_run_id,
    prepare_run_dir,
    runtime_environment,
    write_environment,
)
from candidate_energy.probe import (  # noqa: E402
    _ADAPTER_NAME,
    _ADAPTER_TRAIN_K,
    _adapter_context,
    _build_eval_dataset,
    _input_device,
    _peak_memory,
    _reset_peak_memory,
    _run_constrained_decoder,
    _run_free_decoder,
    _argmax_from_dicts,
)
from candidate_energy.scoring import score_candidates_batched  # noqa: E402
from evaluation.recurrent_metrics import exact_match  # noqa: E402
from grip.recurrent import build_recurrent_peft_model, set_recurrent_depth  # noqa: E402
from utils import load_list_json, set_random_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--run_root", default="../results/runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_name", default="qwen-0.5b")
    parser.add_argument("--model_source", choices=("auto", "local", "hf", "modelscope"), default="local")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--local_files_only", "--local-files-only", dest="local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_k1", required=True)
    parser.add_argument("--adapter_k2", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenize_max_length", type=int, default=4096)
    parser.add_argument("--gen_max_length", type=int, default=24)
    parser.add_argument("--max_total_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--adapter_controls", nargs="+", default=["correct", "none", "wrong_depth"], choices=["correct", "none", "wrong_depth"])
    parser.add_argument("--decoders", nargs="+", default=["free", "constrained", "score"], choices=["free", "constrained", "score"])
    return parser.parse_args()


def _max_answer_tokens(tokenizer, samples: list[dict]) -> int:
    lengths = []
    for sample in samples:
        for candidate in sample.get("candidate_relations", []):
            lengths.append(len(tokenizer.encode(f"<answer>{candidate}</answer>", add_special_tokens=False)))
    return max(lengths, default=1)


def _load_model(args: argparse.Namespace):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model, tokenizer, _ = build_recurrent_peft_model(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
        dtype=args.dtype,
        recurrent_depth_train=2,
        adapter_name="scratch",
        device_map=None,
    )
    adapter_paths = {"correct": Path(args.adapter_k1), "wrong_depth": Path(args.adapter_k2)}
    for control in args.adapter_controls:
        if control == "none":
            continue
        path = adapter_paths[control]
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing {control} adapter_config.json: {path}")
        model.load_adapter(str(path), adapter_name=_ADAPTER_NAME[control], is_trainable=False)
    model.to(device)
    set_recurrent_depth(model, 1)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer, device


def run(args: argparse.Namespace, run_dir: Path) -> dict:
    repository_dir = ROOT.parents[3]
    environment = runtime_environment(repository_dir)
    write_environment(run_dir / "environment.txt", environment)
    config = {"run_id": run_dir.name, "experiment": "FactorGRIP v0.1 candidate-energy probe", "input_file": str(Path(args.input_file).resolve()), "args": vars(args), "environment": environment, "adapter_controls": args.adapter_controls, "decoder_types": args.decoders, "shuffled_control": "unavailable: single NELL23K graph; wrong_depth is available"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    input_rows = load_list_json(args.input_file)
    if len(input_rows) != 1:
        raise ValueError(f"expected one graph record, found {len(input_rows)}")
    set_random_seed(args.seed)
    model, tokenizer, device = _load_model(args)
    record = input_rows[0]
    samples, dataset = _build_eval_dataset(record, tokenizer)
    if args.max_questions > 0:
        samples, dataset = samples[: args.max_questions], dataset
        # GRIPEvalDataset indexes its original arrays; use a bounded wrapper below.
    sample_indices = list(range(min(len(samples), len(dataset))))
    existing_predictions = completed_question_ids(run_dir / "predictions.jsonl")
    existing_scores = set()
    score_rows: dict[tuple[str, str, str], dict] = {}
    score_path = run_dir / "candidate_scores.jsonl"
    if score_path.exists():
        with score_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    key = (str(row["question_id"]), str(row["split"]), str(row["adapter_control"]))
                    existing_scores.add(key)
                    score_rows[key] = row
    pred_path = run_dir / "predictions.jsonl"
    started = time.monotonic()
    constrained_max_new = max(args.gen_max_length, _max_answer_tokens(tokenizer, samples) + 4)
    for control in args.adapter_controls:
        with _adapter_context(model, control):
            train_k = _ADAPTER_TRAIN_K[control]
            for idx in sample_indices:
                sample = samples[idx]
                question_id, split = str(sample["question_id"]), str(sample["split"])
                target = [str(sample["answer"])] if isinstance(sample["answer"], str) else [str(x) for x in sample["answer"]]
                candidates = [str(x) for x in sample.get("candidate_relations", [])]
                if len(candidates) != 10:
                    raise ValueError(f"question {question_id} must have exactly 10 candidates, found {len(candidates)}")
                input_ids, _question, _answer = dataset[idx]
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)
                base = {"question_id": question_id, "split": split, "target": target, "candidates": candidates, "adapter_control": control, "recurrent_train_k": train_k, "eval_k": 1}

                score_key = (question_id, split, control)
                score_dicts = None
                if score_key not in existing_scores:
                    _reset_peak_memory(device)
                    t0 = time.perf_counter()
                    scores = score_candidates_batched(model, tokenizer, input_ids[0].tolist(), candidates, device, max_total_length=args.max_total_length)
                    score_latency = time.perf_counter() - t0
                    score_dicts = [{"candidate": s.candidate, "sum_logprob": s.sum_logprob, "num_tokens": s.num_tokens, "norm_logprob": s.norm_logprob} for s in scores]
                    best_idx, _ = _argmax_from_dicts(score_dicts, True)
                    permuted = list(reversed(score_dicts))
                    perm_idx, _ = _argmax_from_dicts(permuted, True)
                    permuted_prediction = permuted[perm_idx]["candidate"] if perm_idx >= 0 else ""
                    score_row = {**base, "latency_seconds": score_latency, "peak_memory_bytes": _peak_memory(device), "scores": score_dicts, "candidate_order": candidates, "permutation": list(reversed(candidates)), "permutation_predicted_candidate": permuted_prediction, "order_permutation_stable": (candidates[best_idx] if best_idx >= 0 else "") == permuted_prediction}
                    append_jsonl(score_path, score_row)
                    existing_scores.add(score_key)
                    score_rows[score_key] = score_row
                else:
                    score_dicts = score_rows[score_key]["scores"]
                if score_dicts is None:
                    raise RuntimeError(f"score record unavailable for {score_key}")

                order_stable = bool(score_rows.get(score_key, {}).get("order_permutation_stable", True))

                def save_prediction(decoder: str, predicted: str, raw: str | None, latency: float, peak: int, summary: dict | None = None) -> None:
                    key = (question_id, split, control, decoder)
                    if key in existing_predictions:
                        return
                    metadata = {"graph_id": record.get("id", "nell23k"), "question": sample["question"], "response_in_candidates": predicted in candidates, "order_permutation_stable": order_stable}
                    row = {**base, "decoder_type": decoder, "predicted_candidate": predicted, "raw_response": raw, "response": predicted, "correct": exact_match(predicted, target), "latency_seconds": latency, "peak_memory_bytes": peak, "candidate_score_summary": summary, "metadata": metadata}
                    append_jsonl(pred_path, row)
                    existing_predictions.add(key)

                if "free" in args.decoders:
                    _reset_peak_memory(device); t0 = time.perf_counter()
                    parsed, raw = _run_free_decoder(model, tokenizer, input_ids, attention_mask, args.gen_max_length)
                    save_prediction("free", parsed, raw, time.perf_counter() - t0, _peak_memory(device))
                if "constrained" in args.decoders:
                    _reset_peak_memory(device); t0 = time.perf_counter()
                    parsed, raw = _run_constrained_decoder(model, tokenizer, input_ids, attention_mask, candidates, constrained_max_new)
                    save_prediction("constrained", parsed, raw, time.perf_counter() - t0, _peak_memory(device))
    # Select normalization on validation only, then freeze it for every split.
    normalizations = {"length_normalized": True, "raw_sum": False}
    validation_rows = [row for row in score_rows.values() if row.get("split") == "validation"]
    normalized_scores = []
    for name, normalized in normalizations.items():
        correct = 0
        for row in validation_rows:
            idx_best, _ = _argmax_from_dicts(row.get("scores", []), normalized)
            pred = row["scores"][idx_best]["candidate"] if idx_best >= 0 else ""
            correct += int(exact_match(pred, row.get("target", [])))
        normalized_scores.append((correct / len(validation_rows) if validation_rows else 0.0, name))
    selected_normalization = max(normalized_scores, key=lambda item: (item[0], item[1] == "length_normalized"))[1] if normalized_scores else "length_normalized"
    for row in score_rows.values():
        if "score" not in args.decoders:
            continue
        key = (str(row["question_id"]), str(row["split"]), str(row["adapter_control"]), "score")
        if key in existing_predictions:
            continue
        idx_best, best_score = _argmax_from_dicts(row.get("scores", []), normalizations[selected_normalization])
        predicted = row["scores"][idx_best]["candidate"] if idx_best >= 0 else ""
        stable = bool(row.get("order_permutation_stable", True))
        append_jsonl(pred_path, {**row, "decoder_type": "score", "predicted_candidate": predicted, "raw_response": None, "response": predicted, "correct": exact_match(predicted, row.get("target", [])), "latency_seconds": row.get("latency_seconds", 0.0), "peak_memory_bytes": row.get("peak_memory_bytes", 0), "candidate_score_summary": {"normalization": selected_normalization, "best_index": idx_best, "best_score": best_score, "scores": row.get("scores", []), "order_permutation_stable": stable}, "metadata": {"graph_id": record.get("id", "nell23k"), "response_in_candidates": predicted in row.get("candidates", []), "order_permutation_stable": stable}})
        existing_predictions.add(key)
    config["selected_normalization"] = selected_normalization
    config["question_count"] = len(sample_indices)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(json.dumps({"elapsed_seconds": time.monotonic() - started, "status": "complete"}, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    run_id = args.run_id or make_run_id()
    run_dir = prepare_run_dir(run_root, run_id, resume=args.resume)
    config = run(args, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "run_id": config["run_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
