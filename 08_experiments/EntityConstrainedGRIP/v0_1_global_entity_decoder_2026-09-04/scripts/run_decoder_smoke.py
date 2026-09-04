#!/usr/bin/env python3
"""Run validation-only D0 artifact reuse and graph-free constrained decoders."""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from entity_decoder.artifacts import import_d0_predictions
from entity_decoder.candidates import build_diagnostic_candidates
from entity_decoder.checkpoints import resolve_checkpoint_registry
from entity_decoder.composition import prefix_ambiguity
from entity_decoder.config import ALLOWED_PROMPTS, load_config
from entity_decoder.io_utils import environment_snapshot, sha256_file, write_json, write_jsonl
from entity_decoder.metrics import compare_decoder_predictions, score_prediction_rows
from entity_decoder.records import annotate_rows, load_split
from entity_decoder.runtime import decode_trie, load_runtime, tokenize_entities
from entity_decoder.scoring import GlobalEntityScorer, score_candidate_rows, score_rows, select_normalization
from entity_decoder.trie import EntityTokenTrie
from entity_decoder.vocabulary import audit_answer_coverage, read_entities


def _checkpoint(config: dict, name: str, require_files: bool = True) -> dict:
    registry = resolve_checkpoint_registry(config["checkpoints"], REPO, require_files=require_files)
    matches = [entry for entry in registry if entry["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"checkpoint not registered exactly once: {name}")
    return matches[0]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_memory(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _selected_d2_rows(rows: list[dict], normalization: str) -> list[dict]:
    return [
        {
            **{key: value for key, value in row.items() if key not in {"sum", "mean"}},
            "prediction_text": row[normalization]["prediction"],
            "canonical_prediction": row[normalization]["prediction"],
            "gold_rank": row[normalization]["rank"],
            "gold_score": row[normalization]["gold_score"],
            "top_entities": row[normalization]["top_entities"],
            "score_normalization": normalization,
        }
        for row in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase_a_decoder_smoke.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name-or-path")
    ap.add_argument("--decoders", nargs="+", choices=("D0", "D1", "D2", "D3"))
    ap.add_argument("--prompt-protocol", choices=ALLOWED_PROMPTS, help="Prompt-robustness ablation override.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    config = load_config(config_path)
    checkpoint = _checkpoint(config, args.checkpoint)
    protocol = args.prompt_protocol or checkpoint["prompt_protocol"]
    decoders = args.decoders or list(config["decoders"])
    splits_to_run = list(config["evaluation"]["splits"])
    if splits_to_run != ["validation"]:
        raise ValueError("E03 Phase-A runner is validation-only")
    if "D0" in decoders and args.prompt_protocol:
        raise ValueError("D0 artifact reuse forbids prompt override")
    if "D0" in decoders and "validation" not in checkpoint.get("d0_predictions", {}):
        raise ValueError(f"checkpoint has no registered validation D0 artifact: {checkpoint['name']}")

    _set_seed(42)
    entities_path = ROOT / "artifacts/entities_train_graph.jsonl"
    entities = read_entities(entities_path)
    raw_splits = {
        "train": load_split(REPO / config["data"]["train"], "train"),
        "validation": load_split(REPO / config["data"]["validation"], "validation"),
    }
    # The Phase-A process intentionally never opens the registered test file.
    model, tokenizer, device, dtype, injection, resolved_model = load_runtime(
        config, checkpoint, args.model_name_or_path
    )
    entity_tokens = tokenize_entities(entities, tokenizer)
    trie = EntityTokenTrie(entity_tokens)
    token_sequences = list(entity_tokens.values())
    prefix_width = int(config["evaluation"].get("prefix_ambiguity_token_width", 2))
    answers = {str(row["answer"]) for rows in raw_splits.values() for row in rows}
    answer_lengths = {answer: len(entity_tokens[answer]) for answer in answers}
    ambiguities = {
        answer: prefix_ambiguity(entity_tokens[answer], token_sequences, prefix_width)
        for answer in answers
    }
    splits = {
        "validation": annotate_rows(
            raw_splits["train"], raw_splits["validation"],
            answer_token_lengths=answer_lengths,
            prefix_ambiguities=ambiguities,
        )
    }

    runtime_audit = {
        "environment": environment_snapshot(REPO),
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "checkpoint": checkpoint,
        "model": resolved_model,
        "prompt_protocol": protocol,
        "decoders": decoders,
        "evaluation_splits_opened": splits_to_run,
        "test_file_opened": False,
        "entity_vocabulary": {
            "path": str(entities_path.relative_to(REPO)),
            "sha256": sha256_file(entities_path),
            "count": len(entities),
            "validation_answer_coverage": audit_answer_coverage(entities, raw_splits["validation"]),
            "max_entity_token_length": max(map(len, entity_tokens.values())),
            "tokenization_collision_free": True,
        },
        "inference_graph_access": False,
        "query_specific_candidate_access": "D3_diagnostic_only" if "D3" in decoders else False,
        "gold_path_in_prompt": False,
        "injection": None if injection is None else {
            "target_modules": list(injection.target_modules),
            "replaced_module_count": len(injection.replaced_modules),
            "trainable_parameters": injection.trainable_parameters,
            "total_parameters": injection.total_parameters,
        },
    }
    write_json(output_dir / "runtime_audit.json", runtime_audit)

    summary = {"format_version": 1, **runtime_audit, "results": {}, "mechanism_probes": {}}
    predictions_by_decoder: dict[str, dict[str, list[dict]]] = {}
    for decoder in decoders:
        summary["results"][decoder] = {}
        predictions_by_decoder[decoder] = {}
        scorer = None
        if decoder == "D2":
            scorer = GlobalEntityScorer(
                model, tokenizer, entities, entity_tokens, device=device, dtype=dtype,
                chunk_size=int(config["evaluation"]["d2_chunk_size"]),
            )
        for split in splits_to_run:
            _reset_memory(device)
            rows = splits[split]
            if decoder == "D0":
                predictions, artifact_audit = import_d0_predictions(
                    checkpoint["d0_predictions"][split], REPO, rows
                )
                metrics = score_prediction_rows(predictions, entities)
                runtime = {
                    "mode": "frozen_artifact_reuse",
                    "model_inference_performed": False,
                    "elapsed_seconds": 0.0,
                    "artifact": artifact_audit,
                }
            elif decoder == "D1":
                predictions, runtime = decode_trie(
                    model, tokenizer, rows, protocol, trie,
                    beam_size=int(config["evaluation"]["trie_beam_size"]), device=device,
                )
                metrics = score_prediction_rows(predictions, entities)
            elif decoder == "D2":
                assert scorer is not None
                raw_rankings, ranking = score_rows(
                    scorer, rows, protocol, top_k=int(config["evaluation"]["d2_top_k"]),
                )
                write_jsonl(output_dir / f"rankings_{decoder}_{split}.jsonl", raw_rankings)
                selected_normalization = select_normalization(
                    ranking, list(config["evaluation"]["d2_score_normalization_candidates"])
                )
                predictions = _selected_d2_rows(raw_rankings, selected_normalization)
                metrics = score_prediction_rows(predictions, entities)
                metrics["ranking"] = ranking
                metrics["selected_normalization"] = selected_normalization
                runtime = {
                    "elapsed_seconds": ranking["elapsed_seconds"],
                    "latency_seconds_per_query": ranking["latency_seconds_per_query"],
                    "peak_gpu_memory_bytes": __import__("torch").cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                }
            else:
                builder = lambda query: build_diagnostic_candidates(
                    query, raw_splits["train"],
                    distractor_count=int(config["evaluation"]["d3_distractor_count"]),
                    seed=int(config["evaluation"]["d3_seed"]),
                )
                raw_rankings, ranking = score_candidate_rows(
                    model, tokenizer, rows, protocol, builder, entity_tokens,
                    device=device, dtype=dtype,
                    chunk_size=int(config["evaluation"]["d3_distractor_count"]) + 1,
                    top_k=int(config["evaluation"]["d3_distractor_count"]) + 1,
                )
                write_jsonl(output_dir / f"rankings_{decoder}_{split}.jsonl", raw_rankings)
                selected_normalization = select_normalization(
                    ranking, list(config["evaluation"]["d2_score_normalization_candidates"])
                )
                predictions = _selected_d2_rows(raw_rankings, selected_normalization)
                metrics = score_prediction_rows(predictions, entities)
                metrics["ranking"] = ranking
                metrics["selected_normalization"] = selected_normalization
                metrics["diagnostic_only"] = True
                runtime = {
                    "elapsed_seconds": ranking["elapsed_seconds"],
                    "latency_seconds_per_query": ranking["latency_seconds_per_query"],
                    "peak_gpu_memory_bytes": __import__("torch").cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                }
            predictions_by_decoder[decoder][split] = predictions
            write_jsonl(output_dir / f"predictions_{decoder}_{split}.jsonl", predictions)
            summary["results"][decoder][split] = {"metrics": metrics, "runtime": runtime}
            print(
                f"{checkpoint['name']} {decoder} {split}: "
                f"raw={metrics['raw_exact_match']:.6f} "
                f"canonical={metrics['canonical_entity_exact_match']:.6f} "
                f"valid={metrics['valid_entity_rate']:.6f}"
            )

    if "D0" in predictions_by_decoder and "D1" in predictions_by_decoder:
        transitions = compare_decoder_predictions(
            predictions_by_decoder["D0"]["validation"],
            predictions_by_decoder["D1"]["validation"],
            entities,
        )
        summary["mechanism_probes"]["d0_to_d1_validation_transitions"] = transitions
        write_json(output_dir / "d0_to_d1_validation_transitions.json", transitions)

    write_json(output_dir / "run_summary.json", summary)
    print(f"DECODER_SMOKE_COMPLETE checkpoint={checkpoint['name']} output={output_dir}")


if __name__ == "__main__":
    main()
