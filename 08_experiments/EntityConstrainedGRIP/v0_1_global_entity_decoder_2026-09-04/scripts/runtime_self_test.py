#!/usr/bin/env python3
"""Fail-fast WSL CUDA/model/checkpoint/tokenizer integration check."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from entity_decoder.checkpoints import resolve_checkpoint_registry
from entity_decoder.config import load_config
from entity_decoder.records import build_prompt, load_split
from entity_decoder.runtime import load_runtime, tokenize_entities
from entity_decoder.scoring import GlobalEntityScorer, rank_entities
from entity_decoder.trie import EntityTokenTrie
from entity_decoder.vocabulary import read_entities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase_a_decoder_smoke.json")
    ap.add_argument("--checkpoint", default="direct_answer_only_seed43")
    ap.add_argument("--model-name-or-path")
    ap.add_argument("--d2-entity-count", type=int, default=16)
    ap.add_argument("--d2-chunk-size", type=int, default=4)
    args = ap.parse_args()
    config = load_config(args.config.resolve())
    registry = resolve_checkpoint_registry(config["checkpoints"], REPO, require_files=True)
    checkpoint = next(entry for entry in registry if entry["name"] == args.checkpoint)
    model, tokenizer, device, dtype, injection, model_name = load_runtime(config, checkpoint, args.model_name_or_path)
    entities = read_entities(ROOT / "artifacts/entities_train_graph.jsonl")
    sample = entities[:128]
    mapping = tokenize_entities(sample, tokenizer)
    trie = EntityTokenTrie(mapping)
    parameter_device = next(model.parameters()).device
    if parameter_device.type != "cuda" or device.type != "cuda":
        raise RuntimeError("runtime self-test requires CUDA")

    d2_count = int(args.d2_entity_count)
    d2_chunk_size = int(args.d2_chunk_size)
    if d2_count <= 0 or d2_count > len(sample):
        raise ValueError(f"d2-entity-count must be in [1, {len(sample)}]")
    if d2_chunk_size <= 0:
        raise ValueError("d2-chunk-size must be positive")
    d2_entities = sample[:d2_count]
    d2_mapping = {entity: mapping[entity] for entity in d2_entities}
    validation_path = REPO / config["data"]["validation"]
    validation_row = load_split(validation_path, "validation")[0]
    prompt = build_prompt(validation_row, checkpoint["prompt_protocol"])
    scorer = GlobalEntityScorer(
        model,
        tokenizer,
        d2_entities,
        d2_mapping,
        device=device,
        dtype=dtype,
        chunk_size=d2_chunk_size,
    )
    score_output = scorer.score_prompt(prompt)
    if len(score_output.sum_scores) != d2_count or len(score_output.mean_scores) != d2_count:
        raise RuntimeError("D2 score shape mismatch")
    if not all(math.isfinite(value) for value in score_output.sum_scores + score_output.mean_scores):
        raise RuntimeError("D2 produced a non-finite score")
    rank_probe = rank_entities(d2_entities, score_output.sum_scores, d2_entities[0], top_k=min(5, d2_count))
    if not 1 <= int(rank_probe["rank"]) <= d2_count:
        raise RuntimeError("D2 rank is outside the candidate range")

    print(
        f"WSL_RUNTIME_READY model={model_name} checkpoint={checkpoint['name']} "
        f"sample_entities={len(trie.token_sequences)} dtype={dtype} device={parameter_device} "
        f"d2_entities={d2_count} d2_chunk_size={d2_chunk_size} "
        f"d2_elapsed_seconds={score_output.elapsed_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
