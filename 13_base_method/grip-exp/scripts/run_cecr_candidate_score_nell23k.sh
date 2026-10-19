#!/usr/bin/env bash
# CECR E1: candidate-constrained teacher-forced decoding on the direct-format split.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
ADAPTER_DIR="${ADAPTER_DIR:-artifacts/nell23k_paper_7b/lora_adapter}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-model_cache}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/grip_inf/nell23k/nell23k_qwen7b_cecr_candidate_score_seed2026.jsonl}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
RELATION_CARD_MODE="${RELATION_CARD_MODE:-none}"
MEMORY_PRIOR_WEIGHT="${MEMORY_PRIOR_WEIGHT:-0.0}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.0}"
COUNTERFACTUAL_WEIGHT="${COUNTERFACTUAL_WEIGHT:-0.0}"

exec "$PYTHON_EXECUTABLE" scripts/infer_eval_nell23k.py \
  --input-file outputs/data/nell23k/processed_test_direct.json \
  --adapter-dir "$ADAPTER_DIR" \
  --output-file "$OUTPUT_FILE" \
  --seed 2026 \
  --model-name qwen-7b \
  --model-source local \
  --model-cache-dir "$MODEL_CACHE_DIR" \
  --local-files-only \
  --tokenize-max-length 4096 \
  --decoding-mode candidate_score \
  --train-triples-file outputs/data/nell23k/train.txt \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --memory-prior-weight "$MEMORY_PRIOR_WEIGHT" \
  --evidence-weight "$EVIDENCE_WEIGHT" \
  --counterfactual-weight "$COUNTERFACTUAL_WEIGHT" \
  --relation-card-mode "$RELATION_CARD_MODE"
